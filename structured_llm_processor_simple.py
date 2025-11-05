import json
import os
from openai import OpenAI
from typing import Dict, Any, List
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

async def process_structured_data_with_llm_async(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process structured data with LLM to extract entities and add context"""
    
    document_text = structured_data.get('document_text', [])
    tables = structured_data.get('tables', [])
    key_values = structured_data.get('key_values', [])
    
    print(f"Processing {len(tables)} tables, {len(key_values)} key-values, {len(document_text)} text lines")
    
    # Extract entities from all sources
    all_entities = []
    
    # Process tables
    for table_idx, table in enumerate(tables):
        if 'rows' in table:
            for row in table['rows']:
                if len(row) >= 2:
                    entity = {
                        'field': str(row[0]).strip(),
                        'value': str(row[1]).strip(),
                        'source': f'Table {table_idx + 1}',
                        'type': 'Table Data'
                    }
                    all_entities.append(entity)
    
    # Process key-value pairs
    for kv in key_values:
        if 'key' in kv and 'value' in kv:
            entity = {
                'field': str(kv['key']).strip(),
                'value': str(kv['value']).strip(),
                'source': 'Key-Value Extraction',
                'type': 'Structured Data'
            }
            all_entities.append(entity)
    
    # Extract additional entities from document text using LLM
    if document_text:
        text_entities = await extract_entities_from_text(document_text)
        all_entities.extend(text_entities)
    
    # Add context to entities
    entities_with_context = await add_context_to_entities(all_entities, document_text)
    
    return {
        'enhanced_data_with_comprehensive_context': entities_with_context,
        'summary': {
            'total_entities': len(entities_with_context),
            'entities_with_context': len([e for e in entities_with_context if e.get('context')]),
            'processing_method': 'simplified_llm'
        }
    }

async def extract_entities_from_text(document_text: List[str]) -> List[Dict[str, Any]]:
    """Extract additional entities from document text using LLM"""
    
    # Join text and limit length for processing
    full_text = '\n'.join(document_text[:100])  # Limit to first 100 lines
    
    if len(full_text) < 100:
        return []
    
    prompt = f"""Extract key financial and business data points from this document text.

Text:
{full_text}

Extract data as field-value pairs. Focus on:
- Financial figures (revenue, profit, growth rates)
- Key metrics and KPIs
- Important dates and periods
- Company information
- Performance indicators

Return JSON format:
{{
  "entities": [
    {{"field": "Revenue_Q1_2023", "value": "$50.2M"}},
    {{"field": "Growth_Rate", "value": "15%"}},
    {{"field": "CEO_Name", "value": "John Smith"}}
  ]
}}"""

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            ))
        
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            entities = result.get('entities', [])
            
            # Format entities
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    'field': entity.get('field', 'Unknown'),
                    'value': entity.get('value', ''),
                    'source': 'Text Extraction',
                    'type': 'Financial Data'
                })
            
            print(f"Extracted {len(formatted_entities)} entities from text")
            return formatted_entities
    
    except Exception as e:
        print(f"Error extracting entities from text: {e}")
        return []

async def add_context_to_entities(entities: List[Dict[str, Any]], document_text: List[str]) -> List[Dict[str, Any]]:
    """Add context to entities using document text"""
    
    if not document_text:
        return entities
    
    full_text = '\n'.join(document_text)
    
    # Process entities in batches
    batch_size = 10
    enhanced_entities = []
    
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        batch_with_context = await process_entity_batch(batch, full_text)
        enhanced_entities.extend(batch_with_context)
    
    return enhanced_entities

async def process_entity_batch(entities: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
    """Process a batch of entities to add context"""
    
    # Create entity summary for the prompt
    entity_list = []
    for i, entity in enumerate(entities):
        entity_list.append(f"{i+1}. {entity.get('field', 'Unknown')}: {entity.get('value', '')}")
    
    entities_text = '\n'.join(entity_list)
    
    # Limit document text length
    text_preview = full_text[:3000] if len(full_text) > 3000 else full_text
    
    prompt = f"""For each entity below, find relevant context from the document text.

Entities:
{entities_text}

Document Text:
{text_preview}

For each entity, provide context that explains or describes it. Return JSON:
{{
  "contexts": [
    {{"entity_number": 1, "context": "relevant context from document"}},
    {{"entity_number": 2, "context": "relevant context from document"}}
  ]
}}

If no relevant context is found for an entity, use empty string."""

    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            ))
        
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            contexts = result.get('contexts', [])
            
            # Apply contexts to entities
            for context_item in contexts:
                entity_num = context_item.get('entity_number', 0)
                context_text = context_item.get('context', '')
                
                if 1 <= entity_num <= len(entities):
                    entities[entity_num - 1]['context'] = context_text
        
        print(f"Added context to {len(entities)} entities")
        return entities
    
    except Exception as e:
        print(f"Error adding context to entities: {e}")
        # Return entities without context
        for entity in entities:
            entity['context'] = ''
        return entities