#!/usr/bin/env python3
"""
Backend endpoints for the Prompt Tester frontend
"""

from flask import Flask, request, jsonify, send_file
import json
import io
import pandas as pd
import tempfile
import os
from typing import Dict, Any, List
import asyncio
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("⚠️  Warning: OPENAI_API_KEY not found in environment")
    print("🔑 Please set your OpenAI API key in the .env file")
else:
    print("✅ OpenAI API key found")

openai_client = openai.OpenAI(api_key=api_key)

def clean_csv_value(value: str) -> str:
    """Clean CSV value for proper formatting"""
    if not value:
        return ""
    
    # Convert to string and clean
    value_str = str(value).strip()
    
    # Escape quotes and handle commas
    if '"' in value_str:
        value_str = value_str.replace('"', '""')
    
    # Wrap in quotes if contains comma, newline, or quote
    if ',' in value_str or '\n' in value_str or '"' in value_str:
        value_str = f'"{value_str}"'
    
    return value_str

async def test_context_extraction_prompt(prompt_template: str, row_data: str, text_content: str) -> Dict[str, Any]:
    """Test the context extraction prompt with given data"""
    
    # Replace placeholders manually to avoid format string issues
    formatted_prompt = prompt_template.replace('{row_data}', row_data)
    formatted_prompt = formatted_prompt.replace('{text_content}', text_content)
    
    try:
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": formatted_prompt
                }],
                response_format={"type": "json_object"}
            )
        )
        
        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            return {
                "success": True,
                "result": result,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
        else:
            return {
                "success": False,
                "error": "No content received from OpenAI"
            }
            
    except Exception as e:
        print(f"Error in test_context_extraction_prompt: {e}")
        print(f"Prompt length: {len(formatted_prompt)}")
        return {
            "success": False,
            "error": str(e)
        }

def generate_sample_entities(text_content: str) -> List[Dict[str, Any]]:
    """Generate sample entities from text content for testing - mimics main app logic"""
    import re
    
    entities = []
    lines = text_content.split('\n')
    
    # Enhanced patterns for financial documents
    financial_patterns = [
        (r'AUD\s*[\d,]+\.?\d*[mn]*', 'Financial Amount'),
        (r'\$[\d,]+\.?\d*[MmBbKk]*', 'Dollar Amount'),
        (r'\d+\.?\d*%', 'Percentage'),
        (r'\d+\.?\d*\s*million', 'Million Amount'),
        (r'\d+\.?\d*\s*mn', 'Million Amount'),
        (r'\d+\.?\d*\s*billion', 'Billion Amount'),
        (r'\d+\.?\d*\s*bn', 'Billion Amount'),
    ]
    
    # Extract financial entities
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 5:
            continue
            
        # Clean line for processing
        clean_line = re.sub(r'[†•\-\(\)]', ' ', line).strip()
        
        # Look for financial figures
        for pattern, entity_type in financial_patterns:
            matches = re.finditer(pattern, line, re.IGNORECASE)
            for match in matches:
                value = match.group().strip()
                
                # Extract context around the match for field name
                start_pos = max(0, match.start() - 30)
                context = line[start_pos:match.start()].strip()
                
                # Clean up field name
                field_name = re.sub(r'[^\w\s]', ' ', context).strip()
                field_name = ' '.join(field_name.split()[-3:])  # Take last 3 words
                
                if not field_name or len(field_name) < 3:
                    field_name = f"{entity_type}_{i+1}"
                
                field_name = field_name.replace(' ', '_')
                
                entities.append({
                    'source': f'Line {i+1}',
                    'type': entity_type,
                    'field': field_name,
                    'value': value,
                    'page': 1
                })
    
    # Look for key business metrics and terms
    business_patterns = [
        (r'revenue.*?(\$[\d,]+\.?\d*[MmBbKk]*|\d+\.?\d*%|AUD\s*[\d,]+\.?\d*[mn]*)', 'Revenue'),
        (r'profit.*?(\$[\d,]+\.?\d*[MmBbKk]*|\d+\.?\d*%|AUD\s*[\d,]+\.?\d*[mn]*)', 'Profit'),
        (r'NPAT.*?(\$[\d,]+\.?\d*[MmBbKk]*|\d+\.?\d*%|AUD\s*[\d,]+\.?\d*[mn]*)', 'NPAT'),
        (r'growth.*?(\d+\.?\d*%)', 'Growth'),
        (r'margin.*?(\d+\.?\d*%)', 'Margin'),
        (r'EBITDA.*?(\$[\d,]+\.?\d*[MmBbKk]*|AUD\s*[\d,]+\.?\d*[mn]*)', 'EBITDA'),
    ]
    
    full_text = ' '.join(lines)
    for pattern, metric_type in business_patterns:
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        for match in matches:
            field_name = f"{metric_type}_Metric"
            value = match.group(1) if match.groups() else "Mentioned"
            
            entities.append({
                'source': 'Document Analysis',
                'type': 'Business Metric',
                'field': field_name,
                'value': value,
                'page': 1
            })
    
    # Add some text-based entities for comprehensive testing
    key_terms = ['CEO', 'Chairman', 'Board', 'acquisition', 'merger', 'dividend', 'outlook']
    for term in key_terms:
        if term.lower() in text_content.lower():
            entities.append({
                'source': 'Document',
                'type': 'Corporate Information',
                'field': f"{term}_Reference",
                'value': "Mentioned",
                'page': 1
            })
    
    # Remove duplicates and limit results
    seen_keys = set()
    unique_entities = []
    
    for entity in entities:
        key = f"{entity['field']}_{entity['value']}"
        if key not in seen_keys and len(entity['field']) < 40:
            seen_keys.add(key)
            unique_entities.append(entity)
    
    # Limit to 10 entities for focused testing
    unique_entities = unique_entities[:10]
    
    # Ensure we have at least a few entities for testing
    if len(unique_entities) < 3:
        fallback_entities = [
            {
                'source': 'Document',
                'type': 'Test Entity',
                'field': 'Sample_Metric_1',
                'value': 'Test Value 1',
                'page': 1
            },
            {
                'source': 'Document',
                'type': 'Test Entity', 
                'field': 'Sample_Metric_2',
                'value': 'Test Value 2',
                'page': 1
            }
        ]
        unique_entities.extend(fallback_entities)
    
    return unique_entities

@app.route('/test-prompt', methods=['POST'])
async def test_prompt():
    """Test the context extraction prompt"""
    
    try:
        prompt_template = request.form.get('prompt_template', '').strip()
        sample_data = request.form.get('sample_data', '').strip()
        
        if not prompt_template:
            return jsonify({
                'success': False,
                'error': 'Prompt template is required'
            })
        
        # Handle file upload or sample data
        text_content = ""
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                # Process PDF file using tesseract
                try:
                    from tesseract_processor import extract_text_from_pdf
                    
                    # Save uploaded file temporarily
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        file.save(tmp_file.name)
                        
                        # Extract text from PDF
                        print(f"Processing PDF: {file.filename}")
                        pdf_data = extract_text_from_pdf(tmp_file.name)
                        
                        # Get document text
                        text_content = '\n'.join(pdf_data.get('document_text', []))
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                        
                        print(f"Extracted {len(text_content)} characters from PDF")
                        
                except Exception as e:
                    return jsonify({
                        'success': False,
                        'error': f'PDF processing failed: {str(e)}'
                    })
        else:
            text_content = sample_data
        
        if not text_content:
            return jsonify({
                'success': False,
                'error': 'Sample data or PDF file is required'
            })
        
        # Generate sample entities from the text
        sample_entities = generate_sample_entities(text_content)
        
        print(f"Debug: Generated {len(sample_entities)} entities from text")
        for entity in sample_entities:
            print(f"  - {entity['field']}: {entity['value']}")
        
        if not sample_entities:
            # Create a fallback entity for testing
            sample_entities = [{
                'source': 'Fallback',
                'type': 'Test Data',
                'field': 'Sample_Entity',
                'value': 'Test Value',
                'page': 1
            }]
            print("Debug: Using fallback entity for testing")
        
        # Test the prompt with each entity
        csv_rows = []
        csv_rows.append("Source,Type,Field,Value,Page,Context")  # Header
        
        successful_tests = 0
        
        for entity in sample_entities:
            row_data = f"Field: {entity['field']}, Value: {entity['value']}, Type: {entity['type']}"
            
            print(f"Testing entity: {entity['field']} = {entity['value']}")
            
            # Test the prompt
            result = await test_context_extraction_prompt(prompt_template, row_data, text_content)
            
            if result['success']:
                successful_tests += 1
                context = result['result'].get('context', '')
                general_commentary = result['result'].get('general_commentary', '')
                
                print(f"  ✅ Success - Context length: {len(context)} chars")
                
                # Add entity row
                csv_row = f"{entity['source']},{entity['type']},{entity['field']},{clean_csv_value(entity['value'])},{entity['page']},{clean_csv_value(context)}"
                csv_rows.append(csv_row)
                
                # Add general commentary if present (only once)
                if general_commentary and len(csv_rows) == 2:  # Only add for first entity
                    csv_row = f"General,Commentary,General_Commentary,{clean_csv_value(general_commentary)},N/A,"
                    csv_rows.append(csv_row)
            else:
                print(f"  ❌ Error: {result['error']}")
                # Add row with error
                csv_row = f"{entity['source']},{entity['type']},{entity['field']},{clean_csv_value(entity['value'])},{entity['page']},Error: {result['error']}"
                csv_rows.append(csv_row)
        
        print(f"Completed testing: {successful_tests}/{len(sample_entities)} successful")
        
        csv_data = '\n'.join(csv_rows)
        csv_preview = csv_data[:2000] + '...' if len(csv_data) > 2000 else csv_data
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'csv_preview': csv_preview,
            'row_count': len(csv_rows) - 1,  # Exclude header
            'entities_tested': len(sample_entities)
        })
        
    except Exception as e:
        print(f"Error in test_prompt endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        })

@app.route('/download-xlsx', methods=['POST'])
def download_xlsx():
    """Convert CSV data to XLSX and return as download"""
    
    try:
        data = request.get_json()
        csv_data = data.get('csv_data', '')
        
        if not csv_data:
            return jsonify({'error': 'No CSV data provided'}), 400
        
        # Parse CSV data
        csv_lines = csv_data.strip().split('\n')
        if len(csv_lines) < 2:
            return jsonify({'error': 'Invalid CSV data'}), 400
        
        # Create DataFrame
        header = csv_lines[0].split(',')
        rows = []
        
        for line in csv_lines[1:]:
            # Simple CSV parsing (not handling quoted commas properly, but good enough for testing)
            row = line.split(',')
            # Pad row if needed
            while len(row) < len(header):
                row.append('')
            rows.append(row[:len(header)])  # Trim if too long
        
        df = pd.DataFrame(rows, columns=header)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            df.to_excel(tmp_file.name, index=False, engine='openpyxl')
            tmp_file_path = tmp_file.name
        
        # Return file
        def remove_file(response):
            try:
                os.unlink(tmp_file_path)
            except Exception:
                pass
            return response
        
        return send_file(
            tmp_file_path,
            as_attachment=True,
            download_name=f'prompt_test_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Simple test to check if OpenAI client is working
        api_key = os.getenv('OPENAI_API_KEY')
        return jsonify({
            'status': 'healthy',
            'openai_key_present': bool(api_key),
            'openai_key_length': len(api_key) if api_key else 0
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

@app.route('/')
def index():
    """Serve the frontend"""
    return send_file('prompt_tester.html')

if __name__ == '__main__':
    print("🧪 Prompt Tester Server Starting...")
    print("📝 Open http://localhost:5001 to access the prompt tester")
    print("🔧 Use this tool to fine-tune your context extraction prompts")
    app.run(debug=True, port=5001, host='0.0.0.0')