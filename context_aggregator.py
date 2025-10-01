#!/usr/bin/env python3
"""
Context Aggregation Module for PDF Table Extractor

This module implements comprehensive context tracking and aggregation for entities
across document pages, ensuring all descriptive sentences are captured word-for-word
in the Context column of the output table.
"""

import re
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import difflib


class ContextAggregator:
    """
    Handles entity tracking across pages and context aggregation for structured data extraction.
    """
    
    def __init__(self):
        self.entity_contexts = defaultdict(set)  # {entity_key: set of context sentences}
        self.entity_identifiers = defaultdict(set)  # {entity_key: set of identifiers/names}
        self.processed_sentences = set()  # Track processed sentences to avoid duplicates
        self.document_sentences = []  # Store all document sentences with metadata
        
    def add_document_text(self, document_text: List[str], page_numbers: List[int] = None):
        """
        Add document text for processing, splitting into sentences and storing metadata.
        
        Args:
            document_text: List of text lines from the document
            page_numbers: Optional list of page numbers corresponding to each line
        """
        if page_numbers is None:
            page_numbers = [1] * len(document_text)
            
        # Process each line and extract sentences
        for i, line in enumerate(document_text):
            if not line.strip():
                continue
                
            page_num = page_numbers[i] if i < len(page_numbers) else 1
            
            # Split line into sentences while preserving original text
            sentences = self._split_into_sentences(line.strip())
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # Only meaningful sentences
                    self.document_sentences.append({
                        'text': sentence.strip(),
                        'page': page_num,
                        'line_index': i,
                        'original_line': line.strip()
                    })
    
    def register_entity(self, entity_key: str, entity_data: Dict[str, Any]):
        """
        Register an entity and its identifiers for context tracking.
        
        Args:
            entity_key: Unique key for the entity (e.g., "student_john_doe")
            entity_data: Dictionary containing entity fields like Name, ID, etc.
        """
        # Extract potential identifiers from entity data
        identifiers = set()
        
        # Priority fields for identification (full names, companies, etc.)
        priority_fields = ['Name', 'Company', 'Symbol', 'Title', 'ID']
        
        for field, value in entity_data.items():
            if value and isinstance(value, str):
                # Clean and add the value as an identifier
                clean_value = str(value).strip()
                if len(clean_value) > 2:  # Avoid very short identifiers
                    identifiers.add(clean_value)
                    
                    # Add variations (without punctuation, lowercase, etc.)
                    clean_alpha = re.sub(r'[^\w\s]', '', clean_value)
                    if clean_alpha != clean_value and len(clean_alpha) > 2:
                        identifiers.add(clean_alpha)
                    
                    # For priority fields, add full names with higher precedence
                    if field in priority_fields:
                        # Add the full value with highest priority
                        identifiers.add(clean_value)
                        
                        # For compound names, only add individual words if they're substantial
                        words = clean_value.split()
                        if len(words) > 1:
                            for word in words:
                                if len(word) > 3:  # Only substantial words
                                    identifiers.add(word)
                    else:
                        # For non-priority fields, be more selective about individual words
                        words = clean_value.split()
                        if len(words) > 1:
                            for word in words:
                                if len(word) > 4:  # Higher threshold for non-priority fields
                                    identifiers.add(word)
        
        self.entity_identifiers[entity_key].update(identifiers)
    
    def find_entity_contexts(self, entity_key: str) -> List[str]:
        """
        Find all context sentences for a specific entity.
        
        Args:
            entity_key: The entity key to find contexts for
            
        Returns:
            List of context sentences related to the entity
        """
        if entity_key not in self.entity_identifiers:
            return []
        
        identifiers = self.entity_identifiers[entity_key]
        contexts = []
        
        for sentence_data in self.document_sentences:
            sentence = sentence_data['text']
            
            # Check if any identifier appears in this sentence
            if self._sentence_mentions_entity(sentence, identifiers):
                # Check if this sentence is already claimed by another entity with higher specificity
                sentence_key = f"{entity_key}_{sentence}"
                
                # Avoid duplicates within the same entity
                if sentence not in self.entity_contexts[entity_key]:
                    contexts.append(sentence)
                    self.entity_contexts[entity_key].add(sentence)
        
        return sorted(list(self.entity_contexts[entity_key]))
    
    def _sentence_mentions_entity(self, sentence: str, identifiers: Set[str]) -> bool:
        """
        Check if a sentence mentions any of the entity identifiers.
        
        Args:
            sentence: The sentence to check
            identifiers: Set of identifiers to look for
            
        Returns:
            True if the sentence mentions the entity
        """
        sentence_lower = sentence.lower()
        sentence_clean = re.sub(r'[^\w\s]', ' ', sentence_lower)
        
        # Sort identifiers by length (longest first) to prioritize full names
        sorted_identifiers = sorted(identifiers, key=len, reverse=True)
        
        for identifier in sorted_identifiers:
            identifier_lower = identifier.lower()
            
            # Skip very short identifiers that might cause false matches
            if len(identifier_lower) < 3:
                continue
            
            # Exact match with word boundaries (most precise)
            if re.search(r'\b' + re.escape(identifier_lower) + r'\b', sentence_lower):
                return True
            
            # For compound names (first + last), check if both parts are present
            if ' ' in identifier_lower:
                name_parts = identifier_lower.split()
                if len(name_parts) >= 2:
                    # All parts must be present for compound names
                    if all(re.search(r'\b' + re.escape(part) + r'\b', sentence_lower) for part in name_parts):
                        return True
            
            # Fuzzy match for similar names (90% similarity, higher threshold)
            for word in sentence_clean.split():
                if len(word) > 4 and len(identifier_lower) > 4:
                    if difflib.SequenceMatcher(None, word, identifier_lower).ratio() > 0.90:
                        return True
        
        return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences while preserving original formatting.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting that preserves original text
        sentences = []
        
        # Split on sentence endings but be careful with abbreviations
        sentence_endings = r'[.!?]+(?:\s+|$)'
        parts = re.split(f'({sentence_endings})', text)
        
        current_sentence = ""
        for i, part in enumerate(parts):
            current_sentence += part
            
            # If this part ends with sentence punctuation and next part starts with capital or is end
            if re.match(sentence_endings, part):
                if i + 1 >= len(parts) or (i + 1 < len(parts) and 
                                         (parts[i + 1].strip() == '' or 
                                          (parts[i + 1].strip() and parts[i + 1].strip()[0].isupper()))):
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s.strip()]
    
    def aggregate_contexts_for_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate contexts for a list of entities and add Context column.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List of entities with Context column populated
        """
        # First pass: Register all entities
        entity_keys = []
        for i, entity in enumerate(entities):
            entity_key = self._create_entity_key(entity, i)
            self.register_entity(entity_key, entity)
            entity_keys.append(entity_key)
        
        # Second pass: Find contexts with conflict resolution
        claimed_sentences = set()  # Track sentences already claimed by entities
        enhanced_entities = []
        
        # Sort entities by specificity (entities with more specific identifiers first)
        entity_specificity = []
        for i, (entity, entity_key) in enumerate(zip(entities, entity_keys)):
            identifiers = self.entity_identifiers[entity_key]
            # Calculate specificity score (longer identifiers = more specific)
            specificity_score = sum(len(id_str) for id_str in identifiers if ' ' in id_str or len(id_str) > 5)
            entity_specificity.append((specificity_score, i, entity, entity_key))
        
        # Process entities in order of specificity (most specific first)
        entity_specificity.sort(key=lambda x: x[0], reverse=True)
        
        # Create a mapping to restore original order
        result_mapping = {}
        
        for specificity_score, original_index, entity, entity_key in entity_specificity:
            contexts = []
            identifiers = self.entity_identifiers[entity_key]
            
            for sentence_data in self.document_sentences:
                sentence = sentence_data['text']
                
                # Skip if sentence is already claimed
                if sentence in claimed_sentences:
                    continue
                
                # Check if this sentence mentions this specific entity
                if self._sentence_mentions_entity(sentence, identifiers):
                    # Additional check: ensure this sentence is most relevant to this entity
                    if self._is_sentence_most_relevant_to_entity(sentence, entity_key, entity_keys):
                        contexts.append(sentence)
                        claimed_sentences.add(sentence)
                        self.entity_contexts[entity_key].add(sentence)
            
            # Create enhanced entity with context
            enhanced_entity = entity.copy()
            
            # Aggregate contexts into a single string
            if contexts:
                # Remove duplicates while preserving order
                unique_contexts = []
                seen = set()
                for context in contexts:
                    if context not in seen:
                        unique_contexts.append(context)
                        seen.add(context)
                
                # Join contexts with period separation
                enhanced_entity['Context'] = '. '.join(unique_contexts)
            else:
                enhanced_entity['Context'] = ''
            
            result_mapping[original_index] = enhanced_entity
        
        # Restore original order
        enhanced_entities = [result_mapping[i] for i in range(len(entities))]
        
        return enhanced_entities
    
    def _is_sentence_most_relevant_to_entity(self, sentence: str, target_entity_key: str, all_entity_keys: List[str]) -> bool:
        """
        Determine if a sentence is most relevant to the target entity compared to other entities.
        
        Args:
            sentence: The sentence to check
            target_entity_key: The entity key we're checking relevance for
            all_entity_keys: All entity keys to compare against
            
        Returns:
            True if the sentence is most relevant to the target entity
        """
        target_identifiers = self.entity_identifiers[target_entity_key]
        target_score = self._calculate_sentence_relevance_score(sentence, target_identifiers)
        
        # Check against other entities
        for other_key in all_entity_keys:
            if other_key == target_entity_key:
                continue
                
            other_identifiers = self.entity_identifiers[other_key]
            other_score = self._calculate_sentence_relevance_score(sentence, other_identifiers)
            
            # If another entity has a higher relevance score, this sentence belongs to that entity
            if other_score > target_score:
                return False
        
        return target_score > 0
    
    def _calculate_sentence_relevance_score(self, sentence: str, identifiers: Set[str]) -> float:
        """
        Calculate how relevant a sentence is to a set of identifiers.
        
        Args:
            sentence: The sentence to score
            identifiers: Set of identifiers for an entity
            
        Returns:
            Relevance score (higher = more relevant)
        """
        sentence_lower = sentence.lower()
        score = 0.0
        
        for identifier in identifiers:
            identifier_lower = identifier.lower()
            
            # Skip very short identifiers
            if len(identifier_lower) < 3:
                continue
            
            # Exact full match (highest score)
            if re.search(r'\b' + re.escape(identifier_lower) + r'\b', sentence_lower):
                if ' ' in identifier_lower:  # Compound names get higher score
                    score += len(identifier_lower) * 2
                else:
                    score += len(identifier_lower)
            
            # Partial match (lower score)
            elif identifier_lower in sentence_lower:
                score += len(identifier_lower) * 0.5
        
        return score
    
    def _create_entity_key(self, entity: Dict[str, Any], index: int) -> str:
        """
        Create a unique key for an entity.
        
        Args:
            entity: Entity dictionary
            index: Index of the entity in the list
            
        Returns:
            Unique entity key
        """
        # Try to use meaningful fields for the key
        key_fields = ['Name', 'Company', 'Symbol', 'ID', 'Title', 'Field']
        
        for field in key_fields:
            if field in entity and entity[field]:
                clean_value = re.sub(r'[^\w]', '_', str(entity[field]).lower())
                return f"{field.lower()}_{clean_value}"
        
        # Fallback to index-based key
        return f"entity_{index}"
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of context aggregation results.
        
        Returns:
            Dictionary with aggregation statistics
        """
        return {
            'total_entities': len(self.entity_identifiers),
            'total_sentences': len(self.document_sentences),
            'entities_with_context': len([k for k, v in self.entity_contexts.items() if v]),
            'total_contexts_found': sum(len(v) for v in self.entity_contexts.values()),
            'processed_sentences': len(self.processed_sentences)
        }


def enhance_extracted_data_with_context(extracted_data: List[Dict[str, Any]], 
                                      document_text: List[str],
                                      page_numbers: List[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Main function to enhance extracted data with comprehensive context aggregation.
    
    Args:
        extracted_data: List of extracted entity dictionaries
        document_text: List of document text lines
        page_numbers: Optional list of page numbers for each line
        
    Returns:
        Tuple of (enhanced_data_with_context, aggregation_summary)
    """
    # Initialize context aggregator
    aggregator = ContextAggregator()
    
    # Add document text for processing
    aggregator.add_document_text(document_text, page_numbers)
    
    # Aggregate contexts for all entities
    enhanced_data = aggregator.aggregate_contexts_for_entities(extracted_data)
    
    # Get summary statistics
    summary = aggregator.get_context_summary()
    
    return enhanced_data, summary


# Example usage and testing
if __name__ == "__main__":
    # Test data
    sample_entities = [
        {"Name": "John Doe", "Age": "25", "Class": "A"},
        {"Name": "Jane Smith", "Age": "30", "Class": "B"},
        {"Company": "ABC Corp", "Revenue": "$1M", "Year": "2024"}
    ]
    
    sample_document = [
        "John Doe is a hardworking student who excels in mathematics.",
        "The company ABC Corp has shown remarkable growth this year.",
        "Jane Smith loves to ride bikes and is very athletic.",
        "ABC Corp's revenue increased by 20% compared to last year.",
        "John Doe also participates in various extracurricular activities."
    ]
    
    enhanced_data, summary = enhance_extracted_data_with_context(
        sample_entities, sample_document
    )
    
    print("Enhanced Data with Context:")
    for entity in enhanced_data:
        print(f"Entity: {entity}")
        print(f"Context: {entity.get('Context', 'No context found')}")
        print("-" * 50)
    
    print(f"\nAggregation Summary: {summary}")