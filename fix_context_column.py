#!/usr/bin/env python3
"""
Script to fix ONLY the Context column in existing XLSX files
Preserves all other columns exactly as they are
"""

import pandas as pd
from context_enhancer import enhance_existing_contexts
import sys
import os


def fix_context_column_in_xlsx(xlsx_path: str, document_text: str, output_path: str = None):
    """
    Fix only the Context column in an existing XLSX file
    
    Args:
        xlsx_path: Path to existing XLSX file
        document_text: Full document text for context enhancement
        output_path: Output path (if None, overwrites original)
    """
    try:
        # Read existing XLSX
        df = pd.read_excel(xlsx_path)
        
        print(f"Loaded XLSX with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Convert to list of dictionaries
        rows = df.to_dict('records')
        
        # Enhance ONLY the Context column
        enhanced_rows = enhance_existing_contexts(rows, document_text)
        
        # Convert back to DataFrame
        enhanced_df = pd.DataFrame(enhanced_rows)
        
        # Ensure column order is preserved
        if list(df.columns) == list(enhanced_df.columns):
            enhanced_df = enhanced_df[df.columns]
        
        # Save to output path
        if output_path is None:
            output_path = xlsx_path.replace('.xlsx', '_fixed_context.xlsx')
        
        enhanced_df.to_excel(output_path, index=False)
        
        print(f"Fixed Context column and saved to: {output_path}")
        
        # Show sample of improvements
        print("\nSample Context Improvements:")
        for i in range(min(3, len(enhanced_rows))):
            original_context = rows[i].get('Context', '')
            enhanced_context = enhanced_rows[i].get('Context', '')
            
            print(f"\nRow {i+1} - Field: {enhanced_rows[i].get('Field', 'N/A')}")
            print(f"  Original Context Length: {len(original_context)} chars")
            print(f"  Enhanced Context Length: {len(enhanced_context)} chars")
            if len(enhanced_context) > len(original_context):
                print(f"  ✓ Improved by {len(enhanced_context) - len(original_context)} characters")
            
        return output_path
        
    except Exception as e:
        print(f"Error fixing Context column: {e}")
        return None


def create_sample_document_text():
    """
    Create sample document text for testing
    """
    return """
    Summary:
    † Underlying NPAT¹ of AUD 46.7mn (1HFY22: AUD 30.6mn)
    † Revenue growth of 15% year-on-year driven by strong performance
    † Total revenue of AUD 234.5mn for the half year
    
    CEO Commentary:
    "We are pleased with the underlying NPAT performance which exceeded our forecasts by 12%."
    "The underlying NPAT of AUD 46.7mn represents strong operational efficiency and disciplined cost management."
    "Our revenue growth demonstrates the strength of our diversified business model."
    
    Tysers Performance:
    † Tysers underlying pre-tax profit contributed AUD 18.0mn for the 3 months to 31 December 2022
    † Tysers performance for the three months since acquisition on 1 October has exceeded our forecasts
    † Integration of Tysers operations proceeding ahead of schedule
    
    Australian Broking Division:
    † Australian Broking delivered strong results with revenue growth of 8%
    † New client acquisitions increased by 25% compared to prior period
    † Market share expansion in key segments
    
    Guidance Update:
    FY23 Underlying NPAT guidance upgraded to AUD 95-100mn based on strong performance.
    Revenue guidance maintained at AUD 450-470mn for the full year.
    
    Footnotes:
    ¹ Underlying NPAT excludes one-off acquisition costs and restructuring expenses.
    ² Revenue figures are presented on a pro-forma basis including acquisitions.
    """


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        xlsx_file = sys.argv[1]
        if os.path.exists(xlsx_file):
            # Use sample document text (in real usage, this would be the actual document)
            document_text = create_sample_document_text()
            
            output_file = fix_context_column_in_xlsx(xlsx_file, document_text)
            if output_file:
                print(f"\n✓ Successfully enhanced Context column in: {output_file}")
            else:
                print("✗ Failed to enhance Context column")
        else:
            print(f"File not found: {xlsx_file}")
    else:
        print("Usage: python fix_context_column.py <xlsx_file_path>")
        print("This script will enhance ONLY the Context column while preserving all other data.")