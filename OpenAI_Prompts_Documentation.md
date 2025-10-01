# OpenAI Prompts Documentation
## PDF Text Extractor and Tabulator System

This document contains all the OpenAI prompts used in the PDF Text Extractor and Tabulator system for generating XLSX files from document analysis.

---

## System Overview

The system uses **GPT-4o** model with JSON response format to process documents through multiple specialized prompts:

1. **Text Chunk Processing** - Main document analysis and data extraction
2. **Table Data Processing** - Structured table data extraction
3. **Key-Value Processing** - Key-value pair organization
4. **Commentary Matching** - Context generation for data points

---

## Prompt 1: Text Chunk Processing (Main Document Analysis)

**Function:** `process_text_chunk()`  
**Model:** GPT-4o  
**Purpose:** Extract and tabulate ALL meaningful data from document text segments

### Prompt Template:
```
You are a financial document analyst. Extract and tabulate ALL meaningful data from this text segment.

Create a comprehensive table structure that captures the key information in a tabulated format.

Text:
{text_content}

Requirements:
1. Extract ALL meaningful data points and organize them into a table structure
2. Create appropriate column headers based on the content type
3. Structure data into logical rows and columns
4. Include financial metrics, dates, percentages, company info, etc.
5. If the text contains narrative information, extract key facts and tabulate them
6. IGNORE superscript numbers and footnote reference markers (¹²³ or (1)(2)(3) or [1][2][3])
7. Extract clean data values without footnote symbols

Return JSON with BOTH table structure AND individual facts:
{
  "table_headers": ["Metric", "Value", "Period", "Context"],
  "table_rows": [
    ["Revenue", "$115.5M", "Q4 2023", "33% growth"],
    ["MAU", "65.8M", "Q4 2023", "Global users"],
    ["Market Share", "12%", "2023", "Primary market"]
  ],
  "extracted_facts": {
    "Company_Name": "Life360",
    "Q4_Revenue": "$115.5 million",
    "MAU_Growth": "33%",
    "Market_Position": "Leading family safety platform"
  }
}

Extract comprehensive data - do not limit to just a few items. Return the response as valid JSON format.
```

### Expected Output:
- **table_headers**: Array of column headers
- **table_rows**: Array of data rows matching headers
- **extracted_facts**: Object with key-value pairs of extracted data

---

## Prompt 2: Table Data Processing

**Function:** `process_table_data()`  
**Model:** GPT-4o  
**Purpose:** Extract key data points from detected tables as simple field-value pairs

### Prompt Template:
```
Extract key data points from this table as simple field-value pairs.

Table data:
{json.dumps(table_data, indent=2)}

Instructions:
1. Extract important data points as field-value pairs
2. Use clear, descriptive field names
3. Focus on financial figures, dates, and key metrics
4. Keep it simple and straightforward

Return JSON with field-value pairs:
{
  "Revenue": "value",
  "Growth_Rate": "value",
  "Date": "value"
}
```

### Expected Output:
- Simple JSON object with descriptive field names as keys and extracted values

---

## Prompt 3: Key-Value Pairs Processing

**Function:** `process_key_value_pairs()`  
**Model:** GPT-4o  
**Purpose:** Organize pre-extracted key-value pairs into clear, structured data

### Prompt Template:
```
You are a data extraction specialist. Below are key-value pairs extracted from a document.

Extract and organize this information into clear field-value pairs. Focus on extracting actual data values like company names, dates, amounts, percentages, and other factual information.

Key-Value pairs:
{json.dumps(key_value_pairs, indent=2)}

Return a simple JSON object where each key is a descriptive field name and each value is the actual extracted data. Do not create nested structures or arrays. Provide the response as valid JSON format.
```

### Expected Output:
- Flat JSON object with organized field-value pairs
- No nested structures or arrays
- Clean, descriptive field names

---

## Prompt 4: Commentary Matching (Context Generation)

**Function:** `match_commentary_to_data()`  
**Model:** GPT-4o  
**Purpose:** Extract comprehensive context for entities while preserving all document text

### Prompt Template:
```
You are a precise document analysis expert. Your job is to:

For each entity (e.g., a student, company, metric) represented in {row_data}, find all sentences and phrases across the entire document {text_content} that describe, explain, or add information about that entity.

Copy the sentences word-for-word exactly as they appear in the document (do not paraphrase, do not correct grammar, do not drop words).

Concatenate multiple mentions into a single Context value for that entity, separated by periods or newlines.

Ensure integrity of language: text must remain unaltered, in the same language and order.

Any text that does not describe any entity in {row_data} must be placed in a separate "General Commentary" row. Every single word from the input document must appear in either:
- An entity's Context, or
- The "General Commentary" row (if it is not linked to any entity).

STRICT MATCHING CRITERIA:
- Include any sentence or phrase that clearly refers to the entity by name, value, or associated identifiers.
- Accept contextual descriptions, traits, or commentary even if not directly numerical (e.g., "ABC is hardworking", "ABC loves to ride bikes").
- Reject text only if it cannot reasonably be tied to any entity.

Return JSON:
For entity rows:
{"context": "exact text from document related to this entity", "general_commentary": null}

For general commentary row:
{"context": null, "general_commentary": "all remaining text, word for word, that was not tied to any entity"}

Be exhaustive: make sure no text is lost or skipped. The final CSV/Excel must account for every word in the input file.
```

### Expected Output:
- **context**: String with exact text from document related to the entity, or null
- **general_commentary**: String with remaining unlinked text, or null
- **Comprehensive Coverage**: Every word from input document must be accounted for

---

## Processing Flow

### 1. Document Ingestion
- PDF text is extracted and split into manageable chunks
- Tables and key-value pairs are identified separately

### 2. Parallel Processing
- **Text chunks** → Prompt 1 (Main analysis)
- **Table data** → Prompt 2 (Table processing)  
- **Key-value pairs** → Prompt 3 (KV processing)

### 3. Context Enhancement
- Each extracted data point → Prompt 4 (Commentary matching)
- Generates relevant context for XLSX Context column

### 4. Output Generation
- All processed data combined into structured format
- Exported as XLSX with columns: Source, Type, Field, Value, Page, Context

---

## Technical Configuration

### Model Settings:
- **Model**: `gpt-4o`
- **Response Format**: `{"type": "json_object"}`
- **Temperature**: Default (not specified, likely 0.7)
- **Max Tokens**: Not specified (uses model default)

### Cost Tracking:
The system tracks token usage and costs for each API call:
- Prompt tokens and completion tokens are monitored
- Costs are calculated and logged for each processing step

### Error Handling:
Each prompt includes try-catch blocks that return structured error responses:
```json
{
  "error": "Error description",
  "structured_data": {},
  "original_input": "preserved_input"
}
```

---

## Prompt Optimization Notes

### Strengths:
1. **Comprehensive Extraction**: Prompt 1 ensures all meaningful data is captured
2. **Strict Relevance**: Prompt 4 uses ultra-strict criteria to avoid irrelevant context
3. **Structured Output**: All prompts enforce JSON format for consistent processing
4. **Financial Focus**: Optimized for financial documents and business metrics

### Areas for Customization:
1. **Domain Adaptation**: Prompts can be modified for different document types
2. **Language Support**: Currently optimized for English, can be adapted for other languages
3. **Extraction Depth**: Adjust comprehensiveness vs. processing speed trade-offs
4. **Context Strictness**: Modify relevance scoring thresholds in Prompt 4

---

## Usage Examples

### Input Document Types:
- Financial reports and statements
- Business presentations and decks
- Annual reports and 10-K filings
- Earnings call transcripts
- Investment research reports

### Output XLSX Structure:
| Source | Type | Field | Value | Page | Context |
|--------|------|-------|-------|------|---------|
| Text Chunk 1 | Financial Data | Revenue | $115.5M | 1 | Revenue increased 33% year-over-year... |
| Table 1 | Table Data | Growth_Rate | 15% | 2 | Strong performance driven by... |
| Key-Value Pairs | Structured Data | Company_Name | Life360 | N/A | Leading family safety platform... |

---

## Maintenance and Updates

### Version Control:
- Document prompt changes with version numbers
- Test prompt modifications on sample documents before deployment
- Monitor output quality and adjust prompts as needed

### Performance Monitoring:
- Track processing costs and token usage
- Monitor extraction accuracy and completeness
- Adjust prompt complexity based on performance requirements

---

*Last Updated: [Current Date]*  
*System Version: PDF Text Extractor and Tabulator v1.0*