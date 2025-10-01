# PDF Text Extractor with Context Prompt Testing

A comprehensive web application that extracts text from PDF files, processes it using OpenAI's GPT-4o model to identify structured information, and displays the results in a tabulated format with export options. Features an integrated frontend prompt testing system for fine-tuning context extraction prompts.

## 🚀 Key Features

- **PDF Text Extraction**: Uses Amazon Textract (with Tesseract OCR fallback)
- **AI-Powered Analysis**: GPT-4o model for intelligent data extraction
- **Context Prompt Testing**: Frontend interface for testing custom context extraction prompts
- **Real-time Streaming**: Live processing updates with streaming responses
- **Multiple Export Formats**: XLSX, CSV, JSON, and PDF export options
- **Context Generation**: LLM-based context extraction for every data point
- **Cost Tracking**: Built-in OpenAI API cost monitoring

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Setup and Installation](#setup-and-installation)
- [Technologies Used](#technologies-used)
- [API Integration](#api-integration)
- [Export Options](#export-options)
- [Customization](#customization)

## Overview

This application provides a streamlined way to extract structured information from PDF documents. It uses advanced AI to analyze text content and organize it into a tabular format that can be easily viewed and exported. The system automatically identifies entities and their categories, making it particularly useful for processing documents with consistent information patterns.

## Features

- **PDF Text Extraction**: Upload and extract text from PDF documents with AWS Textract and Tesseract OCR fallback
- **AI-Powered Analysis**: Process text with OpenAI's GPT-4o model to identify structured information
- **Comprehensive Context Aggregation**: Automatically tracks entities across document pages and aggregates all related descriptive sentences word-for-word in the Context column
- **Interactive UI**: Clean, modern interface with drag-and-drop file upload
- **Real-time Processing**: See extracted text immediately after upload with streaming progress
- **Tabulated Results**: View structured data in a clean, organized table with full context
- **Multiple Export Options**: Download results as JSON, CSV, XLSX, or PDF
- **Automatic Fallback**: Seamless fallback from AWS Textract to Tesseract OCR when credentials fail
- **Environment Security**: Secure credential management with .env files
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

The application follows a client-server architecture with:

1. **Frontend**: HTML, CSS, and JavaScript for user interface
2. **Backend**: Flask Python server to handle requests
3. **Processing Layer**: PDF processing and AI analysis components
4. **Export Layer**: Conversion utilities for different output formats

## File Structure

```
├── app.py                  # Main Flask application
├── llm_processor.py        # AI text processing with OpenAI
├── pdf_processor.py        # PDF text extraction with PyPDF2
├── export_utils.py         # Export functionality for various formats
├── templates/
│   └── index.html          # Main HTML template
└── static/
    ├── style.css           # CSS styling
    └── script.js           # Frontend JavaScript
```

## How It Works

1. **PDF Upload**:
   - User uploads a PDF through the web interface
   - The file is sent to the server for processing

2. **Text Extraction**:
   - Primary: AWS Textract extracts structured data, tables, and key-value pairs
   - Fallback: Tesseract OCR processes the PDF if AWS credentials fail
   - Extracted text is displayed to the user with processing method indicated

3. **AI Analysis**:
   - User initiates processing with the "Process with AI" button
   - The `structured_llm_processor.py` module processes data asynchronously with OpenAI's GPT-4o
   - Custom prompting guides the AI to identify structured information
   - The AI returns data categorized into predefined columns

4. **Context Aggregation**:
   - The `context_aggregator.py` module tracks entities across all document pages
   - For each entity (person, company, metric), it finds all related descriptive sentences
   - Contexts are aggregated word-for-word from the original document
   - Conflict resolution ensures each sentence is assigned to the most relevant entity

5. **Result Display**:
   - Processed data is displayed in a table format with comprehensive Context column
   - Information is organized by source, type, field, value, page, and full context
   - Real-time streaming shows progress during processing

6. **Data Export**:
   - User selects preferred export format (JSON, CSV, XLSX, PDF)
   - The `export_utils.py` module handles conversion to the selected format
   - Generated file includes all structured data with complete context information

## Setup and Installation

1. **Prerequisites**:
   - Python 3.11 or higher
   - OpenAI API key

2. **Installation**:
   ```bash
   # Clone the repository
   git clone [repository-url]
   cd pdf-text-extractor

   # Install dependencies using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Environment Variables**:
   - Create a `.env` file in the project root with your API keys:
   ```
   AWS_ACCESS_KEY_ID=your-aws-access-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret-key
   AWS_DEFAULT_REGION=us-east-1
   OPENAI_API_KEY=your-openai-api-key
   ```
   - The application will automatically load these variables from the `.env` file
   - **Important**: Never commit the `.env` file to version control (it's already in `.gitignore`)

4. **Running the Application**:
   ```bash
   python app.py
   ```
   - Access the application at http://localhost:5000

## Technologies Used

- **Flask**: Lightweight web framework for the backend
- **OpenAI API**: GPT-4o model for text analysis
- **PyPDF2**: PDF parsing and text extraction
- **Pandas**: Data manipulation for structured information
- **ReportLab**: PDF generation for exports
- **Bootstrap**: Frontend styling framework
- **JavaScript**: Frontend interactivity and API calls

## API Integration

The application integrates with OpenAI's API to process the extracted text. The `llm_processor.py` file handles this integration with the following approach:

1. Creates an OpenAI client instance with the API key
2. Constructs a specialized prompt for document analysis
3. Sends the prompt along with the extracted text to the GPT-4o model
4. Processes the JSON response to extract structured data
5. Returns the data in a format ready for display and export

## Export Options

The application provides three export formats:

1. **JSON**: Raw structured data in a machine-readable format
2. **CSV**: Tabular data suitable for spreadsheet applications
3. **PDF**: Formatted document with the extracted information in a table

The export functionality is handled in `export_utils.py` and the frontend JavaScript.

## Context Aggregation System

The application features a sophisticated context aggregation system that ensures comprehensive information capture:

### Entity Tracking
- **Cross-Page Tracking**: Entities are tracked across all document pages using unique identifiers
- **Smart Matching**: Uses exact string matching, synonyms, and fuzzy matching for entity identification
- **Conflict Resolution**: Advanced algorithm ensures each sentence is assigned to the most relevant entity

### Context Collection
- **Verbatim Extraction**: All descriptive sentences are captured exactly as they appear in the document, including special characters (†, •, ◦, ¹, etc.)
- **Language Integrity**: Original grammar, casing, punctuation, and formatting are preserved without alteration (e.g., "mn" not "million", "LHFY22" preserved)
- **Comprehensive Coverage**: Every mention of an entity across the entire document is aggregated using multiple search strategies
- **Special Character Preservation**: Bullet points, superscripts, currency symbols, and abbreviations are maintained exactly as in source
- **No Truncation**: Full contexts are preserved without "(truncated X characters)" limitations
- **Advanced Deduplication**: Intelligent duplicate detection removes identical sentences while preserving unique information
- **Context Validation**: Built-in validation checks for empty contexts and duplicate issues with detailed logging

### Output Format
- **Structured Data**: Traditional fields (Name, Value, etc.) remain in their respective columns
- **Context Column**: Contains all related descriptive sentences concatenated with period separation
- **No Duplication**: Duplicate sentences are automatically filtered while preserving unique contexts

### Example
If a financial document mentions:
- Page 1: "† Underlying NPAT¹ of AUD 46.7mn (1HFY22: AUD 30.6mn)"
- Page 1: "We are pleased with the underlying NPAT performance which exceeded our forecasts by 12%."
- Page 2: "† Tysers underlying pre-tax profit contributed AUD 18.0mn for the 3 months to 31 December 2022"

The Context column for "Underlying_NPAT_1HFY23" = "AUD 46.7mn" will contain: "† Underlying NPAT¹ of AUD 46.7mn (1HFY22: AUD 30.6mn). We are pleased with the underlying NPAT performance which exceeded our forecasts by 12%."

### Enhanced Processing Features
- **Multi-Strategy Matching**: Uses exact phrase matching, word boundary matching, and fuzzy matching for comprehensive context discovery
- **Financial Document Optimized**: Specifically designed for financial reports with bullet points, footnotes, and structured data
- **Duplicate Prevention**: Automatically removes duplicate sentences while preserving unique contexts
- **Entity Grouping**: Groups related fields (e.g., all Tysers-related metrics) for comprehensive context aggregation

## Customization

The AI prompts in `structured_llm_processor.py` can be customized to extract different types of information. The context aggregation system in `context_aggregator.py` can be configured for:

- **Entity Priority**: Adjust which fields are considered primary identifiers
- **Matching Sensitivity**: Modify fuzzy matching thresholds
- **Context Filtering**: Customize which types of sentences are included as context

You can customize these components to focus on specific types of information relevant to your documents.
## 🧪 
Context Prompt Testing

The application includes a powerful frontend prompt testing system that allows you to:

- **Test Custom Prompts**: Input custom context extraction prompts and see immediate results
- **Real-time Feedback**: View how different prompts affect context generation
- **Prompt Comparison**: Compare results from different prompt variations
- **Live Processing**: Uses the same full pipeline as normal document processing

### How to Use Prompt Testing:

1. Upload a PDF document
2. Scroll to the "Test Context Extraction Prompt" section
3. Modify the prompt template (use `{row_data}` and `{text_content}` placeholders)
4. Click "Test Prompt" to see results
5. View context extraction results in the results table

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Magenta91/prompt_tester101.git
cd prompt_tester101
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
AWS_ACCESS_KEY_ID=your_aws_key (optional)
AWS_SECRET_ACCESS_KEY=your_aws_secret (optional)
AWS_DEFAULT_REGION=us-east-1 (optional)
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## 📋 Requirements

- Python 3.8+
- OpenAI API key (required)
- AWS credentials (optional, for Textract)
- Tesseract OCR (fallback option)

## 🔧 Configuration

### Context Extraction Prompts

You can modify context extraction prompts in two ways:

1. **Frontend Testing**: Use the integrated prompt tester for temporary testing
2. **Code Modification**: Edit the default prompt in `structured_llm_processor.py` (line ~305)

### Processing Options

- **Streaming Mode**: Real-time processing updates (default)
- **Batch Mode**: Process entire document at once
- **Cost Tracking**: Monitor OpenAI API usage and costs

## 📊 Export Formats

- **XLSX**: Excel format with structured data and context columns
- **CSV**: Comma-separated values for data analysis
- **JSON**: Raw structured data for programmatic use
- **PDF**: Formatted report with extracted information

## 🏗️ Architecture

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Backend**: Flask (Python)
- **AI Processing**: OpenAI GPT-4o
- **Text Extraction**: Amazon Textract / Tesseract OCR
- **Data Processing**: Pandas, NumPy
- **Export**: openpyxl, reportlab

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ for intelligent document processing**