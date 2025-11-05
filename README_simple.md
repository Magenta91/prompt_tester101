# PDF Data Extractor - Simple Version

A clean, simplified version of the PDF data extraction system that focuses on core functionality without the complexity of prompt testing.

## Features

✅ **PDF Upload & Processing**
- Drag & drop PDF upload
- Automatic text and table extraction
- Key-value pair detection

✅ **AI-Powered Context Generation**
- Intelligent entity extraction
- Context matching using OpenAI GPT
- Financial data focus

✅ **Export Functionality**
- XLSX export with formatting
- CSV export
- Clean, readable output

✅ **Simple Interface**
- Clean, intuitive UI
- Real-time processing feedback
- Error handling and status messages

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements_simple.txt
   ```

2. **Set Environment Variables**
   Create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the Application**
   ```bash
   python app_simple.py
   ```

4. **Access the Interface**
   Open http://localhost:5000 in your browser

## Usage

1. **Upload PDF**: Drag and drop or browse for a PDF file
2. **Process with AI**: Click the "Process with AI" button to extract data and add context
3. **Export Results**: Download as XLSX or CSV format

## File Structure

```
├── app_simple.py                    # Main Flask application
├── structured_llm_processor_simple.py  # Simplified LLM processing
├── templates/
│   └── index_simple.html           # Clean web interface
├── static/
│   └── script_simple.js            # Frontend JavaScript
├── tesseract_processor.py          # PDF text extraction (reused)
├── requirements_simple.txt         # Python dependencies
└── README_simple.md               # This file
```

## Key Differences from Complex Version

- ❌ No prompt testing functionality
- ❌ No streaming processing
- ❌ No complex debugging features
- ✅ Simple, reliable processing flow
- ✅ Clean, focused interface
- ✅ Straightforward export functionality

## Processing Flow

1. **PDF Upload** → Extract text, tables, key-value pairs
2. **AI Processing** → Extract entities and add context
3. **Export** → Generate XLSX/CSV with formatted data

## Dependencies

- **Flask**: Web framework
- **OpenAI**: AI processing
- **openpyxl**: Excel file generation
- **pytesseract**: OCR processing
- **pdf2image**: PDF to image conversion

## Error Handling

- File size validation (50MB limit)
- PDF format validation
- Processing error recovery
- User-friendly error messages

This simplified version provides all the core functionality you need for PDF data extraction and export without the complexity of advanced features.