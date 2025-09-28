# PDF Text Extractor and Tabulator

## Overview

This application extracts text from PDF files, processes the extracted content using an OpenAI language model to identify structured information, and displays the results in a tabulated format. Users can then download the results in various formats.

## User Preferences

Preferred communication style: Simple, everyday language.
Commentary relevance: Only attach commentary to data fields if it's highly relevant and directly explains the specific data point.

## System Architecture

The application follows a simple, streamlined architecture:

1. **Frontend**: Streamlit web interface for user interactions
2. **Backend Processing**: Python modules for PDF processing and LLM interaction 
3. **Data Flow**: PDF upload → Text extraction → LLM processing → Tabulation → Export options

The architecture prioritizes simplicity and straightforward user interaction, with each component handling a specific responsibility in the data processing pipeline.

## Key Components

### 1. Streamlit Interface (`app.py`)
The main entry point that provides:
- PDF file upload functionality
- Display of extracted and processed data
- Export options for the processed data

### 2. PDF Processing (`pdf_processor.py`)
Handles the extraction of text from PDF files using PyPDF2:
- `extract_text_from_pdf`: Processes PDF files from a file path
- `extract_text_from_pdf_bytes`: Processes PDF content provided as bytes

### 3. LLM Processing (`llm_processor.py`)
Leverages OpenAI's GPT-4o model to:
- Analyze extracted text from PDFs
- Convert unstructured text into structured, tabular JSON data
- Uses LangChain for prompt construction and model interaction

### 4. Export Utilities (`export_utils.py`)
Provides functionality to:
- Export processed data to PDF format using ReportLab
- Create downloadable links for the exported data

## Data Flow

1. **Input**: User uploads a PDF document through the Streamlit interface
2. **Processing**:
   - PDF text is extracted using PyPDF2
   - Extracted text is sent to OpenAI's GPT-4o model
   - The LLM analyzes the text and structures it into tabular format
3. **Output**:
   - Structured data is displayed as a table in the interface
   - User can download the data in various formats

## External Dependencies

### Core Libraries
- `streamlit`: Web application framework
- `pandas`: Data manipulation and analysis
- `PyPDF2`: PDF processing
- `langchain`: LLM interaction framework
- `openai`: API for GPT-4o access
- `reportlab`: PDF generation for exports

### External Services
- **OpenAI API**: Requires an API key set as an environment variable (`OPENAI_API_KEY`)

## Deployment Strategy

The application is configured for deployment on Replit with:
- Python 3.11 runtime
- Streamlit server running on port 5000
- Autoscaling deployment target
- Custom workflow configuration for the run button
- Headless server mode with external accessibility

The deployment uses Nix packaging to ensure all dependencies are properly managed and the application runs consistently in the Replit environment.

## Recent Changes

### June 2025 - Enhanced Table Processing and Commentary System
- **Multi-Column Table Reconstruction**: Enhanced prompts to preserve ALL columns from extracted tables instead of simplifying to 2 columns
- **Document Text Tabulation**: Added comprehensive tabulation of document text content into structured table format
- **Commentary Matching System**: Implemented intelligent matching between extracted data and document commentary
- **Streaming Processing**: Added real-time streaming output for progressive data processing updates
- **Enhanced Frontend Display**: Reconstructed tables now show original multi-column structure with proper headers and data rows
- **Dual Table Views**: Both reconstructed original tables and detailed data point tables are displayed
- **Document Content Tables**: Narrative text is now analyzed and tabulated into structured format alongside extracted tables
- **Model Change**: Uses GPT-4o for all AI processing for highest quality results
- **Commentary Summarization**: Added automatic summarization of long commentaries using GPT-4o
- **Superscript Filtering**: Added filtering to ignore superscript numbers and footnote markers in data extraction
- **Cost Calculator**: Implemented comprehensive cost tracking for all LLM calls with detailed token usage and dollar costs
- **Dependency Cleanup**: Removed unnecessary libraries (crewai, langchain) that were making unauthorized OpenAI API calls

### Architecture Improvements
- **Advanced LLM Processing**: Updated prompts to extract comprehensive data while maintaining table structure integrity
- **Asynchronous Commentary Matching**: Each data point is matched against document text to find relevant explanations
- **Enhanced Data Structure**: Extended data model to include table headers, rows, commentary, and metadata
- **Real-time Progress Updates**: Streaming endpoint provides live processing status updates
- **Model Optimization**: All OpenAI API calls now use GPT-4o for highest quality results

## Development Notes

1. The application uses GPT-4o as the language model for highest quality processing
2. Uses Amazon Textract for advanced PDF text and table extraction
3. Export functionality supports JSON, CSV, and PDF formats
4. All processed data is session-based and not persisted between sessions
5. Commentary matching uses intelligent text analysis to relate document content to extracted data
6. Commentary over 300 characters is automatically summarized to keep output concise
7. Simple field-value table format for better readability
8. Comprehensive cost tracking displays individual call costs and total session spend in real-time

### December 2025 - Migration and Model Updates
- **Replit Migration**: Successfully migrated from Replit Agent to standard Replit environment
- **Security Enhancements**: Implemented proper API key management and secure credential storage
- **Model Upgrade**: Switched all AI processing to use GPT-4o for highest quality results
- **Performance Optimization**: Fixed commentary matching performance issues that were causing crashes
- **Dependency Management**: Updated package management and workflow configuration for Replit compatibility