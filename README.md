# PDF Data Extractor with AI Context - Production

🚀 **Production-ready PDF processing system with AI-powered context generation and custom prompt testing.**

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template-id)

## 🌟 Features

- **PDF Upload & Processing** - Extract tables, key-values, and text from PDFs
- **AI Context Generation** - Intelligent context matching using OpenAI GPT
- **Custom Prompt Testing** - Real-time testing of custom context extraction prompts
- **Export Functionality** - XLSX and CSV export with proper formatting
- **Ultra-Performance** - 90% local processing, 10% AI for optimal speed and cost
- **Production Ready** - Optimized for Railway deployment with health checks

## 🚀 Quick Deploy to Railway

1. **Click the Deploy button above** or manually deploy:
   - Fork this repository
   - Connect to Railway
   - Set environment variables
   - Deploy!

2. **Required Environment Variables:**
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_for_sessions
   ```

3. **Optional Environment Variables:**
   ```
   AWS_ACCESS_KEY_ID=your_aws_access_key (for Textract)
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key (for Textract)
   AWS_DEFAULT_REGION=us-east-1
   ```

## 📋 Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Magenta91/prompt_tester101.git -b prod2
   cd prompt_tester101
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the application:**
   ```
   http://localhost:5000
   ```

## 🎯 Usage

### Basic PDF Processing:
1. Upload a PDF file
2. Click "Process with AI"
3. View extracted data with AI-generated context
4. Export results as XLSX or CSV

### Custom Prompt Testing:
1. Upload and process a PDF
2. Enter a custom prompt in the testing section
3. Click "Test Custom Prompt"
4. Compare results with different prompts
5. Export custom prompt results

### Example Custom Prompts:
```
# Business Analysis
Explain each data point in business terms and why it's important for investors.

# Creative Context
Write a short story about how each data point came to be.

# Technical Analysis
Provide technical analysis of each metric including trends and implications.
```

## 🏗️ Architecture

- **Frontend**: HTML5, Bootstrap 5, Vanilla JavaScript
- **Backend**: Flask (Python)
- **AI Processing**: OpenAI GPT-4o-mini
- **PDF Processing**: Tesseract OCR + AWS Textract (fallback)
- **Export**: openpyxl for XLSX, native CSV
- **Deployment**: Railway with Gunicorn

## 📊 Performance

- **Processing Time**: 5-10 seconds (vs 50+ seconds traditional)
- **Cost Efficiency**: 95% reduction in AI costs
- **Local Processing**: 90% of entities matched locally
- **AI Usage**: Only for ambiguous cases and custom prompts
- **Success Rate**: 90-95% entities with context

## 🔧 Configuration

### Railway Environment Variables:
```bash
# Required
OPENAI_API_KEY=sk-...
SECRET_KEY=your-secret-key

# Optional
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
FLASK_ENV=production
```

### Health Check:
- **Endpoint**: `/health`
- **Response**: `{"status": "healthy", "service": "PDF Extractor"}`

## 🚨 Production Notes

- **File Upload Limit**: 50MB
- **Session Storage**: Temporary files for prompt testing
- **Error Handling**: Comprehensive error recovery
- **Logging**: Detailed server-side logging
- **Security**: Environment-based configuration

## 🛠️ API Endpoints

- `GET /` - Main application interface
- `GET /health` - Health check for Railway
- `POST /extract` - PDF upload and extraction
- `POST /process_stream` - AI processing with streaming
- `POST /test_prompt` - Custom prompt testing
- `POST /export_xlsx` - XLSX export
- `POST /export_csv` - CSV export

## 📈 Monitoring

The application includes:
- Health check endpoint for Railway monitoring
- Comprehensive error logging
- Performance metrics tracking
- Cost analysis and optimization stats

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Check the logs in Railway dashboard
- Verify environment variables are set
- Ensure OpenAI API key has sufficient credits
- Check file size limits (50MB max)

---

**Built with ❤️ for efficient PDF processing and AI-powered data extraction.**