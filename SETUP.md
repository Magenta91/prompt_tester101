# Setup Guide

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Magenta91/prompt_tester101.git
cd prompt_tester101
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
# Copy the template
cp .env.template .env

# Edit .env file with your API keys
# At minimum, you need to set OPENAI_API_KEY
```

### 5. Configure Your API Keys

#### OpenAI API Key (Required)
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key and paste it in your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

#### AWS Credentials (Optional)
If you want to use Amazon Textract for better PDF processing:
1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/)
2. Create a new user with Textract permissions
3. Generate access keys
4. Add to your `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   AWS_DEFAULT_REGION=us-east-1
   ```

**Note**: If AWS credentials are not provided, the system will automatically fall back to Tesseract OCR.

### 6. Install Tesseract OCR (Fallback Option)

#### Windows:
1. Download from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to PATH
3. Or use chocolatey: `choco install tesseract`

#### macOS:
```bash
brew install tesseract
```

#### Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr
```

### 7. Run the Application
```bash
python app.py
```

### 8. Access the Application
Open your browser and go to: `http://localhost:5000`

## Testing the Setup

1. **Upload a PDF**: Use the sample PDF included in the repository
2. **Process Document**: Click "Process with AI" to extract data
3. **Test Prompts**: Use the "Test Context Extraction Prompt" section
4. **Export Data**: Try exporting to XLSX, CSV, or JSON

## Troubleshooting

### Common Issues:

#### "No module named 'xyz'"
```bash
pip install -r requirements.txt
```

#### "OpenAI API key not found"
- Check your `.env` file exists
- Verify `OPENAI_API_KEY` is set correctly
- Make sure there are no extra spaces or quotes

#### "Tesseract not found"
- Install Tesseract OCR (see step 6 above)
- Add Tesseract to your system PATH

#### "AWS credentials not found"
- This is normal if you haven't set up AWS
- The system will automatically use Tesseract OCR instead

### Getting Help

1. Check the [README.md](README.md) for detailed documentation
2. Review the [OpenAI Prompts Documentation](OpenAI_Prompts_Documentation.md)
3. Open an issue on GitHub if you encounter problems

## Development Setup

For development work:

1. Set `FLASK_ENV=development` in your `.env` file
2. Set `LOG_LEVEL=DEBUG` for detailed logging
3. The application will auto-reload on code changes

## Production Deployment

For production deployment:

1. Set `FLASK_ENV=production` in your `.env` file
2. Change the `SECRET_KEY` to a secure random value
3. Use a proper WSGI server like Gunicorn
4. Set up proper logging and monitoring

---

**You're all set! 🚀**