#!/usr/bin/env python3
"""
Simple script to run the Prompt Tester
"""

import subprocess
import sys
import os

def main():
    print("🧪 Starting Prompt Tester...")
    print("=" * 50)
    
    # Check if required packages are installed
    try:
        import flask
        import pandas
        import openpyxl
        import openai
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Please install required packages:")
        print("   pip install flask pandas openpyxl openai")
        return
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  Warning: .env file not found")
        print("🔑 Make sure you have OPENAI_API_KEY set in your environment")
    
    print("🚀 Starting server on http://localhost:5001")
    print("🔧 Use this tool to test and fine-tune your context extraction prompts")
    print("📝 Left panel: Input your prompt template")
    print("📊 Right panel: See CSV output in real-time")
    print("=" * 50)
    
    try:
        # Run the Flask app
        subprocess.run([sys.executable, 'prompt_tester_backend.py'], check=True)
    except KeyboardInterrupt:
        print("\n👋 Prompt Tester stopped")
    except Exception as e:
        print(f"❌ Error running prompt tester: {e}")

if __name__ == '__main__':
    main()