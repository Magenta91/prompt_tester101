#!/usr/bin/env python3
"""
Alternative way to run the Flask app without watchdog restart issues
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Set environment variables to prevent watchdog issues
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    print("🚀 Starting PDF Extractor App...")
    print("📍 Access at: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Run without reloader to prevent infinite restart
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)