#!/usr/bin/env python3
"""
Run the simple version without watchdog issues
"""

import os
import sys
from app_simple import app

if __name__ == '__main__':
    print("🚀 Starting Simple PDF Extractor...")
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