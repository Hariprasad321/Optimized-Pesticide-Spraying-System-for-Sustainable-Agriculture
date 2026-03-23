#!/usr/bin/env python3
"""
AgriSprayAI - Simple Startup Script
"""
import uvicorn
import os
import sys

def main():
    print("🌾 AgriSprayAI - Pest Detection & Spraying Optimization")
    print("=" * 60)
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ ERROR: app.py not found!")
        print("Please make sure you're in the correct directory.")
        sys.exit(1)
    
    # Check if static directory exists
    if not os.path.exists("static"):
        print("❌ ERROR: static/ directory not found!")
        print("Creating static directory...")
        os.makedirs("static", exist_ok=True)
    
    # Check if models directory exists
    if not os.path.exists("models"):
        print("⚠️  WARNING: models/ directory not found!")
        print("Creating models directory...")
        os.makedirs("models", exist_ok=True)
        print("Note: Place your YOLO model (best.pt) in models/ directory")
    
    print("✅ Starting AgriSprayAI server...")
    print("📱 Open your browser: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()