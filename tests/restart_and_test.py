"""
Quick script to test if API is running and test it
"""
import time
import requests
import subprocess
import sys

def wait_for_api(max_wait=10):
    """Wait for API to be ready."""
    print("Waiting for API to be ready...")
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/api/v1/models/status", timeout=1)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except:
            time.sleep(1)
            print(f"  Waiting... ({i+1}/{max_wait})")
    return False

if __name__ == "__main__":
    if wait_for_api():
        print("\nRunning tests...\n")
        subprocess.run([sys.executable, "test_api_integration.py"])
    else:
        print("❌ API is not running. Please start it with: python backend/main.py")
