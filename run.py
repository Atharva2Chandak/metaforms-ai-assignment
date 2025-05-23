# run.py
import subprocess
import time
import webbrowser
import os

def start_servers():
    print("Starting FastAPI server...")
    fastapi_process = subprocess.Popen([
        "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])
    
    # Wait for FastAPI to start
    time.sleep(3)
    
    print("Starting Streamlit UI...")
    streamlit_process = subprocess.Popen([
        "streamlit", "run", "app.py", "--server.port", "8501"
    ])
    
    # Wait for Streamlit to start
    time.sleep(3)
    
    # Open browser
    webbrowser.open("http://localhost:8501")
    
    try:
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("Shutting down servers...")
        fastapi_process.terminate()
        streamlit_process.terminate()

if __name__ == "__main__":
    start_servers()
