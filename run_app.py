"""
fuzzyBACH - Single-Click Launcher for Backend & Vite Frontend
Starts FastAPI backend (port 8000) and Vite development server (port 5173).
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def main():
    print("=" * 70)
    print("  fuzzyBACH - Breast Cancer Histology Deep AI Diagnostics")
    print("  Xception 4-Class Classification & Architecture Platform")
    print("=" * 70)

    # 1. Start FastAPI Backend
    python_exe = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    print("\n[1/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [str(python_exe), "-m", "uvicorn", "main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
    )

    # Give backend a moment to warm up
    time.sleep(3)

    # 2. Start Vite Dev Server
    print("\n[2/3] Starting Vite React Frontend on http://localhost:5173 ...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(ROOT_DIR / "frontend"),
        shell=True,
    )

    time.sleep(2)
    print("\n[3/3] Launching Web Interface...")
    print("  -> Frontend: http://localhost:5173")
    print("  -> Backend API & Docs: http://127.0.0.1:8000/docs")
    print("\nPress Ctrl+C in this terminal to stop both servers.")
    print("=" * 70)

    try:
        webbrowser.open("http://localhost:5173")
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping fuzzyBACH servers...")
        backend_proc.terminate()
        frontend_proc.terminate()

if __name__ == "__main__":
    main()
