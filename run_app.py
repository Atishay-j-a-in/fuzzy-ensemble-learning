"""
fuzzyBACH - Single-Click Launcher for Backend & Vite Frontend.

Strict order: backend FIRST (wait until all 5 models are loaded),
then frontend, then open the browser.
"""

import json
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:5173"
# 5 backbones + warmup can take several minutes on CPU — poll generously.
BACKEND_READY_TIMEOUT_S = 20 * 60
POLL_INTERVAL_S = 3


def _get_json(path: str, timeout: float = 5):
    try:
        with urllib.request.urlopen(BACKEND_URL + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def wait_for_backend(backend_proc: subprocess.Popen) -> bool:
    """Block until the backend reports models loaded (or timeout / crash).

    Returns True when the backend is fully ready, False on timeout/crash
    (caller decides whether to continue anyway).
    """
    print("    Waiting for backend to finish loading all 5 models...")
    print("    (watch per-model stages below; this can take several minutes on CPU)")
    deadline = time.time() + BACKEND_READY_TIMEOUT_S
    last_msg = None
    while time.time() < deadline:
        if backend_proc.poll() is not None:
            print(f"\n    ERROR: backend process exited early with code {backend_proc.returncode}.")
            print("    Check the traceback above — likely a missing dependency or bad model file.")
            return False

        health = _get_json("/api/health")
        if health and health.get("model_loaded"):
            print("\n    Backend READY — all 5 models + fuzzy ensemble online.")
            return True

        startup = _get_json("/api/startup/progress")
        if startup and startup.get("message") != last_msg:
            last_msg = startup.get("message")
            idx, total = startup.get("stage_index", 0), startup.get("stage_total", 6)
            print(f"    [backend {idx}/{total}] {last_msg}")
        elif health is None and last_msg != "__waiting__":
            # uvicorn not accepting connections yet — print once, then stay quiet.
            last_msg = "__waiting__"
            print("    [backend] starting uvicorn, waiting for first response...")

        time.sleep(POLL_INTERVAL_S)

    print(f"\n    WARNING: backend not ready after {BACKEND_READY_TIMEOUT_S // 60} min — continuing anyway.")
    print("    The frontend will show live loading progress via the Navbar indicator.")
    return False


def main():
    print("=" * 70)
    print("  fuzzyBACH - Breast Cancer Histology Deep AI Diagnostics")
    print("  Xception 4-Class Classification & Architecture Platform")
    print("=" * 70)

    # 1. Start FastAPI Backend FIRST
    python_exe = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = sys.executable

    print(f"\n[1/3] Starting FastAPI Backend on {BACKEND_URL} ...")
    backend_proc = subprocess.Popen(
        [str(python_exe), "-u", "-m", "uvicorn", "main:app", "--app-dir", "backend",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT_DIR),
    )

    # 2. Wait until backend is FULLY ready before touching the frontend.
    backend_ready = wait_for_backend(backend_proc)
    if backend_proc.poll() is not None:
        # Backend crashed during load — don't start a frontend that can never work.
        sys.exit(1)

    # 3. Start Vite Dev Server only after backend is ready (or timed out).
    print(f"\n[{'2' if backend_ready else '2'}/3] Starting Vite React Frontend on {FRONTEND_URL} ...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(ROOT_DIR / "frontend"),
        shell=True,
    )

    time.sleep(2)
    if frontend_proc.poll() is not None:
        print(f"\n    ERROR: frontend process exited early with code {frontend_proc.returncode}.")
        print("    Is npm installed and are dependencies present? Try: cd frontend && npm install")
        backend_proc.terminate()
        sys.exit(1)

    print("\n[3/3] Launching Web Interface...")
    print(f"  -> Frontend: {FRONTEND_URL}")
    print(f"  -> Backend API & Docs: {BACKEND_URL}/docs")
    print("\nPress Ctrl+C in this terminal to stop both servers.")
    print("=" * 70)

    try:
        webbrowser.open(FRONTEND_URL)
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping fuzzyBACH servers...")
    finally:
        for proc, name in ((frontend_proc, "frontend"), (backend_proc, "backend")):
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception as e:
                print(f"    (could not stop {name}: {e})")


if __name__ == "__main__":
    main()
