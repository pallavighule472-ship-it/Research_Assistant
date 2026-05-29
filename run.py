"""
run.py - Research Assistant Orchestrator
Launches both the FastAPI backend and Streamlit frontend concurrently.

Usage:
    python run.py
"""

import subprocess
import sys
import os
import time
import signal
import threading
import webbrowser
import urllib.request

# ─── Configuration ────────────────────────────────────────────────────────────
BACKEND_HOST  = "127.0.0.1"
BACKEND_PORT  = 8000
FRONTEND_PORT = 8501
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ─── Color Helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def print_banner():
    print(f"""
{BOLD}{CYAN}
╔══════════════════════════════════════════════════════════╗
║          🔬  AI Research Assistant - Launcher            ║
║                                                          ║
║   FastAPI  Backend  →  http://{BACKEND_HOST}:{BACKEND_PORT}         ║
║   Streamlit Frontend →  http://localhost:{FRONTEND_PORT}         ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")

# ─── Process Runners ──────────────────────────────────────────────────────────
_shutdown = threading.Event()

def stream_output(process, label, color):
    """Streams stdout from a subprocess to the console with a colored label prefix."""
    try:
        for line in iter(process.stdout.readline, b""):
            if _shutdown.is_set():
                break
            try:
                print(f"{color}[{label}]{RESET} {line.decode('utf-8', errors='ignore').rstrip()}", flush=True)
            except (ValueError, OSError):
                break
    except Exception:
        pass

def launch_backend():
    """Starts the FastAPI server using uvicorn."""
    print(f"{GREEN}{BOLD}[LAUNCHER] Starting FastAPI Backend on port {BACKEND_PORT}...{RESET}")
    # Use a self-contained Python command that injects sys.path before importing uvicorn.
    # This is the most reliable approach on Windows since --reload spawns child processes
    # that don't always inherit modified PYTHONPATH values.
    cmd = (
        f"import sys; sys.path.insert(0, r'{SCRIPT_DIR}'); "
        f"import uvicorn; "
        f"uvicorn.run('Research_Backend_Server:server', "
        f"host='{BACKEND_HOST}', port={BACKEND_PORT}, reload=False)"
    )
    return subprocess.Popen(
        [sys.executable, "-c", cmd],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

def launch_frontend():
    """Starts the Streamlit frontend."""
    print(f"{CYAN}{BOLD}[LAUNCHER] Starting Streamlit Frontend on port {FRONTEND_PORT}...{RESET}")
    return subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run",
            "Research_Frontend.py",
            "--server.port", str(FRONTEND_PORT),
            "--server.headless", "true"
        ],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

# ─── Main Entrypoint ──────────────────────────────────────────────────────────
def main():
    print_banner()

    backend_proc = launch_backend()

    print(f"{YELLOW}[LAUNCHER] Waiting for backend to be ready...{RESET}")
    deadline = time.time() + 30
    ready = False
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://{BACKEND_HOST}:{BACKEND_PORT}/health", timeout=1)
            ready = True
            break
        except Exception:
            time.sleep(0.5)
    if ready:
        print(f"{GREEN}[LAUNCHER] Backend is ready.{RESET}")
    else:
        print(f"{YELLOW}[LAUNCHER] Backend health check timed out — launching frontend anyway.{RESET}")

    frontend_proc = launch_frontend()

    backend_thread  = threading.Thread(target=stream_output, args=(backend_proc,  "FastAPI  ", GREEN), daemon=False)
    frontend_thread = threading.Thread(target=stream_output, args=(frontend_proc, "Streamlit", CYAN),  daemon=False)
    backend_thread.start()
    frontend_thread.start()

    print(f"\n{BOLD}{YELLOW}[LAUNCHER] Both services are running! Press Ctrl+C to stop all.{RESET}\n")

    def open_browser():
        time.sleep(4)
        print(f"{CYAN}[LAUNCHER] Opening browser → http://localhost:{FRONTEND_PORT}{RESET}")
        webbrowser.open(f"http://localhost:{FRONTEND_PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    def _stop_all():
        _shutdown.set()
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (backend_proc, frontend_proc):
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        for proc in (backend_proc, frontend_proc):
            try:
                proc.stdout.close()
            except Exception:
                pass
        backend_thread.join(timeout=2)
        frontend_thread.join(timeout=2)

    def shutdown(sig, frame):
        print(f"\n{RED}{BOLD}[LAUNCHER] Shutting down all services...{RESET}")
        _stop_all()
        print(f"{RED}[LAUNCHER] All services stopped. Goodbye!{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Block the main thread while services run
    backend_proc.wait()
    _stop_all()

if __name__ == "__main__":
    main()
