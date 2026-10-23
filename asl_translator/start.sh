#!/usr/bin/env bash
# ==============================================================================
# ASL Translator - Unified Production Launcher
# Launches FastAPI Backend (Port 8000) & Vite React Frontend (Port 5173)
# ==============================================================================

set -e

# Resolve paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "============================================================"
echo " Starting ASL-to-English Digital Image Processing Translator"
echo "============================================================"
echo " Project Root : $ROOT_DIR"
echo " Backend Dir  : $BACKEND_DIR"
echo " Frontend Dir : $FRONTEND_DIR"
echo "============================================================"

# Find Python executable
if [ -f "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3" ]; then
    PYTHON_EXE="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
elif command -v python3 &> /dev/null; then
    PYTHON_EXE="python3"
else
    echo "Error: python3 not found."
    exit 1
fi

echo "Using Python: $PYTHON_EXE"

# 1. Start Backend in background
echo "-> Starting FastAPI Backend on http://localhost:8000..."
cd "$BACKEND_DIR"
$PYTHON_EXE run_backend.py &
BACKEND_PID=$!

# Trap signals to cleanly terminate both processes
cleanup() {
    echo ""
    echo "Shutting down ASL Translator..."
    kill $BACKEND_PID 2>/dev/null || true
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Give backend a moment to initialize
sleep 2

# 2. Start Frontend
echo "-> Starting Vite React Frontend on http://localhost:5173..."
cd "$FRONTEND_DIR"

if command -v npm &> /dev/null; then
    npm run dev &
    FRONTEND_PID=$!
else
    echo "Error: npm not found."
    exit 1
fi

echo ""
echo "============================================================"
echo " System Online!"
echo " Web UI   : http://localhost:5173"
echo " API Docs : http://localhost:8000/docs"
echo " Health   : http://localhost:8000/api/health"
echo "============================================================"
echo "Press Ctrl+C to stop all servers."
echo ""

wait $FRONTEND_PID
