"""
Backend Runner Script
Starts the FastAPI application on http://localhost:8000.
"""

import sys
from pathlib import Path
import uvicorn

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

if __name__ == "__main__":
    print("Starting ASL Translator FastAPI Backend on http://localhost:8000 ...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
