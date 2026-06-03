#!/bin/bash
# Start the FastAPI backend
set -a
source .env 2>/dev/null
set +a

uvicorn app.main:app --reload --port 8000
