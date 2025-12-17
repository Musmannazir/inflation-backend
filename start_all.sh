#!/bin/bash

echo "Starting backend on port 8000..."
# We run this in the background (&) on localhost:8000
# NOTE: We use cd backend to ensure imports work correctly
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
cd ..

# Wait 5 seconds to ensure backend is ready before frontend starts
sleep 5

echo "Starting frontend on port 8501..."
# We run Streamlit on port 8501 (which Render listens to)
cd frontend
streamlit run app.py --server.port 8501 --server.address 0.0.0.0