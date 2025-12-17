#!/bin/sh

echo "Starting backend..."
cd backend
./start.sh &

echo "Starting frontend..."
cd ../frontend
streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 &

wait
