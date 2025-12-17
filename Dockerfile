FROM python:3.12-slim

WORKDIR /app

# Copy backend & frontend requirements
COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt

# Install backend deps
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install frontend deps
RUN pip install --no-cache-dir -r frontend/requirements.txt

# Copy all code
COPY . .

# Make scripts executable
RUN chmod +x backend/start.sh
RUN chmod +x start_all.sh

# Expose both ports
EXPOSE 8000
EXPOSE 8501

# Start both backend & frontend
CMD ["./start_all.sh"]
