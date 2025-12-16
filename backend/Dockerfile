# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make sure the start script is executable
RUN chmod +x start.sh

# Expose the port the app runs on (usually 8000 for FastAPI/Django or 5000 for Flask)
EXPOSE 8000

# Run the start script
CMD ["./start.sh"]