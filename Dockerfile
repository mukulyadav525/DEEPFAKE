# Dockerfile for Drishyam Forensic Engine (v4.0.0)
FROM python:3.14-slim

# Install FFmpeg and system libraries for OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Start Uvicorn (Railway uses the $PORT env var)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
