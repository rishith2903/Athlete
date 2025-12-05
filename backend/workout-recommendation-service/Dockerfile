FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p models/trained data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/trained/
ENV PORT=8001

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "api/main.py"]