FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for C-extensions if needed, and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory structures that are required
RUN mkdir -p data/raw data/processed models logs

# Expose the API server port
EXPOSE 8000

# Set environment variable to make imports resolved correctly
ENV PYTHONPATH=/app

# Command to run the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
