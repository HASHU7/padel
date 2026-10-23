FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and models
COPY padelcoach/ ./padelcoach/
COPY models/ ./models/
COPY README.md .

EXPOSE 8000

CMD ["python", "-m", "padelcoach", "serve", "--host", "0.0.0.0", "--port", "8000"]
