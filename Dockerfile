# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose API port
EXPOSE 8000

# Start API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
