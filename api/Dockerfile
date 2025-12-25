# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files
COPY ./app.py /app/
COPY ./artifacts /app/artifacts/
COPY ./requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
