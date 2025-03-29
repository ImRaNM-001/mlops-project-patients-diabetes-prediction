# Use Python 3.12 slim image as the base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3100 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 

# Copy app requirements first to leverage Docker cache (layer caching)
COPY flask_app/app_requirements.txt .

# Install dependencies
RUN pip3 install --no-cache-dir -r app_requirements.txt && \
    rm app_requirements.txt

# Copy application code
COPY . .

# Create non-root user for security and set appropriate permissions
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose the port the app runs on
EXPOSE ${PORT}

# Healthcheck to verify the service is running properly
HEALTHCHECK --interval=20s --timeout=5s --start-period=5s --retries=1 \
    CMD curl -f http://0.0.0.0:${PORT}/ || exit 1

# Run the application
ENTRYPOINT ["python3"]

# for a typical localhost
CMD ["flask_app/app.py"]

# for prod servers
# CMD ["gunicorn", "--bind", "0.0.0.0:3100", "--timeout", "120", "app:app"]