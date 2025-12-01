# Use official Python 3.13 slim image
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS deps required for some packages and streamlit health check
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user to run the app
RUN adduser --disabled-password --gecos '' --uid 1000 streamlit \
    && mkdir -p /home/streamlit/.streamlit \
    && chown -R streamlit:streamlit /home/streamlit

# Copy only requirements and install them to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY . /app

# Make sure files are owned by the streamlit user
RUN chown -R streamlit:streamlit /app

USER streamlit

# Expose streamlit default port
EXPOSE 8501

# Healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Streamlit server - use the entrypoint script path in this repo
ENTRYPOINT ["streamlit", "run", "app/web/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
