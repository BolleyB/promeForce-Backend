# -----------------------------
# 1st Stage: Build dependencies
# -----------------------------
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies in a virtual environment
COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 2nd Stage: Final image
# -----------------------------
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy only necessary runtime dependencies from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY . .

# Set the environment for using the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
