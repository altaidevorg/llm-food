# Use a uv base image with Python 3.10
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# ENV UV_SYSTEM_PYTHON=1 # Alternative to using --system flag in each command

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# e.g., pymupdf might need some C libraries. 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential poppler-utils \
    # Add other system dependencies if needed by your Python packages, e.g., for OCR: tesseract-ocr
    && rm -rf /var/lib/apt/lists/*

# Copy the project
COPY . .

# Install the package using uv
# Use --system to install into the system Python environment within the container
# Use --no-cache to avoid caching, similar to pip's --no-cache-dir
RUN uv pip install --system --no-cache .[server]

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# Use 0.0.0.0 to allow external connections to the container
CMD ["llm-food-serve"]