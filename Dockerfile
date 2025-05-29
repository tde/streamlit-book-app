FROM python:3.10-slim

WORKDIR /app

# Set stable Ubuntu mirrors
RUN echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Increase apt-get timeout
RUN echo 'Acquire::http::Timeout "60";' > /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::ftp::Timeout "60";' >> /etc/apt/apt.conf.d/99timeout

# Update and install dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy torch .whl file (manually downloaded)
COPY torch-2.1.1+cu121-cp310-cp310-linux_x86_64.whl /app/

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir /app/torch-2.1.1+cu121-cp310-cp310-linux_x86_64.whl

# Copy handler
COPY handler.py .

# RunPod Serverless handler
CMD ["python", "handler.py"]