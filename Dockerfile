FROM python:3.10-slim

WORKDIR /app

# Set stable Ubuntu mirrors
RUN echo "deb http://archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# Increase apt-get timeout
RUN echo 'Acquire::http::Timeout "60";' > /etc/apt/apt.conf.d/99timeout && \
    echo 'Acquire::ftp::Timeout "60";' >> /etc/apt/apt.conf.d/99timeout

# Установка необходимых пакетов
RUN apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        libopenblas-dev

# Очистка временных файлов apt
RUN apt-get clean

# Удаление списков репозиториев
RUN rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# Copy handler
COPY handler.py .

# RunPod Serverless handler
CMD ["python", "handler.py"]