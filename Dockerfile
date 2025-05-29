FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
        git \
        cmake \
        build-essential \
        libopenblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

CMD ["python", "handler.py"]