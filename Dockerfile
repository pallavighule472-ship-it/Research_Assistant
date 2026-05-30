FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

COPY . .

EXPOSE 8000 8501

CMD ["streamlit", "run", "Research_Frontend.py", "--server.port", "8000", "--server.address", "0.0.0.0", "--server.headless", "true"]
