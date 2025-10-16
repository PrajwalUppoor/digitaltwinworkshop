FROM python:3.10-slim

WORKDIR /app

# Install system deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application package and the model file to avoid huge build contexts
COPY app /app/app
COPY smart_home_model.pkl /app/smart_home_model.pkl
COPY digital_twin_history.csv /app/data/digital_twin_history.csv
COPY docker-compose.yml /app/docker-compose.yml

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
