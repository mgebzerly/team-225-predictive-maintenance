FROM python:3.11-slim

WORKDIR /app

COPY requirements-ml.txt .

RUN pip install --no-cache-dir -r requirements-ml.txt

COPY app ./app
COPY models ./models
COPY src ./src

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]