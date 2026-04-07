FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8501"]
