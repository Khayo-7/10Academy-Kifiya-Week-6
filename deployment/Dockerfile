# Building stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY deployment/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Finalizing stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# # Copy the necessary files and directories
# COPY . .
# COPY deployment/logs /app/logs
COPY deployment/app /app/app
COPY deployment/main.py /app/main.py
# COPY scripts /app/scripts
# COPY dashboard /app/dashboard
# COPY resources/data/store_processed.csv /app/resources/store.csv
# COPY resources/encoders/onehot_encoder.pkl /app/resources/encoder.pkl
# COPY resources/scalers/scaler.pkl /app/resources/scaler.pkl
# COPY resources/models/credit_scores_model.pkl /app/resources/credit_scores_model.pkl
# COPY resources/models/credit_scores_model.h5 /app/resources/credit_scores_model.h5

# EXPOSE 7777 8888 9999
# ENV NAME env


# Command to run all three services
# CMD ["sh", "-c", "streamlit run /app/dashboard/app.py --server.port 9999 & uvicorn app.main:app --host 0.0.0.0 --port 8888 & python app/main.py"]
# CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "7777"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8888"]
# CMD ["streamlit", "run", "app.py", "--host", "0.0.0.0", "--port", "9999"]

# docker build -t credit-scores-api .
# docker run -p 7777:7777 credit-scores-api

# docker build -f deployment/Dockerfile -t credit-scores-app .
# docker run -d -p 7777:7777 -p 8888:8888 -p 9999:9999 credit-scores-app
