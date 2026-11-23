# AWS Lambda container image: a normal FastAPI/uvicorn app plus the AWS Lambda Web Adapter extension.
# The same image also runs locally:  docker run -p 8080:8080 malaria-risk
FROM --platform=linux/amd64 public.ecr.aws/docker/library/python:3.13-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.0 /lambda-adapter /opt/extensions/lambda-adapter

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 AWS_LWA_READINESS_CHECK_PATH=/health MAX_UPLOAD_BYTES=3000000
WORKDIR /var/task

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Code, trained model and aggregated cluster scores only. No raw or processed DHS data (see .dockerignore).
COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
