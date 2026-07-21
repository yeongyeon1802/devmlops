# syntax=docker/dockerfile:1

# 빌드 단계: 패키지 설치와 모델 학습을 여기서 끝낸다.
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# 이미지 빌드 시점에 모델을 학습한다. 이미지 안에 코드와 모델이 함께 고정되므로
# 같은 이미지 태그는 항상 같은 모델을 서빙한다.
RUN python -m devmlops_ml.train

# 실행 단계: 실행에 필요한 가상환경과 모델 파일만 가져온다.
FROM python:3.12-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/model ./model

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"

CMD ["uvicorn", "devmlops_ml.main:app", "--host", "0.0.0.0", "--port", "8000"]
