"""학습된 iris 모델을 API로 제공하는 FastAPI 서빙 애플리케이션.

ch03-1의 main.py에 추론 로깅(Inference Logging)을 추가한 버전이다. 이 파일로
ch03-1/code/src/devmlops_ml/main.py를 교체한다. 엔드포인트 동작과 응답 형식은
ch03-1과 동일하며, `/predict` 요청마다 입력값과 예측 결과를 로그로 남기는
부분만 추가됐다. 이 로그는 ECS 태스크 정의의 awslogs 드라이버를 통해
CloudWatch Logs(`/ecs/devmlops-fastapi`)에 그대로 쌓인다.
"""

import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from devmlops_ml import __version__

# train.py가 저장한 위치와 같은 기준(작업 디렉터리)으로 모델을 찾는다.
MODEL_PATH = Path(os.environ.get("MODEL_DIR", "model")) / "model.joblib"

# iris 품종 번호(0, 1, 2)를 사람이 읽을 수 있는 이름으로 바꾸기 위한 목록이다.
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# uvicorn이 표준 출력으로 보내는 로그는 awslogs 드라이버가 그대로 수집하므로,
# 별도 로그 저장소 설정 없이 logging 모듈만 사용한다.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("devmlops_ml.inference")
logger.setLevel(logging.INFO)

app = FastAPI(
    title="DevMLOps Iris Serving",
    version=__version__,
    description="Scikit-learn iris model serving API for ML CI/CD practice.",
)

# 앱 시작 시점에 모델을 한 번만 읽어 메모리에 올린다.
# 요청마다 파일을 읽으면 응답이 느려지므로 전역으로 유지한다.
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class IrisFeatures(BaseModel):
    """예측 요청 본문. iris의 입력 특성 4개를 cm 단위로 받는다."""

    sepal_length: float = Field(..., ge=0, description="꽃받침 길이(cm)")
    sepal_width: float = Field(..., ge=0, description="꽃받침 너비(cm)")
    petal_length: float = Field(..., ge=0, description="꽃잎 길이(cm)")
    petal_width: float = Field(..., ge=0, description="꽃잎 너비(cm)")


@app.get("/")
def read_root() -> dict[str, str]:
    """서비스 접근 확인용 기본 응답."""
    return {
        "service": "devmlops-ml",
        "message": "Iris model serving API is running.",
    }


@app.get("/health")
def read_health() -> dict[str, str]:
    """헬스체크 응답. 모델이 로드되어야 서비스 가능 상태로 판단한다."""
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "version": __version__,
    }


@app.post("/predict")
def predict(features: IrisFeatures) -> dict[str, object]:
    """입력 특성 4개로 iris 품종을 예측하고, 입력과 결과를 로그로 남긴다."""
    if model is None:
        # 모델 파일 없이 컨테이너가 뜬 경우. 빌드 단계의 학습 누락이 원인이다.
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]
    class_index = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][class_index])
    prediction = CLASS_NAMES[class_index]

    # 추론 로깅: 데이터 드리프트나 이상 예측을 의심할 때 이 시점의
    # 실제 입력 분포와 예측 결과를 다시 확인할 수 있도록 남긴다.
    logger.info(
        "predict request=%s prediction=%s probability=%.4f",
        features.model_dump(),
        prediction,
        probability,
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "model_version": __version__,
    }
