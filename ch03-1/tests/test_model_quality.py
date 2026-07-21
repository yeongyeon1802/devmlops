"""모델 성능 임계값 검증 테스트.

일반 코드 테스트와 달리 이 테스트는 "모델의 성능"을 검사한다.
정확도가 임계값보다 낮으면 테스트가 실패하고, CI 파이프라인이 배포를 차단한다.

임계값은 환경 변수 MODEL_ACCURACY_THRESHOLD로 조절할 수 있다.
실습에서는 이 값을 1.0으로 올려 배포 차단 상황을 재현한다.
"""

import os

from devmlops_ml import train

# 기본 임계값 0.9: iris + 로지스틱 회귀는 보통 0.95 이상이 나온다.
THRESHOLD = float(os.environ.get("MODEL_ACCURACY_THRESHOLD", "0.9"))


def test_model_accuracy_meets_threshold() -> None:
    metrics = train.train()

    assert metrics["accuracy"] >= THRESHOLD, (
        f"모델 정확도 {metrics['accuracy']}가 임계값 {THRESHOLD}보다 낮다. "
        "배포를 진행하면 안 되는 상태이다."
    )
