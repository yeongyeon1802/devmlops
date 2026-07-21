"""서빙 API 동작 테스트. 테스트 시작 전에 모델을 학습해 파일을 준비한다."""

from fastapi.testclient import TestClient

from devmlops_ml import train

# import 순서가 중요하다. main은 import 시점에 모델 파일을 읽으므로,
# 학습을 먼저 실행해 model.joblib을 만들어 둔다.
train.train()

from devmlops_ml.main import app  # noqa: E402

client = TestClient(app)


def test_health_returns_ok_when_model_loaded() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_known_class() -> None:
    # setosa의 전형적인 치수. 어떤 학습 결과에서도 setosa로 분류된다.
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == "setosa"
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_negative_input() -> None:
    # 음수 치수는 물리적으로 불가능하므로 422 검증 오류가 나야 한다.
    response = client.post(
        "/predict",
        json={
            "sepal_length": -1.0,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 422
