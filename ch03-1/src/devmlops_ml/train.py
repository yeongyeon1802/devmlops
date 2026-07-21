"""iris 데이터셋으로 분류 모델을 학습하고 파일로 저장하는 스크립트.

학습 결과는 두 파일이 생성된다.
- model/model.joblib  : 학습된 모델 (서빙 API가 시작할 때 읽는다)
- model/metrics.json  : 평가 지표 (CI의 모델 검증 단계가 읽는다)
"""

import json
import os
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 모델과 지표를 저장할 폴더. 명령을 실행한 위치(작업 디렉터리) 기준이며,
# 환경 변수 MODEL_DIR로 바꿀 수 있다. Docker에서는 WORKDIR 아래에 만들어진다.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))


def train() -> dict[str, float]:
    """모델을 학습하고 정확도를 반환한다."""
    # 1. 데이터 준비: iris는 꽃받침·꽃잎 치수 4개로 품종 3종을 맞추는 데이터이다.
    data = load_iris()

    # 2. 학습용과 평가용을 분리한다. 같은 데이터로 학습과 평가를 하면
    #    실제 성능보다 점수가 부풀려지므로 반드시 나눈다.
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0
    )

    # 3. 로지스틱 회귀로 학습한다. 작은 데이터에 충분하고 결과 해석이 쉽다.
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)

    # 4. 평가용 데이터로 정확도를 계산한다.
    accuracy = accuracy_score(y_test, model.predict(x_test))

    # 5. 모델과 지표를 파일로 저장한다.
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")

    metrics = {"accuracy": round(float(accuracy), 4)}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return metrics


if __name__ == "__main__":
    result = train()
    print(f"model saved. accuracy={result['accuracy']}")
