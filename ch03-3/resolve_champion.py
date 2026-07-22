"""MLflow Model Registry에서 champion 별칭이 가리키는 모델 버전을 조회하는 스크립트.

배포 워크플로우에서 이 스크립트가 실패하면(별칭 없음) 이후 ECS 배포 job이
실행되지 않는다. champion 별칭이 검증되지 않은 상태에서 배포가 진행되는
상황을 막는 배포 조건(Deployment Condition) 역할을 한다.

사용 예:
    python resolve_champion.py --model-name iris-classifier --alias champion
"""

import argparse
import os

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def resolve_alias(model_name: str, alias: str) -> None:
    """별칭이 가리키는 모델 버전과 run_id를 조회해 표준 출력에 남긴다."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    try:
        version = client.get_model_version_by_alias(model_name, alias)
    except MlflowException as exc:
        raise SystemExit(
            f"{model_name}에 '{alias}' 별칭이 없습니다. 배포를 진행할 수 없습니다. ({exc})"
        )

    print(f"model={model_name} alias={alias} version={version.version} run_id={version.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="iris-classifier", help="조회할 등록 모델 이름")
    parser.add_argument("--alias", default="champion", help="조회할 별칭")
    args = parser.parse_args()

    resolve_alias(args.model_name, args.alias)
