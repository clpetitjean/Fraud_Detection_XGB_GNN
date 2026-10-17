import yaml

from fraud_detection.data.load import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
