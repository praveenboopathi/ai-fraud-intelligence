from pathlib import Path

from fraud_intelligence.utils.config import load_yaml
from fraud_intelligence.utils.reproducibility import set_global_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_project_config_exists() -> None:
    """Verify that the main project configuration exists."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    assert config_path.exists()


def test_config_loads() -> None:
    """Verify that the project configuration can be loaded."""
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    config = load_yaml(config_path)

    assert config["project"]["name"] == "ai-fraud-intelligence"
    assert config["project"]["random_seed"] == 42


def test_seed_function_runs() -> None:
    """Verify that the reproducibility utility executes."""
    set_global_seed(42)