import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads the configuration from a YAML file.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    config = load_config()
    print("Configuration loaded successfully.")
