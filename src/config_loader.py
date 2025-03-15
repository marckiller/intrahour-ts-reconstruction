import yaml

def load_config(config_path="config/config.yaml"):
    """Loads configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config