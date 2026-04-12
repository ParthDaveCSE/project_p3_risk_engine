import yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(path="config/clinical_rules.yaml"):
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise

if __name__ == "__main__":
    rules = load_config()
    print(f"Loaded {len(rules['parameters'])} parameters")