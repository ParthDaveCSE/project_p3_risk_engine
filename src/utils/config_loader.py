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


def validate_config_on_startup(path="config/clinical_rules.yaml"):
    """Validate config has all required sections before pipeline starts."""
    logger = get_logger(__name__)
    try:
        config = load_config(path)
        required_sections = ["parameters", "confidence", "system_settings"]
        for section in required_sections:
            if section not in config:
                logger.error(f"STARTUP FAILURE: Required config section '{section}' missing from {path}")
                raise SystemExit(1)

        required_parameters = ["hemoglobin", "glucose", "wbc", "platelets", "creatinine"]
        for param in required_parameters:
            if param not in config["parameters"]:
                logger.error(f"STARTUP FAILURE: Required parameter '{param}' missing from config parameters section")
                raise SystemExit(1)

        logger.info(f"Config validation passed | {len(config['parameters'])} parameters loaded")
        return config

    except Exception as e:
        logger.error(f"STARTUP FAILURE: Config loading error - {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    rules = load_config()
    print(f"Loaded {len(rules['parameters'])} parameters")