import os
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelArtifactNotFoundError(Exception):
    """Raised when model artifact file is missing."""
    pass


def load_model(version="1.0.0"):
    """Stub for L8 - will load trained model."""
    model_dir = "models/artifacts"
    path = f"{model_dir}/risk_engine_v{version}.joblib"

    if not os.path.exists(path):
        logger.error(f"STARTUP FAILURE: Model artifact not found: {path}")
        raise ModelArtifactNotFoundError(f"Model artifact not found: {path}")

    logger.info(f"Model loaded from {path}")
    return None  # Will return actual model in L8