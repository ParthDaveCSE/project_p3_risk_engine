"""
Model Store – Versioned Artifact Management

WHY THIS EXISTS:
- A model without its threshold is not deployable
- If stored separately, they go out of sync
- This saves both components atomically in a production bundle

DANGER ZONE:
- Latest ≠ Best ≠ Approved
- Never use load_latest_model in production code
- Production must load via explicit versions or the MLflow Model Registry
"""

import os
import glob
import re
import hashlib
import joblib
from typing import Optional, List, Dict, Any, Tuple

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

# Get model directory from config, with fallback
MODEL_DIR = config.get('paths', {}).get('model_dir', 'models/artifacts')


class ModelArtifactNotFoundError(Exception):
    """Raised when a required model artifact cannot be found."""
    pass


def _hash_config(config_path: str = "config/clinical_rules.yaml") -> str:
    """
    Compute MD5 hash of the clinical config file.
    Stored in the production bundle so that at inference time the CLI
    can detect if the config has changed since the bundle was created.
    """
    if not os.path.exists(config_path):
        return "config_not_found"
    with open(config_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_next_version(name: str) -> str:
    """
    Compute the next semantic version for a named model.
    Scans existing artifacts and increments the patch version.
    """
    existing = list_available_models()
    versions = [m['version'] for m in existing if m['name'] == name and m['type'] == 'model']

    if not versions:
        return '1.0.0'

    parsed = [tuple(map(int, v.split('.'))) for v in versions]
    latest = max(parsed)
    return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"


def save_model(model, name: str, version: Optional[str] = None) -> str:
    """
    Save a trained model artifact with versioned naming convention.

    If version is None, auto-increment the patch version.
    Explicit versions are for intentional major/minor architecture bumps.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if version is None:
        version = get_next_version(name)

    path = os.path.join(MODEL_DIR, f'{name}_v{version}.joblib')
    joblib.dump(model, path)
    logger.info(f'Model saved: {path}')
    return path


def load_model(name: str, version: str = '1.0.0'):
    """
    Load a versioned model artifact by name and version.
    """
    path = os.path.join(MODEL_DIR, f'{name}_v{version}.joblib')

    if not os.path.exists(path):
        message = (
            f"Model artifact not found: {path} | "
            f"Run the training pipeline first: "
            f"uv run python src/models/trainer.py"
        )
        logger.error(f"ARTIFACT MISSING: {message}")
        raise ModelArtifactNotFoundError(message)

    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model


def load_latest_model(name: str):
    """
    Load the most recently saved version of a named model.

    DANGER – latest != best != approved.
    Never use this in production code.
    """
    pattern = os.path.join(MODEL_DIR, f'{name}_v*.joblib')
    matches = [p for p in glob.glob(pattern) if '_bundle_' not in p]

    if not matches:
        message = (
            f"No model artifacts found for {name} in {MODEL_DIR} | "
            f"Run the training pipeline first."
        )
        logger.error(f"ARTIFACT MISSING: {message}")
        raise ModelArtifactNotFoundError(message)

    def version_key(path: str) -> Tuple[int, int, int]:
        filename = os.path.basename(path)
        match = re.search(r'_v(\d+\.\d+\.\d+)\.joblib$', filename)
        if match:
            return tuple(int(x) for x in match.group(1).split('.'))
        return (0, 0, 0)

    latest_path = max(matches, key=version_key)
    model = joblib.load(latest_path)
    logger.info(f"Latest model loaded: {latest_path} | WARNING: latest != approved for production")
    return model


def save_production_bundle(
    model,
    threshold: float,
    name: str,
    version: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save model and threshold together as a single deployable artifact.

    WHY THIS EXISTS:
    A model without its threshold is not deployable.
    If stored separately, they go out of sync.
    This saves both components atomically.

    CONFIG HASH:
    The bundle also stores the MD5 hash of clinical_rules.yaml at
    training time. The CLI compares this to the current config hash
    at load time and logs a warning if the config changed since training.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if version is None:
        existing_bundles = glob.glob(os.path.join(MODEL_DIR, f'{name}_bundle_v*.joblib'))

        if existing_bundles:
            def version_key(path: str) -> Tuple[int, int, int]:
                match = re.search(r'_bundle_v(\d+\.\d+\.\d+)\.joblib$', path)
                if match:
                    return tuple(int(x) for x in match.group(1).split('.'))
                return (0, 0, 0)

            latest_path = max(existing_bundles, key=version_key)
            match = re.search(r'_bundle_v(\d+\.\d+\.\d+)\.joblib$', latest_path)
            if match:
                parts = tuple(map(int, match.group(1).split('.')))
                version = f'{parts[0]}.{parts[1]}.{parts[2] + 1}'
            else:
                version = '1.0.0'
        else:
            version = '1.0.0'

    bundle = {
        'model': model,
        'threshold': threshold,
        'config_hash': _hash_config(),
        'metadata': {
            'name': name,
            'version': version,
            'bundle_version': '1.0',
            **(metadata or {}),
        },
    }

    path = os.path.join(MODEL_DIR, f'{name}_bundle_v{version}.joblib')
    joblib.dump(bundle, path)
    logger.info(f'Production bundle saved: {path} | threshold={threshold} | config_hash={bundle["config_hash"][:8]}...')
    return path


def load_production_bundle(name: str, version: str = '1.0.0'):
    """
    Load a production bundle (model + threshold) by name and version.
    """
    path = os.path.join(MODEL_DIR, f'{name}_bundle_v{version}.joblib')

    if not os.path.exists(path):
        message = (
            f"Production bundle not found: {path} | "
            f"Run the training pipeline first."
        )
        logger.error(f"ARTIFACT MISSING: {message}")
        raise ModelArtifactNotFoundError(message)

    bundle = joblib.load(path)
    logger.info(f"Production bundle loaded: {path} | threshold={bundle['threshold']}")
    return bundle['model'], bundle['threshold'], bundle['metadata']


def list_available_models() -> List[Dict[str, Any]]:
    """
    List all versioned model artifacts in the artifact directory.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    pattern = os.path.join(MODEL_DIR, '*.joblib')
    matches = glob.glob(pattern)

    results = []
    for path in matches:
        filename = os.path.basename(path)

        # Match bundle files
        bundle_match = re.match(r'^(.+)_bundle_v(\d+\.\d+\.\d+)\.joblib$', filename)
        if bundle_match:
            results.append({
                "name": bundle_match.group(1),
                "version": bundle_match.group(2),
                "type": "bundle",
                "path": path,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
            })
            continue

        # Match regular model files
        model_match = re.match(r'^(.+)_v(\d+\.\d+\.\d+)\.joblib$', filename)
        if model_match:
            results.append({
                "name": model_match.group(1),
                "version": model_match.group(2),
                "type": "model",
                "path": path,
                "size_kb": round(os.path.getsize(path) / 1024, 1),
            })

    results.sort(key=lambda x: (x['name'], x['version']))
    logger.info(f"Available artifacts: {len(results)} in {MODEL_DIR}")
    return results


if __name__ == "__main__":
    print("\nAvailable models:")
    for m in list_available_models():
        print(f"  {m['name']} v{m['version']} ({m['type']}) - {m['size_kb']} KB")