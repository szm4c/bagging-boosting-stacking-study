from __future__ import annotations
import datetime as dt
import yaml
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent  # …/configs
PROJECT_ROOT = CONFIG_DIR.parent.parent.parent  # repo top-level


def save_params(dataset: str, model: str, metric: float, params: dict) -> Path:
    """Write YAML and return the full path."""
    CONFIG_DIR.mkdir(exist_ok=True)

    file = CONFIG_DIR / f"{dataset.lower()}_{model.lower()}.yaml"
    payload = {
        "updated": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "metric": float(metric),
        "params": params,
    }
    file.write_text(yaml.safe_dump(payload, sort_keys=False))

    # Always prints a valid, readable path
    try:
        print(f"Params saved to {file.relative_to(PROJECT_ROOT)}")
    except ValueError:
        print(f"Params saved to {file}")
    return file


def load_params(dataset: str, model: str) -> dict[str, Any]:
    """
    Load the saved YAML for a given dataset & model,
    and return its contents as a dict with keys: updated, metric, params.
    """
    file = CONFIG_DIR / f"{dataset.lower()}_{model.lower()}.yaml"
    if not file.exists():
        raise FileNotFoundError(f"No config file found at {file}")
    payload = yaml.safe_load(file.read_text())
    return payload
