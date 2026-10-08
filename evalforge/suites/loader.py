"""Load evaluation suites from YAML or JSON into validated models."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from evalforge.schemas import Suite


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported suite format: {p.suffix} (use .yaml or .json)")
    return Suite.model_validate(data)
