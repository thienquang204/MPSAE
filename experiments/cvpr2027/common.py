"""Small, CPU-only utilities shared by the paper reproduction tools."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else default


def read_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                         for k, v in row.items()})
    write_text(path, stream.getvalue())


def inside(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path escapes package: {relative}')
    return result


def merge(base, patch):
    from copy import deepcopy
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result
