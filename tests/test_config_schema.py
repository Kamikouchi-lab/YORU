from pathlib import Path
import json
import yaml
import pytest

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "yoru_config.schema.json"

def _load_yaml(path: Path):
    b = path.read_bytes()
    # sanitize common invisible bytes like in template test
    b = (b
         .replace(b'\xef\xbb\xbf', b'')
         .replace(b'\xc2\xa0', b' ')
         .replace(b'\xe2\x80\x8b', b'')
         .replace(b'\xc2', b''))
    return yaml.safe_load(b.decode("utf-8"))

@pytest.mark.parametrize("cfg_path", ["config/template.yaml"])
def test_config_schema(repo_root, cfg_path):
    if jsonschema is None:
        pytest.skip("jsonschema not installed; pip install jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    cfg_abs = (repo_root / cfg_path)
    assert cfg_abs.exists(), f"{cfg_path} not found"
    data = _load_yaml(cfg_abs)
    jsonschema.validate(instance=data, schema=schema)
