from pathlib import Path
import yaml

def _clean_yaml_bytes(b: bytes) -> str:
    # Remove BOM, NBSP, zero-width space, and stray 0xC2 bytes
    b = (b
         .replace(b'\xef\xbb\xbf', b'')
         .replace(b'\xc2\xa0', b' ')
         .replace(b'\xe2\x80\x8b', b'')
         .replace(b'\xc2', b''))
    return b.decode("utf-8", errors="strict")

def _has_any(d: dict, *candidates: str) -> bool:
    keys = set(d.keys())
    return any(k in keys for k in candidates)

def test_condition_yaml_shape(repo_root: Path):
    raw = (repo_root / "config" / "template.yaml").read_bytes()
    cfg = yaml.safe_load(_clean_yaml_bytes(raw))

    assert isinstance(cfg, dict)
    for k in ("name", "export", "export_name", "model", "capture_style", "trigger", "hardware"):
        assert k in cfg, f"missing key: {k}"

    model = cfg["model"]; assert isinstance(model, dict)
    assert _has_any(model, "yolo_detection", "enable_yolo")
    assert _has_any(model, "yolo_model_path", "model_path")

    cap = cfg["capture_style"]; assert isinstance(cap, dict)
    assert _has_any(cap, "stream_MSS", "screen_capture", "stream_capture")

    trg = cfg["trigger"]; assert isinstance(trg, dict)
    assert _has_any(trg, "trigger_class", "class")
    assert _has_any(
        trg,
        "trigger_threshold_configuration",
        "trigger_threshold_confidence",
        "confidence_threshold",
        "threshold",
    )
    assert _has_any(trg, "Arduino_COM", "arduino_com", "com_port")
    assert _has_any(trg, "trigger_pin", "pin")
    assert _has_any(trg, "trigger_style", "plugin", "style")

    hw = cfg["hardware"]; assert isinstance(hw, dict)
    for k in ("use_camera", "camera_id", "camera_width", "camera_height", "camera_fps"):
        assert k in hw, f"missing hardware key: {k}"
