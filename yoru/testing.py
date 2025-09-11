# yoru/testing.py (project side)
from pathlib import Path

def run_inference(images_dir: str, weights_path: str, out_dir: str) -> None:
    """
    Minimal test entry: run inference on images_dir using weights_path,
    and write results into out_dir. Raise on error; return None on success.
    """
    # 例）あなたの実装に合わせて呼び出す：
    # from yoru.pipeline import InferenceRunner
    # runner = InferenceRunner(weights=weights_path, output_dir=out_dir)
    # runner.run_folder(images_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # デモ：実体が未実装でも、最低限フォルダを生成して戻る
    return

def run_training(data_dir: str, out_dir: str, epochs: int = 1, device: str = "cpu") -> None:
    """
    Minimal training entry for smoke tests. Should run quickly and write some artifact into out_dir.
    """
    # 例：
    # from yoru.train import train_one_epoch
    # train_one_epoch(data_dir=data_dir, out_dir=out_dir, device=device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 疑似的にチェックポイントファイルを1つ置いておく実装でもOK：
    (Path(out_dir) / "checkpoint.pt").write_bytes(b"dummy")