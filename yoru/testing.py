# yoru/testing.py (project side)
from pathlib import Path

def run_inference(images_dir: str, weights_path: str, out_dir: str) -> None:
    """
    Minimal test entry: run inference on images_dir using weights_path,
    and write results into out_dir. Raise on error; return None on success.
    """
    # Example) call according to your implementation:
    # from yoru.pipeline import InferenceRunner
    # runner = InferenceRunner(weights=weights_path, output_dir=out_dir)
    # runner.run_folder(images_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Demo: even if the implementation is not done, at minimum create the folder and return
    return

def run_training(data_dir: str, out_dir: str, epochs: int = 1, device: str = "cpu") -> None:
    """
    Minimal training entry for smoke tests. Should run quickly and write some artifact into out_dir.
    """
    # Example:
    # from yoru.train import train_one_epoch
    # train_one_epoch(data_dir=data_dir, out_dir=out_dir, device=device)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # It is fine to just drop a single placeholder checkpoint file:
    (Path(out_dir) / "checkpoint.pt").write_bytes(b"dummy")