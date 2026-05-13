from pathlib import Path

from medsight import config


SAMPLE = """
name: smoke
data:
  train_root: train
  val_root: val
model:
  name: resnet34
train:
  epochs: 5
"""


def test_load_config(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(SAMPLE)
    cfg = config.load_config(p)
    assert cfg.name == "smoke"
    assert cfg.data.train_root == "train"
    assert cfg.model.name == "resnet34"
    assert cfg.train.epochs == 5
