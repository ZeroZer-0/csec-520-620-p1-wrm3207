import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "data" / "build_dataset.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


@pytest.fixture
def raw(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    frame = lambda n: pd.DataFrame(rng.normal(size=(n, 2)), columns=["x", "y"])
    files = {"benign.csv": frame(40), "a.csv": frame(30), "b.csv": frame(8)}
    files["b.csv"].loc[0, "x"] = np.inf                      
    for name, df in files.items():
        df.to_csv(tmp_path / name, index=False)
    monkeypatch.setattr(build, "BENIGN_FILE", "benign.csv")
    monkeypatch.setattr(build, "MALICIOUS_FILES", {"a.csv": "FamA", "b.csv": "FamB"})
    monkeypatch.setattr(build, "MAX_ROWS_PER_MALICIOUS_FILE", 10)
    return tmp_path


def test_missing_input_file_is_an_error_not_a_skip(raw):
    (raw / "b.csv").unlink()
    with pytest.raises(FileNotFoundError, match="b.csv"):
        build.assemble(raw)


def test_cap_inf_drop_labels_and_alignment(raw):
    df = build.assemble(raw)
    assert (df["source"] == "a.csv").sum() == 10                      
    assert (df["source"] == "b.csv").sum() == 7                       
    assert (df["source"] == "benign.csv").sum() == 40                 
    assert set(df.loc[df["source"] == "benign.csv", "Label"]) == {0}
    assert set(df.loc[df["family"] == "FamA", "Label"]) == {1}
    assert np.isfinite(df[["x", "y"]].to_numpy()).all()
    assert df["family"].where(df["source"] == "b.csv").dropna().eq("FamB").all()


def test_assembly_is_deterministic(raw):
    pd.testing.assert_frame_equal(build.assemble(raw), build.assemble(raw))


def test_output_has_crlf_endings_no_helper_columns_and_checks_the_pinned_hash(raw, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["build_dataset.py", "--data-dir", str(raw), "--out-dir", str(out)])
    with pytest.raises(SystemExit, match="does not match"):           
        build.main()
    blob = (out / "dataset.csv").read_bytes()
    assert blob.startswith(b"x,y,Label\r\n") and blob.replace(b"\r\n", b"").count(b"\n") == 0
    sidecar = pd.read_csv(out / "dataset_sources.csv")
    assert list(sidecar.columns) == ["source", "family"] and len(sidecar) == len(pd.read_csv(out / "dataset.csv"))


def test_declared_inputs_and_families():
    assert len(build.MALICIOUS_FILES) == 13 and build.BENIGN_FILE not in build.MALICIOUS_FILES
    assert set(build.MALICIOUS_FILES.values()) == {"DDoS", "Brute Force", "Spoofing", "Mirai", "Recon", "Web-based"}


def test_config_and_pinned_checksum_agree():
    from src.integrity import read_pinned
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert cfg["data"]["sha256"] == read_pinned(ROOT / "data" / "dataset.sha256")
