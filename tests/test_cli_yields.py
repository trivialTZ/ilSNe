from __future__ import annotations

import os

import pandas as pd

from ilsne.cli import main


def test_cli_yields_parquet_monkeypatched(monkeypatch, tmp_path):
    # Route through test adapter to avoid real slsim
    monkeypatch.setenv("ILSNE_SLSIM_ADAPTER", "ilsne.test_adapters:dummy_slsim_adapter")

    # Monkeypatch to_parquet to avoid pyarrow dependency during tests
    called = {"ok": False}

    def fake_to_parquet(self, path, index=False):  # type: ignore[no-redef]
        # write CSV instead so we can inspect
        nonlocal called
        called["ok"] = True
        # mimic creating the directories
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        csv_path = str(path) + ".csv"
        self.to_csv(csv_path, index=index)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=True)

    out = tmp_path / "yields" / "t.parquet"
    argv = [
        "yields",
        "--n-sn",
        "15",
        "--z-pivot",
        "0.1",
        "--mp-mean",
        "18.966",
        "--mp-sigma",
        "0.008",
        "--sigma-int",
        "0.1",
        "--sigma-model",
        "0.05",
        "--epochs",
        "5",
        "--days-between",
        "2.5",
        "--depths",
        "r:25.0,i:24.7",
        "--out-parquet",
        str(out),
    ]
    main(argv)
    assert called["ok"] is True
    # Ensure CSV written by the monkeypatched function exists and is readable
    csv_path = str(out) + ".csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 15
    assert set(["il", "sl"]).issuperset(set(df["kind"].unique()))
