import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ilsne import (
    FlatLCDM,
    LensDataset,
    LensedSN,
    gaussian_stack_lambda,
    infer_lambda_for_lens,
    load_sigma_micro_from_csv,
    m_sn_from_pivot,
    plot_lambda_posterior,
    sigma_micro_constant,
)
from ilsne.io import lens_from_dataframe


def test_flatlcdm_and_pivot():
    cosmo = FlatLCDM()
    assert cosmo.distance_modulus(0.0) == 0.0
    z_pivot = 0.1
    m_p = 20.0
    assert m_sn_from_pivot(z_pivot, z_pivot, m_p, cosmo) == m_p


def test_infer_lambda_simple():
    cosmo = FlatLCDM()
    z_pivot = 0.1
    m_p_true = 20.0
    sn = LensedSN(z=z_pivot, mu_model=1.0, m_obs=m_p_true, sigma_m=0.1)
    lens = LensDataset(name="L1", sne=[sn])
    res = infer_lambda_for_lens(
        lens,
        cosmo,
        z_pivot=z_pivot,
        mp_mean=m_p_true,
        mp_sigma=0.01,
        lam_grid=(0.9, 1.1, 201),
    )
    assert abs(res["lam_map"] - 1.0) < 5e-3
    assert abs(res["lam_mean"] - 1.0) < 1e-2
    assert res["lam_std"] > 0


def test_microlens_and_io(tmp_path):
    const_fn = sigma_micro_constant(0.1)
    assert const_fn(None) == 0.1
    assert const_fn(1.0) == 0.1

    csv_path = tmp_path / "sig.csv"
    df = pd.DataFrame({"R_arcsec": [0.0, 1.0], "sigma_micro_mag": [0.1, 0.2]})
    df.to_csv(csv_path, index=False)
    fn = load_sigma_micro_from_csv(str(csv_path))
    assert abs(fn(0.0) - 0.1) < 1e-6
    assert abs(fn(0.5) - 0.15) < 1e-6
    assert abs(fn(2.0) - 0.2) < 1e-6

    df2 = pd.DataFrame(
        {
            "z": [0.5],
            "mu_model": [2.0],
            "m_obs": [22.0],
            "sigma_m": [0.1],
            "R_arcsec": [0.7],
        }
    )
    lens = lens_from_dataframe("L", df2)
    assert lens.name == "L" and len(lens.sne) == 1
    assert lens.sne[0].R_arcsec == 0.7


def test_stacking_and_viz():
    posts = [{"lam_mean": 1.0, "lam_std": 0.1}, {"lam_mean": 1.1, "lam_std": 0.2}]
    res = gaussian_stack_lambda(posts)
    mu_true = (1.0 / 0.1**2 + 1.1 / 0.2**2) / (1 / 0.1**2 + 1 / 0.2**2)
    sig_true = np.sqrt(1 / (1 / 0.1**2 + 1 / 0.2**2))
    assert abs(res["lam_mean"] - mu_true) < 1e-12
    assert abs(res["lam_std"] - sig_true) < 1e-12

    plot_res = {"lam_grid": np.array([1.0, 1.1]), "posterior": np.array([0.5, 0.5])}
    fig, ax = plot_lambda_posterior(plot_res)
    assert fig is not None and ax is not None
    plt.close(fig)


def test_cli_stack(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    with open(a, "w") as f:
        json.dump({"lam_mean": 1.0, "lam_std": 0.1}, f)
    with open(b, "w") as f:
        json.dump({"lam_mean": 1.1, "lam_std": 0.2}, f)
    out_json = tmp_path / "out.json"
    out_png = tmp_path / "out.png"
    result = subprocess.run(
        [
            "python",
            "-m",
            "ilsne.cli",
            "stack",
            "--json",
            str(a),
            "--json",
            str(b),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    with open(out_json) as f:
        data = json.load(f)
    mu_true = (1.0 / 0.1**2 + 1.1 / 0.2**2) / (1 / 0.1**2 + 1 / 0.2**2)
    sig_true = np.sqrt(1 / (1 / 0.1**2 + 1 / 0.2**2))
    assert abs(data["lam_mean"] - mu_true) < 1e-12
    assert abs(data["lam_std"] - sig_true) < 1e-12
    assert "λ =" in result.stdout
