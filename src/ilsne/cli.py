"""Command-line interface for ilSNe MST utilities."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from . import (
    FlatLCDM,
    exact_convolution_stack,
    gaussian_stack_lambda,
    infer_lambda_for_lens,
    load_sigma_micro_from_csv,
    plot_lambda_posterior,
    sigma_micro_constant,
)
from .io import lens_from_dataframe
from .slsim_yields import SurveyConfig, synthesize_yields


def main(argv: Sequence[str] | None = None) -> None:
    r"""Entry point for the ``ilsne`` command-line tool.

    Parameters
    ----------
    argv : list[str], optional
        Command-line arguments. If ``None`` (default), ``sys.argv[1:]`` is used.

    Notes
    -----
    Provides two subcommands:

    ``fit-lambda``
        Fit the mass-sheet transform scale :math:`\lambda` for a single lens
        given an input CSV file and write the posterior and plot to disk.

    ``stack``
        Multiply independent per-lens posteriors to constrain a shared
        mass-sheet scaling :math:`\lambda` using
        :func:`ilsne.stacking.gaussian_stack_lambda`.
    """
    parser = argparse.ArgumentParser("ilsne")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit-lambda", help="Fit λ for one lens CSV")
    fit.add_argument("--csv", required=True)
    fit.add_argument("--name", default="Lens")
    fit.add_argument("--z-pivot", type=float, default=0.1)
    fit.add_argument("--mp-mean", type=float, required=True)
    fit.add_argument("--mp-sigma", type=float, required=True)
    fit.add_argument("--sigma-micro-csv")
    fit.add_argument("--sigma-micro-const", type=float)
    fit.add_argument("--sigma-int", type=float, default=0.0)
    fit.add_argument(
        "--pantheonplus",
        action="store_true",
        help="Use hierArc Pantheon+ pivot block",
    )
    fit.add_argument(
        "--pantheonplus-data",
        help="Optional path or config for PantheonPlusData (JSON)",
    )
    fit.add_argument(
        "--sigma-micro-scale",
        type=float,
        default=1.0,
        help="Multiply σ_micro by this factor",
    )
    fit.add_argument("--out-json", required=True)
    fit.add_argument("--out-png", required=True)

    sta = sub.add_parser("stack", help="Stack per-lens λ posteriors into λ_tot")
    sta.add_argument(
        "--json", action="append", required=True, help="Per-lens posterior JSON (repeatable)"
    )
    sta.add_argument("--out-json", required=True, help="Path to write λ_tot result JSON")
    sta.add_argument("--out-png", required=True, help="Path to write λ_tot posterior plot")
    sta.add_argument("--method", choices=["gaussian", "exact"], default="gaussian")

    sen = sub.add_parser("microlens-sweep", help="Grid sensitivity of λ to microlensing model")
    sen.add_argument("--csv", required=True)
    sen.add_argument("--name", default="Lens")
    sen.add_argument("--z-pivot", type=float, default=0.1)
    sen.add_argument("--mp-mean", type=float, required=True)
    sen.add_argument("--mp-sigma", type=float, required=True)
    sen.add_argument("--sigma-int", type=float, default=0.0)
    sen.add_argument("--alphas", help="Comma-separated list (e.g., 0.02,0.05,0.1)")
    sen.add_argument("--betas", help="Comma-separated list (e.g., -0.03,0,0.03)")
    sen.add_argument("--reff", type=float, default=1.0)
    sen.add_argument("--scales", help="Comma-separated list of multipliers (e.g., 0.5,1,2)")
    sen.add_argument("--base-sigma", type=float)
    sen.add_argument("--out-csv", required=True)

    yie = sub.add_parser("yields", help="Generate ilSNe/slSNe yields (Parquet)")
    yie.add_argument("--n-sn", type=int, required=True, help="Number of SNe to draw")
    yie.add_argument("--z-pivot", type=float, default=0.1)
    yie.add_argument("--mp-mean", type=float, required=True)
    yie.add_argument("--mp-sigma", type=float, required=True)
    yie.add_argument("--sigma-int", type=float, default=0.1)
    yie.add_argument("--sigma-model", type=float, default=0.05)
    yie.add_argument("--seed", type=int)
    yie.add_argument("--epochs", type=int, default=20)
    yie.add_argument("--days-between", type=float, default=2.5)
    yie.add_argument(
        "--depths",
        required=True,
        help='Comma-separated per-band 5σ depths, e.g. "r:25,i:24.7,z:24.5"',
    )
    yie.add_argument("--name", default="demo", help="Survey name to embed in the output")
    yie.add_argument("--out-parquet", required=True, help="Output Parquet path")

    args = parser.parse_args(argv)
    if args.cmd == "fit-lambda":
        df = pd.read_csv(args.csv)
        lens = lens_from_dataframe(args.name, df)
        cosmo = FlatLCDM()
        sig_fn = None
        if args.sigma_micro_csv:
            sig_fn = load_sigma_micro_from_csv(args.sigma_micro_csv)
        elif args.sigma_micro_const is not None:
            sig_fn = sigma_micro_constant(args.sigma_micro_const)
        if sig_fn is not None and args.sigma_micro_scale != 1.0:
            base_fn = sig_fn

            def scaled_fn(r: float | None) -> float:
                return float(args.sigma_micro_scale) * float(base_fn(r))

            sig_fn = scaled_fn
        mp_block = None
        if args.pantheonplus:
            from .hierarc_glue import PantheonPivotBlock

            # Build block from either a JSON config (for PantheonPlusData kwargs) or a data path.
            if args.pantheonplus_data:
                path = args.pantheonplus_data
                if os.path.isfile(path) and path.lower().endswith(".json"):
                    with open(path) as f:
                        cfg = json.load(f)
                    mp_block = PantheonPivotBlock(z_anchor=args.z_pivot, config=cfg)
                else:
                    mp_block = PantheonPivotBlock(z_anchor=args.z_pivot, data_path=path)
            else:
                mp_block = PantheonPivotBlock(z_anchor=args.z_pivot)
        res = infer_lambda_for_lens(
            lens,
            cosmo,
            args.z_pivot,
            args.mp_mean,
            args.mp_sigma,
            mp_block=mp_block,
            sigma_micro_fn=sig_fn,
            sigma_int=args.sigma_int,
        )
        serializable: dict[str, object] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()
        }
        import os

        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(serializable, f, indent=2)
        fig, _ = plot_lambda_posterior(res, title=f"λ posterior — {args.name}")
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, bbox_inches="tight")
    elif args.cmd == "stack":
        raw_posts = []
        for path in args.json:
            with open(path) as f:
                raw_posts.append(json.load(f))
        if args.method == "gaussian":
            posts = [
                {"lam_mean": float(p["lam_mean"]), "lam_std": float(p["lam_std"])}
                for p in raw_posts
            ]
            out = cast(dict[str, object], gaussian_stack_lambda(posts))
            mu = cast(float, out["lam_mean"])
            sig = cast(float, out["lam_std"])
            lam_grid = np.linspace(mu - 5 * sig, mu + 5 * sig, 2001)
            posterior = np.exp(-0.5 * ((lam_grid - mu) / sig) ** 2)
            posterior /= np.trapezoid(posterior, lam_grid)
            out.update({"lam_grid": lam_grid, "posterior": posterior, "lam_map": mu})
        else:
            posts_arr = [
                {
                    "lam_grid": np.asarray(p["lam_grid"], float),
                    "posterior": np.asarray(p["posterior"], float),
                }
                for p in raw_posts
            ]
            out = cast(dict[str, object], exact_convolution_stack(posts_arr))
        print(f"λ = {out['lam_mean']:.6f} ± {out['lam_std']:.6f}")
        serializable_out: dict[str, object] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in out.items()
        }
        import os

        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(serializable_out, f, indent=2)
        fig, _ = plot_lambda_posterior(out, title="λ_tot posterior")
        os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
        fig.savefig(args.out_png, bbox_inches="tight")
    elif args.cmd == "microlens-sweep":
        df = pd.read_csv(args.csv)
        lens = lens_from_dataframe(args.name, df)
        cosmo = FlatLCDM()
        from .sensitivity import sweep_sigma_micro_params, sweep_sigma_micro_scale

        if args.alphas and args.betas:
            alphas = [float(x) for x in args.alphas.split(",")]
            betas = [float(x) for x in args.betas.split(",")]
            out = sweep_sigma_micro_params(
                lens,
                cosmo,
                args.z_pivot,
                args.mp_mean,
                args.mp_sigma,
                alphas,
                betas,
                args.reff,
                sigma_int=args.sigma_int,
            )
        elif args.scales and (args.base_sigma is not None):
            scales = [float(x) for x in args.scales.split(",")]
            out = sweep_sigma_micro_scale(
                lens,
                cosmo,
                args.z_pivot,
                args.mp_mean,
                args.mp_sigma,
                scales,
                args.base_sigma,
                sigma_int=args.sigma_int,
            )
        else:
            raise SystemExit(
                "Provide either (--alphas & --betas & --reff) or (--scales & --base-sigma)"
            )
        import os

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
    elif args.cmd == "yields":

        def parse_depths(spec: str) -> dict[str, float]:
            out: dict[str, float] = {}
            for tok in spec.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if ":" not in tok:
                    raise SystemExit(f"Bad depths token '{tok}'. Expected format like 'r:25.0'")
                band, val = tok.split(":", 1)
                band = band.strip()
                try:
                    out[band] = float(val)
                except ValueError:
                    raise SystemExit(
                        f"Depth value for band '{band}' must be a float (got '{val}')"
                    ) from None
            if not out:
                raise SystemExit("No valid depths parsed from --depths")
            return out

        cfg = SurveyConfig(
            name=args.name,
            epochs=int(args.epochs),
            days_between=float(args.days_between),
            depths=parse_depths(args.depths),
        )
        cosmo = FlatLCDM()
        df = synthesize_yields(
            n_sn=int(args.n_sn),
            survey=cfg,
            cosmo=cosmo,
            z_pivot=float(args.z_pivot),
            mp_mean=float(args.mp_mean),
            mp_sigma=float(args.mp_sigma),
            sigma_int=float(args.sigma_int),
            sigma_model=float(args.sigma_model),
            seed=(int(args.seed) if args.seed is not None else None),
        )
        import os

        os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
        try:
            df.to_parquet(args.out_parquet, index=False)
        except Exception as e:
            raise SystemExit(
                "Failed to write Parquet. Install a Parquet engine (e.g., 'pyarrow').\n"
                f"Original error: {e}"
            ) from e
        n = len(df)
        frac_sl = float((df["is_sl"].astype(bool)).mean()) if n else 0.0
        print(f"Wrote {n} rows to {args.out_parquet} (sl fraction ~ {frac_sl:.3f})")


if __name__ == "__main__":
    main()
