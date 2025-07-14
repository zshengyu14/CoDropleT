#!/usr/bin/env python3

from __future__ import annotations
"""Run solubility‑model inference.

This script detects the longest sequence in *test_csv*, updates the model
configuration, optionally restores a checkpoint, runs inference, and writes

* **output.txt**   – CSV ``id,solubility_score`` (one row per pair)
* **profiles.pkl** – dict { id → per‑residue profile (np.ndarray) }

-------------------------------------------------------------------------------
🩹  GPU‑memory woes?
-------------------------------------------------------------------------------
We disable JAX’s default *grab‑all‑memory* allocator and let it use at most
60 % of the visible GPU. If the node still can’t initialise cuBLAS, export
`JAX_PLATFORM_NAME=cpu` to fall back to CPU inference – this model is light
for single‑sequence runs.
"""

# ─────────────────────────────────── env tweaks (before JAX import) ───────────
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")


import argparse
import functools
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

# Heavy imports after env tweaks
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf

import pipeline_cosep
from config import model_config
from model_solubility import myModel

# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class SolubilityModel:
    """Haiku / JAX wrapper around *myModel* with jit‑compiled helpers."""

    def __init__(self, cfg: Any):
        self._cfg = cfg

        def _forward(feat: Dict[str, jnp.ndarray], *, is_training: bool = False):
            module = myModel(self._cfg)
            return module(feat, is_training=is_training, compute_loss=False)

        self._haiku = hk.transform(_forward)
        self._init = jax.jit(self._haiku.init)
        self._apply = jax.jit(self._haiku.apply)
        self.params: hk.Params | None = None

    # --------------------------------------------------------------------- #
    def init_params(self, example_feat: Dict[str, np.ndarray], seed: int = 0):
        """Initialise parameters unless a checkpoint has been restored."""
        if self.params is not None:
            return
        rng = jax.random.PRNGKey(seed)
        self.params = self._init(rng, example_feat)
        logging.info("Model parameters initialised from scratch (seed=%d)", seed)

    # --------------------------------------------------------------------- #
    def restore(self, ckpt_path: Path):
        """Load pickled Haiku parameters saved with `haiku.data_structures`."""
        ckpt_path = ckpt_path.expanduser().resolve()
        with ckpt_path.open("rb") as fh:
            new_params = pickle.load(fh)
        def compare_trees(a: hk.Params, b: hk.Params) -> bool:
            """Recursively compare two Haiku trees for structure equality."""
            if isinstance(a, dict) and isinstance(b, dict):
                if a.keys() != b.keys():
                    print("⚠️  Keys mismatch:", a.keys(), "vs", b.keys())
                    return False
                return all(compare_trees(a[k], b[k]) for k in a)
            return True
        if self.params is not None and not compare_trees(self.params, new_params):
            raise ValueError(
                f"Parameters mismatch: {ckpt_path} has different structure "
                "than the current model. Check if you are using the correct "
                "checkpoint for this model."
            )
        self.params = new_params
        logging.info("Restored parameters from %s", ckpt_path)

    # --------------------------------------------------------------------- #
    def predict(self, feat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.params is None:
            raise RuntimeError("Parameters have not been initialised.")
        rng = jax.random.PRNGKey(0)
        return self._apply(self.params, rng, feat)


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def compute_crop_size(csv_path: Path) -> int:
    """Return max(len_1, len_2) from *csv_path* in a single cheap scan."""
    df = pd.read_csv(csv_path, usecols=["len_1", "len_2"])
    return int(max(df["len_1"].max(), df["len_2"].max()))


def make_dataset(csv_path: Path, cfg) -> tf.data.Dataset:
    gen = pipeline_cosep.DataGenerator(
        {"test": str(csv_path)},
        cfg.data,
        crop_lower=0,
        crop_upper=cfg.data.training.crop_size,
    )
    gen_fn = functools.partial(gen.generate, data_name="test", shuffle=False)
    return tf.data.Dataset.from_generator(
        gen_fn,
        output_types=gen.output_types,
        output_shapes=gen.output_shapes,
    )


# --------------------------------------------------------------------------- #
# CLI helpers
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run solubility‑model inference.")
    p.add_argument("--test_csv", required=True, type=Path, help="Path to test CSV")
    p.add_argument(
        "--model_ckpt",
        type=Path,
        default='params.pkl',
        help="Pickled Haiku params (optional)",
    )
    p.add_argument(
        "--results_dir",
        type=Path,
        default=Path("./results"),
        help="Directory for outputs",
    )
    p.add_argument("--batch_size", type=int, default=2, help="Batch size per step")
    return p


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # -------------------------------------------------------------- crop‑size
    crop_size = compute_crop_size(args.test_csv)
    cfg = model_config(crop_size=crop_size)
    cfg.data.training.batch_size = args.batch_size
    logging.info("Using crop_size=%d", crop_size)

    # -------------------------------------------------------------- dataset
    ds_unbatched = make_dataset(args.test_csv, cfg)
    ds_batched = ds_unbatched.batch(args.batch_size)

    try:
        example_feat = next(ds_batched.as_numpy_iterator())

        # --- Inflate length fields so *init* sees the worst‑case shapes --------
        # Some sub‑modules allocate parameters conditionally on the number of
        # sequence segments (derived from `resi_num` fields).  When the first
        # batch happens to be short, the initialisation omits weights that will
        # later be required for longer proteins →  the "parameter must be
        # created in init" error you hit.  We fix this by bumping all length
        # indicators to the maximum crop_size so *every* conditional branch is
        # exercised during parameter creation.
        if "resi_num" in example_feat:
            example_feat["resi_num"] = np.full_like(
                example_feat["resi_num"], crop_size
            )
        if "resi_num_2" in example_feat:
            example_feat["resi_num_2"] = np.full_like(
                example_feat["resi_num_2"], crop_size
            )
    except StopIteration:
        raise RuntimeError("Dataset is empty – check your input CSV.")

    # -------------------------------------------------------------- model
    model = SolubilityModel(cfg)
    model.init_params(example_feat)
    if args.model_ckpt is not None:
        model.restore(args.model_ckpt)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    out_scores: list[float] = []
    out_ids: list[int] = []
    profile_bank: Dict[int, np.ndarray] = {}

    # -------------------------------------------------------------- inference
    for feat in ds_batched.as_numpy_iterator():
        prediction = model.predict(feat)

        scores = prediction["solubility"]["solubility"].reshape(-1)
        ids = feat["id"].reshape(-1)
        resi_nums1 = feat["resi_num"].reshape(-1)           # first protein
        resi_nums2 = feat["resi_num_2"].reshape(-1)         # second protein
        profiles = prediction["solubility"]["sol_approx_profile"]  # (B, 2*crop, 1)

        out_ids.extend(ids.tolist())
        out_scores.extend(scores.tolist())

        for gid, prof, n1, n2 in zip(ids, profiles, resi_nums1, resi_nums2):
            full_profile = np.concatenate(
                [prof[: n1], prof[crop_size : crop_size + n2]]
            )
            profile_bank[int(gid)] = np.squeeze(full_profile)

    # -------------------------------------------------------------- write
    (args.results_dir / "output.txt").write_text(
        "\n".join(f"{gid},{score}" for gid, score in zip(out_ids, out_scores)) + "\n",
        encoding="utf-8",
    )
    with (args.results_dir / "profiles.pkl").open("wb") as fh:
        pickle.dump(profile_bank, fh)

    logging.info("Wrote %d predictions to %s", len(out_ids), args.results_dir)


if __name__ == "__main__":
    main()
