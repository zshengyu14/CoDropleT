#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert.py
==========
Utility to convert a **TensorFlow-2** checkpoint into a **Haiku** parameter
pickle. This version is specifically adapted for checkpoints where variable
names have been transformed from a path-like structure (e.g., `a/b/c`) to a
dot-separated format with a specific prefix (e.g., `model/_params/a.Sb.Sc`).

Run it like::

    python convert.py \
        --tf_ckpt_dir ckpt/ \
        --test_csv dummy.csv \
        --output params.pkl
"""

from __future__ import annotations

import argparse
import functools
import pickle
from pathlib import Path
from typing import Any, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# These imports are assumed to be in your project structure.
# If they are not found, you may need to adjust your PYTHONPATH.
try:
    from config import model_config
    from model_solubility import myModel
    import pipeline_cosep
except ImportError as e:
    print(f"Error: Could not import project modules. Make sure they are in your Python path. Details: {e}")
    exit(1)

################################################################################
# Name-handling helpers
################################################################################

def _clean_tf_name(name: str) -> str:
    """Strips TF-2 metadata suffixes from a raw checkpoint variable name."""
    # This function is now simpler, as we only care about the core variable path.
    # We remove optimizer slots and other metadata attributes.
    if ".OPTIMIZER_SLOT" in name or ".ATTRIBUTES" not in name:
        return "" # Ignore optimizer variables entirely
    
    # Strip the final "/.ATTRIBUTES/VARIABLE_VALUE"
    suffix = "/.ATTRIBUTES/VARIABLE_VALUE"
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def map_haiku_to_tf(hk_name: str) -> str:
    """
    Translates a Haiku parameter path to its expected TF checkpoint name.
    This is the core of the conversion logic.

    Example:
      Haiku: myModel/protein_encoder/single_transition_input/transition1/weights
      TF:    model/_params/myModel.Sprotein_encoder.Ssingle_transition_input.Stransition1/weights
    """
    # Separate the module path from the parameter name (e.g., 'weights', 'bias')
    parts = hk_name.split("/")
    module_path_parts = parts[:-1]
    param_name = parts[-1]

    # Handle the parameter name translation (e.g., Haiku 'w' -> TF 'weights')
    # Based on your output, it seems TF uses 'weights' and 'bias' directly.
    if param_name == "w":
        tf_param_name = "weights"
    elif param_name == "b":
        tf_param_name = "bias"
    else:
        tf_param_name = param_name

    # Transform the module path
    # e.g., ['myModel', 'protein_encoder'] -> 'myModel.Sprotein_encoder'
    if not module_path_parts:
        # This case shouldn't happen with hk.transform, but handle it just in case
        return f"model/_params/{tf_param_name}"

    transformed_path = module_path_parts[0]
    if len(module_path_parts) > 1:
        transformed_path += ".S" + ".S".join(module_path_parts[1:])

    # Combine everything into the final expected TF name
    return f"model/_params/{transformed_path}/{tf_param_name}"


################################################################################
# Dict flatten / unflatten helpers
################################################################################

import logging
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)

def flatten_dict(
    tree: hk.Params,
    sep: str = "/"
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Flattens a nested dict of arrays into a single-level dict.
    Returns:
      flat:   mapping from joined-key to array
      paths:  mapping from joined-key to list-of-original-keys
    Logging at DEBUG level every leaf it flattens.
    """
    flat: Dict[str, np.ndarray] = {}
    paths: Dict[str, List[str]] = {}

    def _recurse(sub: Dict[str, Any], path: List[str]):
        for k, v in sub.items():
            new_path = path + [k]
            if isinstance(v, dict):
                _recurse(v, new_path)
            else:
                joined = sep.join(new_path)
                flat[joined] = v
                paths[joined] = new_path.copy()
                logger.debug(f"Flattened {'->'.join(new_path)} → '{joined}'")

    _recurse(tree, [])
    return flat, paths


def unflatten_dict(
    flat: Dict[str, np.ndarray],
    paths: Dict[str, List[str]]
) -> hk.Params:
    """
    Rebuilds the original nested dict from:
      flat:   the joined-key→array map
      paths:  the joined-key→original-key-list map
    (Does *not* split on sep at all—uses the recorded paths.)
    """
    tree: Dict[str, Any] = {}
    for joined, arr in flat.items():
        path = paths[joined]
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = arr
    return tree  # type: ignore[return-value]


################################################################################
# TensorFlow helpers
################################################################################

def _load_tf_variables(ckpt_dir: Path) -> Dict[str, np.ndarray]:
    """Loads all variables from a TF checkpoint, cleaning their names."""
    reader = tf.train.load_checkpoint(str(ckpt_dir))
    tf_vars: Dict[str, np.ndarray] = {}
    print("Loading and cleaning TF variable names...")
    for raw_name, _ in tf.train.list_variables(str(ckpt_dir)):
        clean_name = _clean_tf_name(raw_name)
        if clean_name:  # Only include non-empty (non-optimizer) names
            tensor = reader.get_tensor(raw_name)
            tf_vars[clean_name] = tensor
            # print(f"  Loaded: '{raw_name}' -> '{clean_name}'") # Uncomment for verbose debugging
    print(f"Loaded {len(tf_vars)} non-optimizer variables from checkpoint.")
    return tf_vars


################################################################################
# Main conversion routine
################################################################################

def convert(
    tf_ckpt_dir: Path,
    test_csv: Path,
    output_path: Path,
    crop_size: int = 128,
) -> None:
    """Convert *tf_ckpt_dir* to a Haiku parameter pickle at *output_path*."""

    # ------------------------------------------------------------------
    # 1️⃣  Build dummy features so Haiku can be initialised
    # ------------------------------------------------------------------
    print("Initializing Haiku model to get parameter structure...")
    cfg = model_config(crop_size=crop_size)
    gen = pipeline_cosep.DataGenerator(
        {"test": str(test_csv)},
        cfg.data,
        crop_lower=0,
        crop_upper=cfg.data.training.crop_size,
    )
    ds = tf.data.Dataset.from_generator(
        functools.partial(gen.generate, data_name="test", shuffle=False),
        output_types=gen.output_types,
        output_shapes=gen.output_shapes,
    ).batch(1)
    example = next(ds.as_numpy_iterator())

    # # Ensure conditional branches run if necessary
    # example["resi_num"] = np.full_like(example["resi_num"], crop_size)
    # example["resi_num_2"] = np.full_like(example["resi_num_2"], crop_size)
    dummy_feat = {k: jnp.array(v) for k, v in example.items()}

    def _forward(feat: Dict[str, jnp.ndarray]) -> Any:
        return myModel(cfg)(feat, is_training=False, compute_loss=False)

    hk_params = hk.transform(_forward).init(jax.random.PRNGKey(0), dummy_feat)
    hk_flat,flatten_path = flatten_dict(hk_params)
    print(f"Haiku model initialized with {len(hk_flat)} parameters.")
    for name, tensor in hk_flat.items():
        print(f"  '{name}': shape {tensor.shape}, dtype {tensor.dtype}")

    # ------------------------------------------------------------------
    # 2️⃣  Load TensorFlow checkpoint
    # ------------------------------------------------------------------
    tf_vars = _load_tf_variables(tf_ckpt_dir)

    # ------------------------------------------------------------------
    # 3️⃣  Match variables
    # ------------------------------------------------------------------
    print("Attempting to match Haiku parameters to TensorFlow variables...")
    matched, missing = [], []
    new_flat: Dict[str, np.ndarray] = {}
    tf_vars_set = set(tf_vars.keys())

    for hk_name, hk_tensor in hk_flat.items():
        # Generate the expected TF name based on the new mapping logic
        tf_name_candidate = map_haiku_to_tf(hk_name)
        tf_tensor = tf_vars.get(tf_name_candidate)

        if tf_tensor is not None and tf_tensor.shape == hk_tensor.shape:
            new_flat[hk_name] = tf_tensor
            matched.append((hk_name, tf_name_candidate))
            tf_vars_set.discard(tf_name_candidate) # Remove from extras
        else:
            # Keep the original Haiku parameter if no match is found
            new_flat[hk_name] = hk_tensor
            missing.append((hk_name, tf_name_candidate))
            if tf_tensor is not None:
                print(
                    f"⚠️  Shape mismatch for '{hk_name}': "
                    f"Haiku shape {hk_tensor.shape} vs. TF shape {tf_tensor.shape}"
                )

    extras = sorted(list(tf_vars_set))

    # ------------------------------------------------------------------
    # 4️⃣  Report
    # ------------------------------------------------------------------
    print("\n" + "="*25 + " Conversion Summary " + "="*25)
    print(f"Matched:   {len(matched)} parameters")
    print(f"Missing:   {len(missing)} (Haiku parameters not found in TF checkpoint)")
    print(f"Extras:    {len(extras)} (TF variables not used by Haiku model)")
    print("="*72 + "\n")

    if matched:
        print("✅ Example Mappings (Haiku → TF):")
        for hk_n, tf_n in matched[:5]:
            print(f"  '{hk_n}'\n    → '{tf_n}'")
        print("-" * 20)

    if missing:
        print("❌ Example Missing (Haiku parameter → Expected TF name):")
        for hk_n, tf_n in missing[:5]:
            print(f"  '{hk_n}'\n    → '{tf_n}'")
        print("-" * 20)

    if extras:
        print("ℹ️ Example Extras (Unused TF variables):")
        for tf_n in extras[:5]:
            print(f"  '{tf_n}'")
        print("-" * 20)

    # ------------------------------------------------------------------
    # 5️⃣  Save
    # ------------------------------------------------------------------
    new_tree = unflatten_dict(new_flat, flatten_path)
    # compare if tree structure matches the original Haiku params
    def compare_trees(a: hk.Params, b: hk.Params) -> bool:
        """Recursively compare two Haiku trees for structure equality."""
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                print("⚠️  Keys mismatch:", a.keys(), "vs", b.keys())
                return False
            return all(compare_trees(a[k], b[k]) for k in a)
        print(f"Comparing types: {type(a)} vs {type(b)}")
        return type(a) == type(b)
    if not compare_trees(hk_params, new_tree):
        print("⚠️  Warning: The structure of the new Haiku tree does not match the original. "
              "This may indicate missing parameters or mismatches in shapes.")
    with output_path.open("wb") as fh:
        pickle.dump(new_tree, fh)
    print(f"\n✅  Saved Haiku params to '{output_path}'")
    if len(missing) > 0:
        print("\n⚠️  Warning: Some parameters were missing. The saved file contains the original, "
              "un-trained Haiku values for those parameters.")


################################################################################
# CLI wrapper
################################################################################

def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(
        description="Convert a TF checkpoint with mangled names to a Haiku pickle.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--tf_ckpt_dir", required=True, type=Path, help="Directory containing the TensorFlow checkpoint files.")
    p.add_argument("--test_csv", required=True, type=Path, help="Path to a CSV file with a single example for model initialization.")
    p.add_argument("--output", required=True, type=Path, help="Destination path for the output Haiku parameters pickle file.")
    p.add_argument("--crop_size", type=int, default=128, help="Sequence crop size used by the model (default: 128).")
    return p.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = _parse_args()
    if not args.tf_ckpt_dir.is_dir():
        print(f"Error: Checkpoint directory not found at '{args.tf_ckpt_dir}'")
        return
    if not args.test_csv.is_file():
        print(f"Error: Test CSV not found at '{args.test_csv}'")
        return
    
    convert(args.tf_ckpt_dir, args.test_csv, args.output, args.crop_size)


if __name__ == "__main__":
    main()
