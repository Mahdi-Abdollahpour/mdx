# Mahdi Abdollahpour
# mahdi.abdollahpour@unibo.it
# 2026
"""Helpers to inspect, load and transfer Keras model weights (HDF5, .keras, pickle)."""

import re
import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Any, Iterable, Optional


def print_model_layers(model):
    """Print index, type, name and input/output shapes of each layer of ``model``."""
    print('printing model layers:\n')
    for i, layer in enumerate(model.layers):
        input_shape = layer.input_shape if hasattr(layer, 'input_shape') else "Unknown"
        output_shape = layer.output_shape if hasattr(layer, 'output_shape') else "Unknown"
        layer_name = layer.name
        layer_type = layer.__class__.__name__

        print(f"{i:02d}: {layer_type} | Name: {layer_name} | Input shape: {input_shape} | Output shape: {output_shape}")


def extract_relevant_name(name, start_token="neural_pusch_receiver/cgnnofdm"):
    """Return the part of ``name`` starting at ``start_token`` (or ``name`` if absent)."""
    start_index = name.find(start_token)
    if start_index != -1:
        return name[start_index:]
    return name


def extract_weights_from_h5_group(group, prefix="", shape=True):
    """Recursively map dataset paths in an HDF5 group to their shapes or values."""
    weights = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            weights.update(extract_weights_from_h5_group(item, prefix=f"{prefix}{key}/",shape=shape))
        elif isinstance(item, h5py.Dataset):
            if shape:
                weights[f"{prefix}{key}"] = item.shape
            else:
                weights[f"{prefix}{key}"] = item[()]
    return weights


def model_comp(model, weights_path, start_token="neural_pusch_receiver/cgnnofdm"):
    """Print which trainable weights of ``model`` match the weights in an HDF5 file by name and shape."""
    model_weights = {
        extract_relevant_name(w.name, start_token): w.shape.as_list() for w in model.trainable_weights
    }

    with h5py.File(weights_path, 'r') as f:
        saved_weights = {
            extract_relevant_name(key, start_token): shape for key, shape in extract_weights_from_h5_group(f).items()
        }

    matching = []
    mismatching = []

    for name, shape in model_weights.items():
        if name in saved_weights:
            if tuple(shape) == saved_weights[name]:
                matching.append((name, shape))
            else:
                mismatching.append((name, shape, saved_weights[name]))
        else:
            mismatching.append((name, shape, "Not in saved weights"))

    for name, shape in saved_weights.items():
        if name not in model_weights:
            mismatching.append((name, "Not in model weights", shape))

    print("\n=== Matching Weights ===")
    for name, shape in matching:
        print(f"Name: {name} | Shape: {shape}")

    print("\n=== Mismatching or Missing Weights ===")
    for name, model_shape, saved_shape in mismatching:
        print(f"Name: {name} | Model Shape: {model_shape} | Saved Shape: {saved_shape}")


def print_weights(weights, title):
    """Print a numbered list of weight names and shapes."""
    print(f"\n=== {title} ===")
    for i, (name, shape) in enumerate(weights.items()):
        print(f"{i:02}: Name: {name} | Shape: {shape}")


def compute_lr_multipliers(model, saved_weights_path, start_token="neural_pusch_receiver/cgnnofdm", lr_m=1):
    """Per-trainable-weight learning-rate multipliers.

    Weights whose name and shape match an entry of the saved HDF5 file get
    ``lr_m``; all others get 1.0.
    """
    model_weights = {
        extract_relevant_name(w.name, start_token): w.shape.as_list() for w in model.trainable_weights
    }
    model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ll_rs/")
    model_weights = normalize_weights_names(model_weights, prefix="cgnn/readout_ch_est/")
    with h5py.File(saved_weights_path, 'r') as f:
        saved_weights = {
            extract_relevant_name(key, start_token): shape for key, shape in extract_weights_from_h5_group(f).items()
        }
    saved_weights = normalize_weights_names(saved_weights, prefix="cgnn/readout_ll_rs/")
    saved_weights = normalize_weights_names(saved_weights, prefix="cgnn/readout_ch_est/")

    lr_multipliers = []
    for weight in model.trainable_weights:
        relevant_name = extract_relevant_name(weight.name, start_token)
        if relevant_name in saved_weights:
            if tuple(weight.shape.as_list()) == saved_weights[relevant_name]:
                lr_multipliers.append(lr_m)
            else:
                lr_multipliers.append(1.0)
        else:
            lr_multipliers.append(1.0)

    return lr_multipliers


def _canonicalize(name: str) -> str:
    """Normalize TF/H5 paths (remove ':0', leading '/', collapse slashes)."""
    if name is None:
        return ""
    name = name.replace("\\", "/")
    if name.endswith(":0"):
        name = name[:-2]
    name = re.sub(r"/{2,}", "/", name)
    return name.lstrip("/")


def _drop_prefix(name: str, token: Optional[str]) -> str:
    """Return the part of ``name`` after the first occurrence of ``token`` (without leading '/')."""
    name = _canonicalize(name)
    if not token:
        return name
    token = _canonicalize(token)
    i = name.find(token)
    if i < 0:
        return name
    j = i + len(token)
    if j < len(name) and name[j] == "/":
        j += 1
    return name[j:]


def extract_weights_from_h5_group(
    g: h5py.Group,
    shape: bool = False,
    prefix: str = ""
) -> Dict[str, Any]:
    """Recursively flatten the numeric datasets of an HDF5 group.

    Returns a mapping from full HDF5 path to the dataset's value, or to its
    shape if ``shape`` is True.
    """
    out = {}
    for k, v in g.items():
        path = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Group):
            out.update(extract_weights_from_h5_group(v, shape=shape, prefix=path))
        else:
            if hasattr(v, "dtype") and v.dtype.kind in ("i", "u", "f", "c"):
                out[path] = tuple(v.shape) if shape else v[()]
    return out


def extract_relevant_name(name: str, start_token: Optional[str]) -> str:
    """Strip ``start_token`` from ``name``; return the canonical name if it is empty or not found."""
    return _drop_prefix(name, start_token)


def normalize_weights_names(saved_weights: Dict[str, Any], prefix: str = "cgnn/readout_ll_rs/") -> Dict[str, Any]:
    r"""Renumber ``{prefix}{block}_<n>`` names so indices start at 0 per block, preserving order.

    Returns a new mapping with the updated names.
    """
    grouped_names: Dict[str, list] = {}
    updated: Dict[str, Any] = {}

    pref = re.escape(prefix)
    # e.g. "cgnn/readout_ll_rs/foo_12" -> base="cgnn/readout_ll_rs/foo", num="12"
    pattern = re.compile(fr"({pref}[^/]+)_([\d]+)")

    for name, value in saved_weights.items():
        m = pattern.search(name)
        if m:
            base = m.group(1)
            num = int(m.group(2))
            grouped_names.setdefault(base, []).append((num, name))
        else:
            updated[name] = value

    for base, name_list in grouped_names.items():
        name_list.sort(key=lambda x: x[0])
        seen_map: Dict[int, int] = {}
        next_idx = 0
        for _, original_name in name_list:
            original_num = int(pattern.search(original_name).group(2))
            if original_num not in seen_map:
                seen_map[original_num] = next_idx
                next_idx += 1
            new_num = seen_map[original_num]
            new_name = pattern.sub(fr"\1_{new_num}", original_name)
            updated[new_name] = saved_weights[original_name]

    return updated


import re
from collections import defaultdict
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy as np
import tensorflow as tf


def _load_weights_direct(model: tf.keras.Model, model_path: str) -> None:
    """Load weights using the standard model-loading path for non-HDF5 inputs."""
    def _print_load_status(loaded_count: Optional[int] = None, total_count: Optional[int] = None) -> None:
        if loaded_count is None or total_count is None or loaded_count >= total_count:
            print(f"[load_weights] Loaded all model weights from '{model_path}'.", flush=True)
            return
        if loaded_count == 0:
            print(f"[load_weights] No model weights were loaded from '{model_path}'.", flush=True)
            return
        print(
            f"[load_weights] Loaded model weights partially from '{model_path}' "
            f"({loaded_count}/{total_count} weights updated).",
            flush=True,
        )

    extension = os.path.splitext(model_path)[1]
    if not extension:
        with open(model_path, "rb") as f:
            model.set_weights(pickle.load(f))
        _print_load_status()
        return
    if extension == ".keras":
        model.load_weights(model_path)
        _print_load_status()
        return
    if extension == ".h5":
        before_weights = model.get_weights()
        model.load_weights(model_path, skip_mismatch=True, by_name=True)
        after_weights = model.get_weights()
        loaded_count = sum(
            not np.array_equal(before, after)
            for before, after in zip(before_weights, after_weights)
        )
        _print_load_status(loaded_count, len(after_weights))
        return
    raise ValueError(
        f"[load_weights] Error: extension '{extension}' not supported. "
        "Supported formats are 'pkl'(without extention), 'keras', and 'h5'."
    )


def _strip_tensor_suffix(name: str) -> str:
    """Remove TensorFlow tensor suffix like ':0'."""
    return re.sub(r":\d+$", "", name)


def _split_path(name: str) -> List[str]:
    """Split a path-like weight name into non-empty parts."""
    clean = _strip_tensor_suffix(name).strip("/")
    return [p for p in clean.split("/") if p]


def _common_suffix_len(a_parts: Sequence[str], b_parts: Sequence[str]) -> int:
    """Count equal path segments from the end."""
    n = 0
    for a, b in zip(reversed(a_parts), reversed(b_parts)):
        if a != b:
            break
        n += 1
    return n


def _suggest_prefix_tokens(
    model_full_names: Sequence[str],
    file_full_names: Sequence[str],
    *,
    min_common_parts: int = 3,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Suggest prefix tokens to strip from model/file weight names.

    Pairs of names sharing at least ``min_common_parts`` trailing path
    segments are used to infer the differing prefixes, which are ranked by
    support and suffix depth. Returns up to ``top_k`` dicts with the keys
    ``start_token_model``, ``start_token_file``, ``start_token``, ``support``,
    ``avg_common_suffix_parts`` and one example pair.
    """
    model_parts = [(name, _split_path(name)) for name in model_full_names]
    file_parts = [(name, _split_path(name)) for name in file_full_names]

    candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for model_name, mp in model_parts:
        if not mp:
            continue

        for file_name, fp in file_parts:
            if not fp:
                continue

            common = _common_suffix_len(mp, fp)
            if common < min_common_parts:
                continue

            model_prefix = "/".join(mp[:-common])
            file_prefix = "/".join(fp[:-common])
            shared_suffix = "/".join(mp[-common:])

            if model_prefix:
                model_prefix += "/"
            if file_prefix:
                file_prefix += "/"

            if not model_prefix and not file_prefix:
                continue

            key = (model_prefix, file_prefix)
            item = candidates.get(key)
            if item is None:
                item = {
                    "start_token_model": model_prefix or None,
                    "start_token_file": file_prefix or None,
                    "start_token": model_prefix if (model_prefix and model_prefix == file_prefix) else None,
                    "support": 0,
                    "total_common_suffix_parts": 0,
                    "example_model": model_name,
                    "example_file": file_name,
                    "example_shared_suffix": shared_suffix,
                }
                candidates[key] = item

            item["support"] += 1
            item["total_common_suffix_parts"] += common

            if common > len(_split_path(item["example_shared_suffix"])):
                item["example_model"] = model_name
                item["example_file"] = file_name
                item["example_shared_suffix"] = shared_suffix

    ranked = sorted(
        candidates.values(),
        key=lambda x: (x["support"], x["total_common_suffix_parts"]),
        reverse=True,
    )

    out: List[Dict[str, Any]] = []
    for item in ranked[:top_k]:
        avg_common = item["total_common_suffix_parts"] / max(1, item["support"])
        out.append(
            {
                "start_token_model": item["start_token_model"],
                "start_token_file": item["start_token_file"],
                "start_token": item["start_token"],
                "support": item["support"],
                "avg_common_suffix_parts": round(avg_common, 2),
                "example_model": item["example_model"],
                "example_file": item["example_file"],
                "example_shared_suffix": item["example_shared_suffix"],
            }
        )
    return out


def _print_prefix_suggestions(
    suggestions: Sequence[Dict[str, Any]],
    *,
    max_items: int = 8,
) -> None:
    """Pretty-print prefix suggestions."""
    if not suggestions:
        print(
            "\n[HINT] No transferred weights and no clear prefix-token suggestion was found.",
            flush=True,
        )
        return

    print(
        "\n[HINT] No weights were transferred. Suggested prefix tokens to try:",
        flush=True,
    )
    for i, s in enumerate(suggestions[:max_items], start=1):
        st_model = repr(s["start_token_model"])
        st_file = repr(s["start_token_file"])
        st_both = repr(s["start_token"]) if s["start_token"] else "None"
        print(
            f"  {i:>2}. start_token_model={st_model} | "
            f"start_token_file={st_file} | "
            f"start_token={st_both} | "
            f"support={s['support']} | "
            f"avg-common-suffix-parts={s['avg_common_suffix_parts']}",
            flush=True,
        )
        print(
            f"      shared suffix: {s['example_shared_suffix']}",
            flush=True,
        )
        print(
            f"      model example: {s['example_model']}",
            flush=True,
        )
        print(
            f"      file  example: {s['example_file']}",
            flush=True,
        )


def _transfer_weights_from_h5_single(
    model: tf.keras.Model,
    saved_weights_path: str,
    *,
    start_token_model: Optional[str],
    start_token_file: Optional[str],
    normalize_prefixes: Iterable[str],
    verbose: int = 0,
) -> Dict[str, Any]:
    """Transfer weights from a single HDF5 file into ``model``.

    Names are first matched as-is; unmatched names are then retried with the
    start tokens stripped. If nothing transfers, prefix tokens are suggested.
    """
    start_token_model = start_token_model if start_token_model else None
    start_token_file = start_token_file if start_token_file else None

    with h5py.File(saved_weights_path, "r") as f:
        raw_file = extract_weights_from_h5_group(f, shape=False)

    raw_model_full_names = [w.name for w in model.weights]
    raw_file_full_names = list(raw_file.keys())

    def _build_file_maps(
        raw_items: Dict[str, np.ndarray],
        *,
        start_token: Optional[str],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, str], Dict[str, Tuple[int, ...]]]:
        weights: Dict[str, np.ndarray] = {}
        full_names: Dict[str, str] = {}

        for full_name, value in raw_items.items():
            canon = _canonicalize(extract_relevant_name(full_name, start_token))
            weights[canon] = value
            full_names[canon] = full_name

        for pref in normalize_prefixes:
            weights = normalize_weights_names(weights, prefix=pref)
            full_names = normalize_weights_names(full_names, prefix=pref)

        shapes: Dict[str, Tuple[int, ...]] = {k: tuple(v.shape) for k, v in weights.items()}
        return weights, full_names, shapes

    def _build_model_maps(
        vars_: Iterable[tf.Variable],
        *,
        start_token: Optional[str],
    ) -> Tuple[Dict[str, tf.Variable], Dict[str, str]]:
        model_map: Dict[str, tf.Variable] = {}
        model_full_names: Dict[str, str] = {}

        for w in vars_:
            canon = _canonicalize(extract_relevant_name(w.name, start_token))
            model_map[canon] = w
            model_full_names[canon] = w.name

        for pref in normalize_prefixes:
            model_map = normalize_weights_names(model_map, prefix=pref)
            model_full_names = normalize_weights_names(model_full_names, prefix=pref)

        return model_map, model_full_names

    saved_weights_1, saved_full_names_1, file_shapes_1 = _build_file_maps(
        raw_file,
        start_token=None,
    )
    model_map_1, model_full_names_1 = _build_model_maps(
        model.weights,
        start_token=None,
    )

    transferred: List[Tuple[str, str, Tuple[int, ...]]] = []
    name_mismatch: List[Tuple[str, str, Tuple[int, ...]]] = []
    shape_mismatch: List[Tuple[str, str, Optional[str], Tuple[int, ...], Tuple[int, ...]]] = []
    assign_error: List[Tuple[str, str, Optional[str], Tuple[int, ...], Tuple[int, ...], str]] = []

    unmatched_model_vars: List[tf.Variable] = []
    unmatched_file_raw: Dict[str, np.ndarray] = {}

    total_model = len(model_map_1)

    for name, var in model_map_1.items():
        model_full = model_full_names_1.get(name, var.name)
        mshape = tuple(var.shape.as_list())

        if name not in saved_weights_1:
            unmatched_model_vars.append(var)
            continue

        file_full = saved_full_names_1.get(name)
        fshape = file_shapes_1[name]

        if mshape != fshape:
            shape_mismatch.append((name, model_full, file_full, mshape, fshape))
            continue

        try:
            var.assign(saved_weights_1[name])
            transferred.append((name, model_full, mshape))
        except Exception as e:
            assign_error.append((name, model_full, file_full, mshape, fshape, str(e)))

    for name, full_name in saved_full_names_1.items():
        if name not in model_map_1:
            unmatched_file_raw[full_name] = raw_file[full_name]

    if unmatched_model_vars and unmatched_file_raw and (start_token_model or start_token_file):
        saved_weights_2, saved_full_names_2, file_shapes_2 = _build_file_maps(
            unmatched_file_raw,
            start_token=start_token_file,
        )
        model_map_2, model_full_names_2 = _build_model_maps(
            unmatched_model_vars,
            start_token=start_token_model,
        )

        for name, var in model_map_2.items():
            model_full = model_full_names_2.get(name, var.name)
            mshape = tuple(var.shape.as_list())

            if name not in saved_weights_2:
                name_mismatch.append((name, model_full, mshape))
                continue

            file_full = saved_full_names_2.get(name)
            fshape = file_shapes_2[name]

            if mshape != fshape:
                shape_mismatch.append((name, model_full, file_full, mshape, fshape))
                continue

            try:
                var.assign(saved_weights_2[name])
                transferred.append((name, model_full, mshape))
            except Exception as e:
                assign_error.append((name, model_full, file_full, mshape, fshape, str(e)))

        file_extras = [
            (name, saved_full_names_2.get(name, name), file_shapes_2[name])
            for name in saved_weights_2.keys()
            if name not in model_map_2
        ]
    else:
        if unmatched_model_vars:
            for var in unmatched_model_vars:
                name = _canonicalize(extract_relevant_name(var.name, None))
                for pref in normalize_prefixes:
                    name = normalize_weights_names({name: name}, prefix=pref).popitem()[0]
                name_mismatch.append((name, var.name, tuple(var.shape.as_list())))

        file_extras = []
        if unmatched_file_raw:
            saved_weights_extra, saved_full_names_extra, file_shapes_extra = _build_file_maps(
                unmatched_file_raw,
                start_token=None,
            )
            file_extras = [
                (name, saved_full_names_extra.get(name, name), file_shapes_extra[name])
                for name in saved_weights_extra.keys()
            ]

    prefix_suggestions: List[Dict[str, Any]] = []
    if len(transferred) == 0:
        for min_parts in (4, 3, 2):
            prefix_suggestions = _suggest_prefix_tokens(
                raw_model_full_names,
                raw_file_full_names,
                min_common_parts=min_parts,
                top_k=8,
            )
            if prefix_suggestions:
                break

    n_ok = len(transferred)
    n_name = len(name_mismatch)
    n_shape = len(shape_mismatch)
    n_err = len(assign_error)
    n_file_extra = len(file_extras)

    def pct(n: int, d: int) -> float:
        return (100.0 * n / d) if d > 0 else 0.0

    summary = {
        "total_model_weights": total_model,
        "transferred": n_ok,
        "transferred_pct": pct(n_ok, total_model),
        "name_mismatch": n_name,
        "name_mismatch_pct": pct(n_name, total_model),
        "shape_mismatch": n_shape,
        "shape_mismatch_pct": pct(n_shape, total_model),
        "assignment_error": n_err,
        "assignment_error_pct": pct(n_err, total_model),
        "file_extras": n_file_extra,
    }

    if verbose >= 1:
        print(
            "[transfer] file={!r} | total(model)={} | transferred={} ({:.1f}%) | "
            "name-mismatch={} ({:.1f}%) | shape-mismatch={} ({:.1f}%) | "
            "assign-error={} ({:.1f}%) | extras-in-file={}".format(
                saved_weights_path,
                total_model,
                n_ok,
                summary["transferred_pct"],
                n_name,
                summary["name_mismatch_pct"],
                n_shape,
                summary["shape_mismatch_pct"],
                n_err,
                summary["assignment_error_pct"],
                n_file_extra,
            ),
            flush=True,
        )

        if n_ok == 0:
            _print_prefix_suggestions(prefix_suggestions)

    if verbose >= 2:
        if transferred:
            print("\n[OK] Transferred (canonical, full model name, shape):", flush=True)
            for n, full_n, s in transferred:
                print(f"  - {n}  |  {full_n}  |  {s}", flush=True)

        if name_mismatch:
            print("\n[SKIP] Name mismatch (canonical, full model name, model shape):", flush=True)
            for n, full_n, s in name_mismatch:
                print(f"  - {n}  |  {full_n}  |  {s}", flush=True)

        if shape_mismatch:
            print(
                "\n[SKIP] Shape mismatch "
                "(canonical, full model name, full file name, model shape, file shape):",
                flush=True,
            )
            for n, model_full, file_full, ms, fs in shape_mismatch:
                print(f"  - {n}  |  {model_full}  |  {file_full}  |  {ms}  |  {fs}", flush=True)

        if assign_error:
            print(
                "\n[ERR] Assignment errors "
                "(canonical, full model name, full file name, model shape, file shape, error):",
                flush=True,
            )
            for n, model_full, file_full, ms, fs, e in assign_error:
                print(
                    f"  - {n}  |  {model_full}  |  {file_full}  |  {ms}  |  {fs}  |  {e}",
                    flush=True,
                )

    if verbose >= 3:
        if file_extras:
            print(
                "\n[FILE-ONLY] Present in file, not in model (canonical, full file name, file shape):",
                flush=True,
            )
            for n, full_n, s in file_extras:
                print(f"  - {n}  |  {full_n}  |  {s}", flush=True)

        if name_mismatch:
            print(
                "\n[MODEL-ONLY] Present in model, missing in file (canonical, full model name, model shape):",
                flush=True,
            )
            for n, full_n, s in name_mismatch:
                print(f"  - {n}  |  {full_n}  |  {s}", flush=True)

    return {
        "transferred": transferred,
        "name_mismatch": name_mismatch,
        "shape_mismatch": shape_mismatch,
        "assign_error": assign_error,
        "file_extras": file_extras,
        "prefix_suggestions": prefix_suggestions,
        "summary": summary,
    }


def load_or_transfer_weights(
    model: tf.keras.Model,
    saved_weights_path: Union[str, Sequence[str]],
    start_token: Optional[Union[str, Sequence[Optional[str]]]] = None,
    *,
    start_token_model: Optional[Union[str, Sequence[Optional[str]]]] = None,
    start_token_file: Optional[Union[str, Sequence[Optional[str]]]] = None,
    normalize_prefixes: Iterable[str] = ("cgnn/readout_ll_rs/", "cgnn/readout_ch_est/"),
    verbose: int = 0,
    ) -> Dict[str, Any]:
    """Load weights from one or more weight files into ``model``.

    ``.h5`` files use name/shape-based transfer: a weight is assigned if its
    name matches (first as-is, then with the start tokens stripped, after
    index normalization) and its shape matches exactly. ``.keras`` and
    extension-less pickle files are loaded directly.

    Parameters
    ----------
    model : tf.keras.Model
        Destination model.
    saved_weights_path : str or Sequence[str]
        Weight file(s); later files overwrite earlier assignments.
    start_token : str or Sequence[str], optional
        Prefix to strip from both model and file names; used only if
        ``start_token_model`` and ``start_token_file`` are both omitted.
    start_token_model, start_token_file : str or Sequence[str], optional
        Prefixes to strip from model and file weight names. Sequences must
        have one entry per weight file.
    normalize_prefixes : Iterable[str]
        Name prefixes whose index suffixes are renumbered from 0.
    verbose : int
        0 = silent, 1 = summary, 2 = per-weight, 3 = + unmatched diagnostics.

    Returns
    -------
    Dict[str, Any]
        Lists ``transferred``, ``name_mismatch``, ``shape_mismatch``,
        ``assign_error``, ``file_extras`` and a ``summary`` dict. A single ``.h5``
        file also returns ``prefix_suggestions``; multiple files return
        cumulative lists and a ``per_file`` summary list.
    """
    def _broadcast_to_list(x: Any, n: int, name: str) -> List[Optional[str]]:
        if x is None:
            return [None] * n
        if isinstance(x, (list, tuple)):
            if len(x) != n:
                raise ValueError(
                    f"`{name}` length ({len(x)}) does not match number of paths ({n})."
                )
            return list(x)
        return [x] * n

    def _make_direct_load_result() -> Dict[str, Any]:
        transferred = [
            (
                _canonicalize(extract_relevant_name(w.name, None)),
                w.name,
                tuple(w.shape.as_list()),
            )
            for w in model.weights
        ]
        total_model = len(transferred)
        return {
            "transferred": transferred,
            "name_mismatch": [],
            "shape_mismatch": [],
            "assign_error": [],
            "file_extras": [],
            "summary": {
                "total_model_weights": total_model,
                "transferred": total_model,
                "transferred_pct": 100.0,
                "name_mismatch": 0,
                "name_mismatch_pct": 0.0,
                "shape_mismatch": 0,
                "shape_mismatch_pct": 0.0,
                "assignment_error": 0,
                "assignment_error_pct": 0.0,
                "file_extras": 0,
            },
        }

    if not isinstance(saved_weights_path, (list, tuple)):
        if os.path.splitext(str(saved_weights_path))[1] != ".h5":
            _load_weights_direct(model, str(saved_weights_path))
            return _make_direct_load_result()
        if start_token_model is None and start_token_file is None and start_token not in (None, ""):
            st_model = start_token
            st_file = start_token
        else:
            st_model = start_token_model
            st_file = start_token_file

        return _transfer_weights_from_h5_single(
            model,
            str(saved_weights_path),
            start_token_model=st_model,   # type: ignore[arg-type]
            start_token_file=st_file,     # type: ignore[arg-type]
            normalize_prefixes=normalize_prefixes,
            verbose=verbose,
        )

    paths = list(saved_weights_path)
    if len(paths) == 0:
        raise ValueError("`saved_weights_path` must be a non-empty sequence.")

    n_paths = len(paths)

    if start_token_model is None and start_token_file is None and start_token not in (None, "", []):
        st = _broadcast_to_list(start_token, n_paths, "start_token")
        st_model_list = st
        st_file_list = st
    else:
        st_model_list = _broadcast_to_list(start_token_model, n_paths, "start_token_model")
        st_file_list = _broadcast_to_list(start_token_file, n_paths, "start_token_file")

    st_model_list = [t if t else None for t in st_model_list]
    st_file_list = [t if t else None for t in st_file_list]

    all_transferred: List[Tuple[str, str, Tuple[int, ...]]] = []
    all_name_mismatch: List[Tuple[str, str, Tuple[int, ...]]] = []
    all_shape_mismatch: List[
        Tuple[str, str, Optional[str], Tuple[int, ...], Tuple[int, ...]]
    ] = []
    all_assign_error: List[
        Tuple[str, str, Optional[str], Tuple[int, ...], Tuple[int, ...], str]
    ] = []
    all_file_extras: List[Tuple[str, str, Tuple[int, ...]]] = []
    per_file: List[Dict[str, Any]] = []

    for idx, (path, st_m, st_f) in enumerate(zip(paths, st_model_list, st_file_list)):
        path_str = str(path)
        if os.path.splitext(path_str)[1] != ".h5":
            _load_weights_direct(model, path_str)
            res = _make_direct_load_result()
        else:
            res = _transfer_weights_from_h5_single(
                model,
                path_str,
                start_token_model=st_m,
                start_token_file=st_f,
                normalize_prefixes=normalize_prefixes,
                verbose=verbose,
            )

        all_transferred.extend(res["transferred"])
        all_name_mismatch.extend(res["name_mismatch"])
        all_shape_mismatch.extend(res["shape_mismatch"])
        all_assign_error.extend(res["assign_error"])
        all_file_extras.extend(res["file_extras"])

        s = dict(res["summary"])
        s["path"] = str(path)
        s["index"] = idx
        s["prefix_suggestions"] = res.get("prefix_suggestions", [])
        per_file.append(s)

    total_model = per_file[-1]["total_model_weights"] if per_file else len(model.weights)
    n_ok = len(all_transferred)
    n_name = len(all_name_mismatch)
    n_shape = len(all_shape_mismatch)
    n_err = len(all_assign_error)
    n_file_extra = len(all_file_extras)

    def pct_total(n: int, d_model: int, n_files: int) -> float:
        denom = d_model * max(1, n_files)
        return (100.0 * n / denom) if denom > 0 else 0.0

    summary = {
        "total_model_weights": total_model,
        "num_files": n_paths,
        "transferred": n_ok,
        "transferred_pct": pct_total(n_ok, total_model, n_paths),
        "name_mismatch": n_name,
        "name_mismatch_pct": pct_total(n_name, total_model, n_paths),
        "shape_mismatch": n_shape,
        "shape_mismatch_pct": pct_total(n_shape, total_model, n_paths),
        "assignment_error": n_err,
        "assignment_error_pct": pct_total(n_err, total_model, n_paths),
        "file_extras": n_file_extra,
    }

    return {
        "transferred": all_transferred,
        "name_mismatch": all_name_mismatch,
        "shape_mismatch": all_shape_mismatch,
        "assign_error": all_assign_error,
        "file_extras": all_file_extras,
        "summary": summary,
        "per_file": per_file,
    }


def transfer_weights_from_h5(
    model: tf.keras.Model,
    saved_weights_path: Union[str, Sequence[str]],
    start_token: Optional[Union[str, Sequence[Optional[str]]]] = None,
    *,
    start_token_model: Optional[Union[str, Sequence[Optional[str]]]] = None,
    start_token_file: Optional[Union[str, Sequence[Optional[str]]]] = None,
    normalize_prefixes: Iterable[str] = ("cgnn/readout_ll_rs/", "cgnn/readout_ch_est/"),
    verbose: int = 0,
) -> Dict[str, Any]:
    """Backward-compatible alias for ``load_or_transfer_weights``."""
    return load_or_transfer_weights(
        model,
        saved_weights_path,
        start_token=start_token,
        start_token_model=start_token_model,
        start_token_file=start_token_file,
        normalize_prefixes=normalize_prefixes,
        verbose=verbose,
    )
