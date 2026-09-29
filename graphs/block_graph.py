# Mahdi Abdollahpour
# 2026

"""Config-driven block graph (`ModularGraph`) for assembling neural receivers."""

import core.runtime as _runtime
from core.runtime import DEBUGD

import tensorflow as tf
from tensorflow.keras import Model
from collections.abc import Mapping
from datetime import datetime
import os
import shutil
import tempfile

from utils import _stop_gradients, dbg, tattle

import blocks as _blocks  # noqa: F401
from blocks.registry import get_block_needs, get_block_registry


LAYER_REGISTRY = get_block_registry(include_aliases=True)
LAYER_REGISTRY["Concat"] = "CONCAT"

# Init-time dependencies declared via `@register_block(needs=(...))`.
BLOCK_NEEDS = get_block_needs(include_aliases=True)


# Optional: the graphviz package (plus Graphviz binaries) renders the block diagram.
try:
    import graphviz
    _HAS_GRAPHVIZ = True
except Exception:
    _HAS_GRAPHVIZ = False


def _short(s, n=40):
    s = str(s)
    return s if len(s) <= n else (s[: max(0, n-1)] + "…")


def _now_tag():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _is_str(x): return isinstance(x, str)


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_ref(ref, store, extras, cur_len):
    """
    Return a Tensor from: int index (supports negatives), '@name', or '$key'.
    """
    if isinstance(ref, int):
        idx = ref if ref >= 0 else (cur_len + ref)
        if idx not in store:
            raise IndexError(f"feeds main ref {ref} resolved to {idx}, not in store")        
        return store[idx]
    if isinstance(ref, str):
        if ref.startswith('$'):
            key = ref[1:]
            if key not in extras:
                raise KeyError(f"extras['{key}'] missing for feed {ref!r}")
            return extras[ref[1:]]
        nm = ref[1:] if ref.startswith('@') else ref
        if nm not in store:
            raise KeyError(f"Reference {ref!r} not found in store. Known: {list(store.keys())}")        
        return store[nm]
        
    raise ValueError(f"Unsupported ref type: {type(ref)} / {ref}")


def _parse_feeds(cfg, store, extras, targets, training, cur_len):
    """
    Resolve a block's feeds into its main input and keyword inputs.

    `feeds[0]` is the main input: a single ref, or a list/tuple of refs that is
    passed to the block as a tuple. Each entry of `feeds[1:]` becomes a keyword
    argument: `'$key'` passes `extras[key]`, and `'$t:key'` passes
    `targets[key]` (gradient-stopped) during training and `None` otherwise.
    Without feeds, the previous output is the main input.
    """
    feeds = cfg.get("feeds", None)
    if (feeds is None or
        (isinstance(feeds, (list, tuple)) and len(feeds) == 0)):
        main_spec = -1
        rest = []
    else:
        if isinstance(feeds, (list, tuple)):
            main_spec = feeds[0]
            rest = feeds[1:]
        else:
            main_spec = feeds
            rest = []            
    if isinstance(main_spec, (list, tuple)):
        main_inputs = tuple(_resolve_ref(r, store, extras, cur_len) for r in main_spec)
        x_main = main_inputs
    else:
        x_main = _resolve_ref(main_spec, store, extras, cur_len)

    feed_kwargs = {}
    for item in rest:
        if not (_is_str(item) and item.startswith('$')):
            raise ValueError(f"Extra feeds after the first must be '$key' or '$t:key'. Got: {item!r}")
        if item.startswith("$t:"):
            tkey = item[3:]
            if training:
                if tkey not in targets:
                    raise KeyError(f"targets['{tkey}'] not provided for feed {item!r}. "
                                   f"Available: {list(targets.keys())}")
                feed_kwargs[tkey] = _stop_gradients(targets[tkey])
            else:
                # blocks receiving '$t:' feeds must accept None outside training
                feed_kwargs[tkey] = None
        else:
            key = item[1:]
            if key not in extras:
                raise KeyError(f"extras['{key}'] not provided.")
            feed_kwargs[key] = extras[key]

    return x_main, feed_kwargs


def _truthy_save(cfg):
    sv = cfg.get("save", False)
    if not sv:
        return False
    if isinstance(sv, (str, list, tuple)):
        return True
    return bool(sv)


def _save_aliases(cfg):
    sv = cfg.get("save", False)
    if not sv:
        return []
    if isinstance(sv, str):
        return [sv]
    if isinstance(sv, (list, tuple)):
        return list(sv)
    return []


class ModularGraph(Model):
    r"""
    Config-driven block graph.

    Builds a network (typically a neural receiver) from a list of block
    configs and runs the blocks strictly in order. Every output is recorded
    in a unified store: the model input is `store[0]`, block `k` (0-based)
    writes `store[k+1]` and, if named, `store[name]`. A block may only
    reference outputs produced earlier.

    Parameters
    ----------
    name : str
        Model name.
    config : list[dict]
        Block configurations (schema below). Must be static across traces.
    sys : Any
        System parameters, injected into blocks registered with
        `@register_block(needs=('sys',))`.
    shapes : optional
        Static shape container; it and its `"M"`/`"K"` entries are
        injectable via `needs`.
    rg_params : Any
        Resource-grid parameters, injectable via `needs=('rg_params',)`.
    graph_label : str, optional
        File stem of the rendered diagram; defaults to the model name.
    dtype : tf.DType, default: tf.complex64
        Keras model dtype; losses are accumulated in its real dtype.
    render_graph : bool, default: True
        Render the block diagram to `img/<graph_label>.png` at construction.
        A terminal summary is printed in any case.
    training : bool, default: True
        Build-time train/eval flag, injectable via `needs=('training',)`.
        Distinct from the call-time `training` argument, which every block
        receives.
    **kwargs :
        Forwarded to `tf.keras.Model.__init__`.

    Call
    ----
    call(x, targets=None, extras=None, training=False)
        x : Tensor
            Model input, recorded as `store[0]`.
        targets : dict[str, Tensor], optional
            Ground truth keyed by semantic name, e.g.
            `{"channel_ofdm": h_ft, "bits": bits}`. Blocks select entries via
            their `target` config key.
        extras : dict[str, Any]
            Runtime inputs that blocks request with `"$key"` feeds. Must
            contain `"shapes"` (with `B` and `T`), which sizes the per-sample
            loss vectors. The DeepEcho receiver provides `"shapes"`,
            `"pilot_mask"`, `"rx_signal"` (`{"y"}`), `"noise"`
            (`{"no", "err_var", "s"}`) and `"mcs_masks"`
            (`{"mcs_ue_mask", "mcs_arr_eval"}`).
        training : bool, default: False
            If True, loss outputs are collected in `loss_dict`.

    Returns `(y, loss_dict, stash, stash_view)`:
        y : output of the last non-loss block.
        loss_dict : loss outputs keyed by block name, plus `"Total"`, the sum
            of all losses with shape `[B*T]`.
        stash : dict of saved outputs, keyed by index, name and aliases.
        stash_view : read-only mapping over the same saved tensors.

    Config schema (per block)
    -------------------------
    type : str (required)
        Registered block type, or the built-in `"Concat"`.
    name : str
        Unique name; the output is then addressable as `"name"` / `"@name"`.
    feeds : list
        `feeds[0]` is the main input: a ref, or a list/tuple of refs passed to
        the block as a tuple (required for `"Concat"`). A ref is an integer
        store index (negative values count back, `-1` = previous output) or a
        block name (`"name"` / `"@name"`). Each entry of `feeds[1:]` is a
        keyword feed: `"$key"` passes `extras[key]` as kwarg `key`, and
        `"$t:key"` passes `targets[key]` (gradient-stopped) during training
        and `None` otherwise. Defaults to `[-1]`.
    axis : int
        Concatenation axis for `"Concat"` (default -1).
    save : bool | str | list[str]
        Expose the output in the stash under its index, its name and the
        given aliases; aliases are also added to the store.
    target : str | list[str]
        Key(s) into `targets`. The selected tensor (a tuple for a list) is
        passed to the block as second positional argument.

    All other keys are forwarded to the block constructor. Blocks whose type
    ends in `"Loss"` must set `target`; each loss output (a `[B*T]` tensor, a
    dict with a `"Total"` entry, or a list of tensors) is added to
    `loss_dict["Total"]`, and loss blocks never become the graph output `y`.

    Example
    -------
    >>> config = [
    ...     {"type": "LMMSE", "feeds": [-1, "$rx_signal", "$noise", "$mcs_masks", "$shapes"],
    ...      "name": "lmmse0"},
    ...     {"type": "LS", "feeds": [[0, "@lmmse0"], "$rx_signal", "$pilot_mask", "$shapes"],
    ...      "name": "ls"},
    ...     {"type": "Concat", "feeds": [[0, "@ls"]], "name": "cat"},
    ...     {"type": "ResNet", "num_blocks": 5, "block_type": "Conv",
    ...      "block_config": {"filters": 8, "groups": -1}, "name": "resnet"},
    ...     {"type": "Conv", "filters": 2, "groups": -1, "norm": False, "act": False,
    ...      "name": "head", "save": "channel"},
    ...     {"type": "ChLoss", "feeds": ["@head", "$noise", "$shapes"],
    ...      "target": "channel_ofdm", "name": "ch_loss"},
    ... ]
    >>> model = ModularGraph(config=config, sys=sys_params, shapes=shapes)
    >>> y, losses, stash, stash_view = model(h_init, targets=targets,
    ...                                      extras=extras, training=True)
    >>> h_hat = stash["channel"]
    """

    def __init__(self, name="ModularGraph", config=None, sys=None, shapes=None,
                            rg_params=None, graph_label=None,
                            dtype=tf.complex64, render_graph=True,
                            training=True, **kwargs):

        self._dtype = dtype
        self._real_dtype = tf.as_dtype(dtype).real_dtype
        super().__init__(name=name, dtype=dtype, **kwargs)

        self.config = config or []
        self.blocks = []
        self.graph_label = graph_label or self.name or "ModularGraph"
        self._stash_keys = set()

        self._validate_config(self.config)

        # runtime objects a block can request via `@register_block(needs=...)`
        injectable = {"sys": sys, "shapes": shapes, "rg_params": rg_params,
                      "training": training}
        if shapes is not None:
            injectable["M"] = shapes["M"]
            injectable["K"] = shapes["K"]

        self._block_names = []
        for block in self.config:
            btype = block["type"]
            if btype == "Concat":
                self.blocks.append((block.get("name"), None, block))
                self._block_names.append(block.get("name", None))
            else:
                kwargs_b = {k: v for k, v in block.items() if k not in ("type", "feeds", "save", "target")}

                for dep in BLOCK_NEEDS.get(btype, ()):
                    if dep not in injectable:
                        raise KeyError(
                            f"Block '{btype}' declares needs=('{dep}',) but the graph "
                            f"has no '{dep}' to inject. Available: {sorted(injectable)}")
                    kwargs_b.setdefault(dep, injectable[dep]) # config wins if set

                if btype not in LAYER_REGISTRY:
                    raise KeyError(f"Unknown block type '{btype}'. Registry: {list(LAYER_REGISTRY)}")
                
                layer = LAYER_REGISTRY[btype](**kwargs_b)
                self.blocks.append((block.get("name"), layer, block))
                self._block_names.append(block.get("name", None))

            if _truthy_save(block):
                name = block.get("name")
                if name is not None:
                    self._stash_keys.add(name)
                self._stash_keys.update(_save_aliases(block))

        # the diagram is best-effort: rendering problems must not abort a run
        try:
            self.show_graph(rankdir="LR", theme="light", print_terminal=True,
                            render=render_graph)
        except Exception as e:
            print(f"[graph] diagram skipped ({type(e).__name__}: {e})")

    def show_graph(self, filename=None, fmt="png", rankdir="LR", theme="auto",
                   print_terminal=True, max_label_width=36, add_timestamp=False,
                   render=True):
        """
        Render a block diagram PNG and print a terminal summary of the configured graph.

        Parameters
        ----------
        filename : str | None
            Output path ('.png' is appended if missing). Defaults to
            `projectroot/img/<graph_label>.png`.
        fmt : str
            Graphviz output format (e.g., "png", "svg", "pdf"). Default "png".
        rankdir : {"LR","TB","BT","RL"}
            Layout direction: Left→Right, Top→Bottom, etc.
        theme : {"auto","light","dark"}
            Node/edge color theme. "auto" picks based on $TERM/dark backgrounds heuristics.
        print_terminal : bool
            If True, prints a compact table + linear flow to stdout.
        max_label_width : int
            Soft cap for multi-line node labels in the rendered image.
        add_timestamp : bool
            If True, appends `_<YYYYmmdd-HHMMSS>` to the filename stem.
        render : bool
            If False, only print the terminal summary and return None
            (useful when many concurrent jobs share one config).

        Returns
        -------
        out_path : str or None
            Path to the primary output file (PNG/SVG/PDF). If Graphviz is not
            available, returns the DOT file path (same stem, ".dot"). None when
            `render` is False.
        """
        spec = self._collect_graph_spec()

        if print_terminal:
            self._print_graph_terminal(spec)

        if not render:
            return None

        if filename is None:
            stem = os.path.join(_project_root(), "img", self.graph_label)
        else:
            root, ext = os.path.splitext(filename)
            stem = root if ext.lower() == f".{fmt}" else filename
        if add_timestamp:
            stem = f"{stem}_{_now_tag()}"
        os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

        if _HAS_GRAPHVIZ:
            bg = "white"; font = "black"; edge = "#444444"
            fill_regular = "#E8F3FF"
            fill_concat  = "#FFF4E5"
            fill_loss    = "#FFE8EA"
            fill_input   = "#EFEFEF"
            if theme in ("dark",) or (theme == "auto" and os.environ.get("TERM","").endswith("256color") is False):
                bg, font = "#0e1116", "#f7f7f7"
                edge = "#BBBBBB"
                fill_regular = "#1e2a38"
                fill_concat  = "#3a2b1b"
                fill_loss    = "#3a1f25"
                fill_input   = "#22262c"

            g = graphviz.Digraph(name=self.name or "ModularGraph", format=fmt)
            g.attr(rankdir=rankdir, bgcolor=bg)
            g.attr("node", shape="box", style="rounded,filled", color=edge, fontname="Helvetica", fontsize="10")
            g.attr("edge", color=edge)

            g.node("n0",
                   f"0 • INPUT\nstore[0]",
                   fillcolor=fill_input)

            target_nodes = set()

            for n in spec["nodes"]:
                idx  = n["index"]
                typ  = n["type"]
                name = n["name"]
                label_lines = [f"{idx} • {name or typ}"]
                if name and name != typ:
                    label_lines.append(f"({typ})")
                if n["main_refs"]:
                    label_lines.append("in: " + _short(", ".join(n["main_refs"]), max_label_width))
                if n["extras"]:
                    label_lines.append("extras: " + _short(", ".join("$"+e for e in n["extras"]), max_label_width))
                if n["save_aliases"]:
                    label_lines.append("save: " + _short(", ".join(n["save_aliases"]), max_label_width))
                if n["axis"] is not None:
                    label_lines.append(f"axis: {n['axis']}")
                if n["loss_target"] is not None:
                    _t = n["loss_target"]
                    tlabel = ", ".join(_t) if isinstance(_t, (list, tuple)) else _t
                    label_lines.append(f"target: {tlabel}")

                fill = fill_loss if n["is_loss"] else (fill_concat if typ == "Concat" else fill_regular)
                g.node(f"n{idx}", "\n".join(label_lines), fillcolor=fill)

            for (src, dst, kind, elabel) in spec["edges"]:
                style = "solid" if kind == "main" else "dashed"
                lbl = elabel if elabel is not None else ""
                if kind == "target":
                    tn = f"t_{lbl}"
                    if tn not in target_nodes:
                        g.node(tn, f'target["{lbl}"]', shape="folder", style="filled", fillcolor="#DDDDEE")
                        target_nodes.add(tn)
                    g.edge(tn, f"n{dst}", style="dashed", xlabel="y_true")
                else:
                    g.edge(f"n{src}", f"n{dst}", style=style, xlabel=_short(lbl, 24))

            # Render in a private temp dir next to the target and move the result
            # into place atomically, so concurrent processes sharing a stem do
            # not delete each other's DOT source mid-render.
            tmpdir = tempfile.mkdtemp(dir=os.path.dirname(stem) or ".")
            try:
                tmp_out = g.render(os.path.basename(stem), directory=tmpdir,
                                   cleanup=True, quiet=True)
                out_path = f"{stem}.{fmt}"
                if not tmp_out.endswith(f".{fmt}"):
                    out_path = stem + os.path.splitext(tmp_out)[1]
                os.replace(tmp_out, out_path)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return out_path

        # without graphviz, write a DOT stub only
        dot_path = f"{stem}.dot"
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write("// Graphviz DOT fallback (install graphviz to render)\n")
            f.write(f"digraph {self.name or 'ModularGraph'} {{ rankdir={rankdir}; }}\n")
        return dot_path

    @property
    def stash_keys(self):
        return frozenset(self._stash_keys)

    def _collect_graph_spec(self):
        """
        Scan self.config and build a spec dict with nodes and edges.

        Returns
        -------
        spec : dict with keys:
          nodes: list of dict(index, name, type, main_refs, extras, save_aliases, axis, is_loss, loss_target)
          edges: list of tuples (src_index, dst_index, kind, label)
                 where kind ∈ {"main","extras","target"} and label is a short ref/$key/target
        """
        spec = {"nodes": [], "edges": []}

        name_to_idx = {}

        def resolve_ref(ref, cur_i):
            # valid store indices: 0 (input) .. cur_i (previous block)
            if isinstance(ref, int):
                idx = ref if ref >= 0 else (cur_i + 1 + ref)
                if idx < 0 or idx > cur_i:
                    raise IndexError(f"main ref {ref} resolves to {idx}, valid 0..{cur_i}")
                lbl = f"[{ref}]→{idx}"
                return idx, lbl
            if isinstance(ref, str):
                if ref.startswith("$"):
                    raise ValueError(f"main ref cannot be extras: {ref}")
                nm = ref[1:] if ref.startswith("@") else ref
                if nm not in name_to_idx:
                    raise KeyError(f"Unknown ref '{ref}'. Known: {list(name_to_idx.keys())}")
                idx = name_to_idx[nm]
                return idx, f"@{nm}"
            raise TypeError(f"Unsupported ref {ref!r}")

        spec["nodes"].append({
            "index": 0, "name": "INPUT", "type": "INPUT",
            "main_refs": [], "extras": [], "save_aliases": [],
            "axis": None, "is_loss": False, "loss_target": None
        })

        for i, cfg_tuple in enumerate(self.blocks):
            name, layer, cfg = cfg_tuple
            typ = cfg["type"]
            idx_this = i + 1
            is_loss = typ.endswith("Loss")
            axis = cfg.get("axis", None) if typ == "Concat" else None

            feeds = cfg.get("feeds", None)
            if feeds is None:
                main_spec, rest = -1, []
            elif isinstance(feeds, (list, tuple)):
                main_spec, rest = (feeds[0], list(feeds[1:])) if feeds else (-1, [])
            else:
                main_spec, rest = feeds, []

            main_refs = []
            main_sources = []
            if isinstance(main_spec, (list, tuple)):
                for r in main_spec:
                    sidx, slbl = resolve_ref(r, cur_i=i)
                    main_sources.append(sidx)
                    main_refs.append(slbl)
            else:
                sidx, slbl = resolve_ref(main_spec, cur_i=i)
                main_sources.append(sidx)
                main_refs.append(slbl)

            extras = []
            for item in rest:
                if not (isinstance(item, str) and item.startswith("$") and len(item) > 1):
                    raise ValueError(f"feeds extras must be '$key', got {item!r}")
                extras.append(item[1:])

            save_aliases = []
            sv = cfg.get("save", False)
            if isinstance(sv, str):
                save_aliases = [sv]
            elif isinstance(sv, (list, tuple)):
                save_aliases = list(sv)

            loss_tgt = None
            if is_loss:
                loss_tgt = cfg.get("target") or cfg.get("target_key") or None

            spec["nodes"].append({
                "index": idx_this,
                "name": name,
                "type": typ,
                "main_refs": main_refs,
                "extras": extras,
                "save_aliases": save_aliases,
                "axis": axis,
                "is_loss": is_loss,
                "loss_target": loss_tgt,
            })

            for src in main_sources:
                spec["edges"].append((src, idx_this, "main", None))

            for ek in extras:
                spec["edges"].append((0, idx_this, "extras", f"${ek}"))

            if loss_tgt is not None:
                if isinstance(loss_tgt, (list, tuple)):
                    for tk in loss_tgt:
                        spec["edges"].append((None, idx_this, "target", tk))
                else:
                    spec["edges"].append((None, idx_this, "target", loss_tgt))

            if name:
                name_to_idx[name] = idx_this

        return spec

    def _print_graph_terminal(self, spec):
        """
        Print a compact, readable summary in the terminal.
        """
        flow = " -> ".join(f"[{n['index']}:{n['name'] or n['type']}]" for n in spec["nodes"] if n["index"] != 0)
        print("\n┌─ Graph Flow ───────────────────────────────────────────────────────────")
        print("│  INPUT [0] -> " + (flow if flow else "(empty)"))
        print("└────────────────────────────────────────────────────────────────────────\n")

        cols = ["idx", "name", "type", "mains", "extras", "save", "axis", "target"]
        widths = [4, 18, 12, 16, 12, 12, 4, 12]
        def fmt_row(vals):
            return "│ " + " │ ".join(_short(v, w).ljust(w) for v, w in zip(vals, widths)) + " │"

        header = "│ " + " │ ".join(t.ljust(w) for t, w in zip(cols, widths)) + " │"
        line =  "├" + "┼".join("─"* (w+2) for w in widths) + "┤"

        print("┌" + "─"*(len(header)-2) + "┐")
        print(header)
        print(line)
        for n in spec["nodes"]:
            if n["index"] == 0:
                continue
            vals = [
                str(n["index"]),
                n["name"] or "",
                n["type"],
                ", ".join(n["main_refs"]) or "-",
                ", ".join("$"+e for e in n["extras"]) or "-",
                ", ".join(n["save_aliases"]) or ("(auto index/name)" if (n["name"] or n["save_aliases"]) else "-"),
                str(n["axis"]) if n["axis"] is not None else "-",
                (", ".join(n["loss_target"]) if isinstance(n["loss_target"], (list, tuple))
                 else (n["loss_target"] or "-")),
            ]
            print(fmt_row(vals))
        print("└" + "─"*(len(header)-2) + "┘\n")

    def _validate_config(self, config):
        """
        Validate the config before any block is built.

        Checks:
          • known block types
          • unique, well-formed names
          • feeds[0] holds main input ref(s) (a list/tuple for Concat) that
            point to earlier outputs only
          • feeds[1:] are '$key' or '$t:key' entries
          • 'target' is a plain str or list[str]; required for loss blocks
          • 'save' aliases are well-formed and do not collide with names
        """
        if not isinstance(config, (list, tuple)):
            raise TypeError("config must be a list of block dicts")

        seen_names = set()
        for i, cfg in enumerate(config):
            if not isinstance(cfg, dict):
                raise TypeError(f"Block {i}: must be a dict")
            if "type" not in cfg:
                raise KeyError(f"Block {i}: missing required key 'type'")
            btype = cfg["type"]
            name  = cfg.get("name")

            if btype not in LAYER_REGISTRY:
                raise KeyError(f"Block {i}: unknown type '{btype}'. "
                               f"Known: {list(LAYER_REGISTRY.keys())}")

            if name is not None:
                if not isinstance(name, str):
                    raise TypeError(f"Block {i}: 'name' must be str or None")
                if name in seen_names:
                    raise ValueError(f"Block {i}: duplicate name '{name}'")
                if name.startswith(('@', '$')):
                    raise ValueError(f"Block {i}: name '{name}' cannot start with '@' or '$'")
                seen_names.add(name)

            feeds = cfg.get("feeds", None)
            if feeds is None:
                main_spec, rest = -1, []
            elif isinstance(feeds, (list, tuple)):
                if len(feeds) == 0:
                    main_spec, rest = -1, []
                else:
                    main_spec, rest = feeds[0], list(feeds[1:])
            else:
                main_spec, rest = feeds, []

            if btype == "Concat" and not isinstance(main_spec, (list, tuple)):
                raise ValueError(f"Block {i} ('Concat'): feeds[0] must be list/tuple of refs")

            def _check_main_ref(ref):
                cur_len = i + 1  # store holds indices 0..i; 0 is the model input
                if isinstance(ref, int):
                    idx = ref if ref >= 0 else (cur_len + ref)
                    if idx < 0 or idx > i:
                        raise IndexError(f"Block {i}: main ref {ref} resolves to {idx}, "
                                         f"but valid indices are 0..{i}")
                    return
                if isinstance(ref, str):
                    if ref.startswith('$'):
                        raise ValueError(f"Block {i}: main input cannot be extras '$...': {ref!r}")
                    nm = ref[1:] if ref.startswith('@') else ref
                    if nm not in seen_names:
                        raise KeyError(f"Block {i}: unknown reference '{ref}' "
                                       f"(known names so far: {sorted(seen_names)})")
                    return
                raise TypeError(f"Block {i}: invalid main ref type {type(ref)} ({ref!r})")

            if isinstance(main_spec, (list, tuple)):
                if len(main_spec) == 0:
                    raise ValueError(f"Block {i}: feeds[0] cannot be empty")
                for r in main_spec:
                    _check_main_ref(r)
            else:
                _check_main_ref(main_spec)

            for j, item in enumerate(rest, start=1):
                if not (isinstance(item, str) and item.startswith('$') and len(item) > 1):
                    raise ValueError(f"Block {i}: feeds[{j}] must be '$key' or '$t:key', got {item!r}")

            tkey = cfg.get("target") or cfg.get("target_key")
            if tkey is not None or btype.endswith("Loss"):
                if btype.endswith("Loss") and not tkey:
                    raise KeyError(
                        f"Block {i} ('{name or btype}'): loss blocks must include "
                        f"'target' as str or list[str]\" to select from targets."
                    )
                if tkey:
                    ok = (isinstance(tkey, str) and len(tkey) > 0) or (
                          isinstance(tkey, (list, tuple)) and len(tkey) > 0
                          and all(isinstance(k, str) and len(k) > 0 for k in tkey))
                    if not ok:
                        raise KeyError(
                            f"Block {i} ('{name or btype}'): 'target' must be str or list[str]."
                        )
                    if isinstance(tkey, str) and tkey.startswith(('@', '$')):
                        raise ValueError(
                            f"Block {i} ('{name or btype}'): 'target' must be a plain key into targets, "
                            f"not a reference like '{tkey}'."
                        )
                    if isinstance(tkey, (list, tuple)):
                        for k in tkey:
                            if k.startswith(('@', '$')):
                                raise ValueError(
                                    f"Block {i} ('{name or btype}'): 'target' keys must be plain, got '{k}'."
                                )

            sv = cfg.get("save", False)
            aliases = []
            if isinstance(sv, str):
                aliases = [sv]
            elif isinstance(sv, (list, tuple)):
                aliases = list(sv)
            elif sv is True:
                aliases = []
            for alias in aliases:
                if not isinstance(alias, str):
                    raise TypeError(f"Block {i}: save alias must be str, got {type(alias)}")
                if alias.startswith(('@', '$')):
                    raise ValueError(f"Block {i}: save alias '{alias}' cannot start with '@' or '$'")
                if alias.isdigit():
                    raise ValueError(f"Block {i}: save alias '{alias}' cannot be an integer-like string")
                if alias in seen_names:
                    raise ValueError(f"Block {i}: save alias '{alias}' collides with existing block name")

    def _process_block(self, layer, x_main, feed_kwargs, cfg, store, extras, targets, training, cur_len):
        """Call a block; if it has a `target`, pass the selected targets as second argument."""
        btype = cfg["type"]

        tkey = cfg.get("target") or cfg.get("target_key")
        if tkey is not None:
            if not isinstance(tkey, str):
                if not (isinstance(tkey, (list, tuple)) and tkey):
                    raise KeyError(f"Block '{cfg.get('name', btype)}' requires 'target' str or list[str].")
                missing = [k for k in tkey if k not in targets]
                if missing:
                    raise KeyError(f"Missing targets for '{cfg.get('name', btype)}': {missing}. "
                                   f"Available: {list(targets.keys())}")
                y_true = tuple(targets[k] for k in tkey)
            else:
                if tkey not in targets:
                    raise KeyError(
                        f"targets['{tkey}'] not provided for '{cfg.get('name', btype)}'. "
                        f"Available: {list(targets.keys())}"
                    )
                y_true = targets[tkey]
            return layer(x_main, y_true, training=training, **feed_kwargs)
        return layer(x_main, training=training, **feed_kwargs)

    def _concat_or_forward(self, layer, cfg, store, extras, targets, training, cur_len):
        x_main, feed_kwargs = _parse_feeds(cfg, store, extras, targets, training, cur_len)

        if cfg["type"] == "Concat":
            if not isinstance(x_main, (tuple,list)):
                raise ValueError("Concat expects the first feeds item to be a list/tuple of refs.")
            axis = cfg.get("axis", -1)
            return tf.concat(list(x_main), axis=axis)
        return self._process_block(layer, x_main, feed_kwargs, cfg, store, extras, targets, training, cur_len)

    def call(self, x, targets=None, extras=None, training=False):
        """
        Run the graph.

        x: main input (becomes store[0]); targets: ground truth keyed by
        semantic name (e.g. "channel_ofdm", "bits"); extras: runtime inputs
        requested via '$key' feeds, must include "shapes".
        Returns (y, loss_dict, stash, stash_view); see the class docstring.
        """
        extras = extras or {}
        targets = targets or {}

        # int indices, names and aliases -> outputs
        store = {0: x}  
        s = extras["shapes"]
        N = int(s.B * s.T)
        loss_dict = {}
        loss_dict["Total"] = tf.zeros([N], dtype=self._real_dtype)

        stash = {}
        saved_keys = set()

        last_out = store[0]

        for blk_idx, (name, layer, cfg) in enumerate(self.blocks, start=0):
            cur_len = blk_idx + 1
            out = self._concat_or_forward(layer, cfg, store, extras, targets, training, cur_len)
            out_idx = blk_idx + 1
            store[out_idx] = out
            if name:
                store[name] = out

            def _check_loss(l):
                dbg()
                shape_check = tf.logical_and(tf.equal(tf.rank(l), 1),
                        tf.equal(tf.shape(l)[0], N))
                dbg()
                if DEBUGD["verify"]>0:
                    if not shape_check:
                        print(f"Block '{name}' returned unsuported shape({tf.shape(l)}) of loss value.")
                return l

            if cfg["type"].endswith("Loss") and training:
                loss_dict[name] = out
                if isinstance(out, dict):
                    loss_dict["Total"] = loss_dict["Total"] + _check_loss(out["Total"])
                elif tf.is_tensor(out):
                    loss_dict["Total"] = loss_dict["Total"] + _check_loss(out)
                elif isinstance(out, (list, tuple)):
                    for out_ in out:
                        loss_dict["Total"] = loss_dict["Total"] + _check_loss(out_)
                else:
                    raise ValueError(f"Block '{name}' returned unsuported type of loss value.")

            if _truthy_save(cfg):
                stash[out_idx] = out
                if name:
                    stash[name] = out
                for alias in _save_aliases(cfg):
                    stash[alias] = out

                saved_keys.add(out_idx)
                if name:
                    saved_keys.add(name)
                for alias in _save_aliases(cfg):
                    saved_keys.add(alias)
                    store[alias] = out  # expose alias in the unified store as well

            if not cfg["type"].endswith("Loss"):
                last_out = out

        return last_out, loss_dict, stash, _StashView(store, saved_keys)


class _ROMapping(Mapping):
    """Read-only mapping over a fixed set of keys of a backing dict."""

    def __init__(self, backing, keys):
        self._backing = backing
        self._keys = tuple(keys)

    def __getitem__(self, k):
        if k in self._keys:
            return self._backing[k]
        raise KeyError(k)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def items(self):
        for k in self._keys:
            yield k, self._backing[k]

    def keys(self):
        return iter(self._keys)

    def values(self):
        for k in self._keys:
            yield self._backing[k]


def _StashView(store, saved_keys):
    """Return a read-only view of `store` restricted to `saved_keys`."""
    return _ROMapping(store, saved_keys)
