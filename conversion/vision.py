"""Build the fixed-shape NPU vision encoder from the fp32 vision encoder.

Pipeline (all paths are arguments):

1. Freeze graph input/output shapes to batch=1, 960 patches, 240 projected.
2. Fold the position-embedding interpolation, attention-mask assembly and
   projector shape machinery into constants, and decompose every
   ``com.microsoft`` MultiHeadAttention into standard ops.
3. Split every SigLIP2 MLP into 2 chunks along the hidden width so the NPU
   tiler can place the fc1 -> fc2 buffer.

``--check`` runs the CPU parity check: cosine similarity of the final model
vs the original on a random input must be >= 0.9999.

Runs on Windows (needs ``onnx``, ``onnxruntime``, ``numpy``).
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import pathlib
import shutil
import tempfile
import time

FOLD_BATCH = 1
FOLD_PATCHES = 960
FOLD_PROJECTED = 240
FOLD_HEADS = 16
FOLD_HIDDEN = 1152
FOLD_HEAD_DIM = FOLD_HIDDEN // FOLD_HEADS
FOLD_FEAT = 2048
FIXED_SPATIAL = (24, 40)

TILED_NAME = "/model/embeddings/pos_embed/tiled/output_0"
ATTN_BIAS_NAME = "/model/attn_mask_reformat_full/Expand/output_0"
GRID_INDICES_NAME = "/model/multimodal_projector/scatter/flat_indices/output_0"
FLAT_MASK_NAME = "/model/multimodal_projector/scatter/flat_mask/output_0"
OUT_MASK_NAME = "/model/multimodal_projector/out_mask/flat_output_mask/output_0"
EMBEDDINGS_ADD = "/model/embeddings/Add"
SCATTER_NODE = "/model/multimodal_projector/scatter/filled_canvas"
POST_NORM = "/model/post_layernorm_layernorm/LayerNorm"
STEP1_NODE = "/model/multimodal_projector/unshuffle/step1"
STEP3_NODE = "/model/multimodal_projector/unshuffle/step3"
FINAL_NODE = "image_features"

FORBIDDEN_OPS = {
    "Expand", "Resize", "Compress", "Where", "Range", "Shape", "Gather",
    "Slice", "Pad", "Tile", "ScatterND", "Split", "Concat", "Unsqueeze",
    "Squeeze", "ReduceMax", "ReduceMin", "ConstantOfShape", "NonZero",
    "Size", "Mod", "Equal", "Less", "Greater", "And", "Cast", "Sub", "Div",
}


def static_projector_shapes(h, w, patches=FOLD_PATCHES, hidden=FOLD_HIDDEN):
    if h % 2 or w % 2:
        raise ValueError(f"H and W must be even for the 2x2 unshuffle: {(h, w)}")
    if h * w != patches:
        raise ValueError(f"H*W={h * w} != patches={patches}")
    shapes = {
        "canvas": [1, h, w, hidden],
        "reshape1": [1, h, w // 2, hidden * 2],
        "reshape2": [1, w // 2, h // 2, hidden * 4],
        "final": [patches // 4, FOLD_FEAT],
    }
    counts = set()
    for key in ["canvas", "reshape1", "reshape2"]:
        n = 1
        for dim in shapes[key]:
            n *= dim
        counts.add(n)
    if len(counts) != 1:
        raise ValueError(f"element counts disagree: {shapes}")
    return shapes


def expected_grid_indices(h, w):
    import numpy as np

    idx = np.arange(h * w, dtype=np.int64)
    out = np.empty((h * w, 3), dtype=np.int64)
    out[:, 0] = 0
    out[:, 1] = idx // w
    out[:, 2] = idx % w
    return out


def mha_block_spec(prefix, q, k, v, out, heads=FOLD_HEADS,
                   head_dim=FOLD_HEAD_DIM, scale_init="v6b/mha_scale",
                   qshape_init="v6b/mha_qshape", merge_init="v6b/mha_merge",
                   patches=FOLD_PATCHES):
    hidden = heads * head_dim
    assert hidden == FOLD_HIDDEN, hidden
    t = lambda s: f"{prefix}/{s}"  # noqa: E731
    q4, qt, k4, kt, v4, vt = t("q4d"), t("qt"), t("k4d"), t("kt"), t("v4d"), t("vt")
    scores, scaled, probs, ctx, ct = (
        t("scores"), t("scaled"), t("probs"), t("ctx"), t("ct"))
    _ = patches
    return [
        {"name": t("q_reshape"), "op": "Reshape", "inputs": [q, qshape_init],
         "outputs": [q4], "attrs": {}},
        {"name": t("q_transpose"), "op": "Transpose", "inputs": [q4],
         "outputs": [qt], "attrs": {"perm": [0, 2, 1, 3]}},
        {"name": t("k_reshape"), "op": "Reshape", "inputs": [k, qshape_init],
         "outputs": [k4], "attrs": {}},
        {"name": t("k_transpose"), "op": "Transpose", "inputs": [k4],
         "outputs": [kt], "attrs": {"perm": [0, 2, 3, 1]}},
        {"name": t("v_reshape"), "op": "Reshape", "inputs": [v, qshape_init],
         "outputs": [v4], "attrs": {}},
        {"name": t("v_transpose"), "op": "Transpose", "inputs": [v4],
         "outputs": [vt], "attrs": {"perm": [0, 2, 1, 3]}},
        {"name": t("scores"), "op": "MatMul", "inputs": [qt, kt],
         "outputs": [scores], "attrs": {}},
        {"name": t("scale"), "op": "Mul", "inputs": [scores, scale_init],
         "outputs": [scaled], "attrs": {}},
        {"name": t("softmax"), "op": "Softmax", "inputs": [scaled],
         "outputs": [probs], "attrs": {"axis": -1}},
        {"name": t("context"), "op": "MatMul", "inputs": [probs, vt],
         "outputs": [ctx], "attrs": {}},
        {"name": t("merge_transpose"), "op": "Transpose", "inputs": [ctx],
         "outputs": [ct], "attrs": {"perm": [0, 2, 1, 3]}},
        {"name": t("merge_reshape"), "op": "Reshape", "inputs": [ct, merge_init],
         "outputs": [out], "attrs": {}},
    ]


def eliminate_dead_nodes(nodes, graph_outputs):
    live = set(graph_outputs)
    keep = [False] * len(nodes)
    for i in range(len(nodes) - 1, -1, -1):
        node = nodes[i]
        if any(o in live for o in node["outputs"]):
            keep[i] = True
            live.update(n for n in node["inputs"] if n)
    kept = [n for n, k in zip(nodes, keep) if k]
    removed = [n["name"] for n, k in zip(nodes, keep) if not k]
    return kept, removed


def op_policy_ok(op_counts, mha_count, decomposed):
    bad = []
    for (domain, op), count in sorted(op_counts.items()):
        if domain != "ai.onnx":
            if not decomposed and (domain, op) == (
                "com.microsoft", "MultiHeadAttention"
            ):
                continue
            bad.append(f"unexpected domain {domain}::{op} x{count}")
            continue
        if op in FORBIDDEN_OPS and not (decomposed and op == "Mul"):
            bad.append(f"forbidden op {op} x{count}")
    if decomposed:
        if mha_count:
            bad.append(f"{mha_count} MultiHeadAttention nodes remain")
        for op, want in [("Softmax", 27), ("Mul", 27)]:
            got = op_counts.get(("ai.onnx", op), 0)
            if got != want:
                bad.append(f"{op} count {got} != {want}")
    elif mha_count != 27:
        bad.append(f"MultiHeadAttention count {mha_count} != 27")
    return bad


def metrics(a, b):
    import numpy as np

    dtypes = {"a_dtype": str(np.asarray(a).dtype),
              "b_dtype": str(np.asarray(b).dtype)}
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        return {**dtypes, "same_shape": False, "a_shape": list(a.shape),
                "b_shape": list(b.shape)}
    d = np.abs(a - b)
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    cosine = float(np.clip(np.dot(aa, bb) / denom, -1, 1)) if denom else 0.0
    return {**dtypes, "same_shape": True, "shape": list(a.shape),
            "max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rms": float(np.sqrt(np.mean((a - b) ** 2))), "cosine": cosine,
            "identical": bool(np.array_equal(a, b))}


def _node_recs(graph):
    return [{"name": n.name, "inputs": list(n.input),
             "outputs": list(n.output)} for n in graph.node]


def _by_name(graph, name):
    for node in graph.node:
        if node.name == name:
            return node
    raise ValueError(f"node not found: {name}")


def _producer(graph, tensor):
    found = [n for n in graph.node if tensor in n.output]
    if len(found) != 1:
        raise ValueError(f"{len(found)} producers for {tensor}")
    return found[0]


def _input_shape(model, name):
    for info in model.graph.input:
        if info.name == name:
            return [d.dim_value if d.HasField("dim_value") else d.dim_param
                    for d in info.type.tensor_type.shape.dim]
    raise ValueError(f"graph input not found: {name}")


def _shape_of(info):
    dims = []
    for dim in info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append("?")
    return dims


def freeze_shapes(src, dst):
    """Fix symbolic dims to batch=1, 960 patches, 240 projected."""
    import onnx

    mapping = {"batch_size": FOLD_BATCH, "num_patches": FOLD_PATCHES,
               "num_patches_projected": FOLD_PROJECTED,
               "num_image_tokens": FOLD_PROJECTED}
    model = onnx.load(str(src), load_external_data=False)
    graph = model.graph
    unmapped = set()
    for info in list(graph.input) + list(graph.output):
        for dim in info.type.tensor_type.shape.dim:
            if dim.HasField("dim_param"):
                if dim.dim_param in mapping:
                    value = mapping[dim.dim_param]
                    dim.ClearField("dim_param")
                    dim.dim_value = value
                else:
                    unmapped.add(dim.dim_param)
    after = {i.name: _shape_of(i) for i in graph.input}
    remaining = {n: d for n, d in after.items()
                 if any(isinstance(x, str) for x in d)}
    if remaining:
        raise ValueError(f"fixed-shape goal not met: {remaining}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(dst))
    locations = set()
    for tensor in graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    for location in sorted(locations):
        origin = src.parent / location
        target = dst.parent / location
        if origin.resolve() != target.resolve():
            shutil.copyfile(origin, target)
    print(f"FREEZE shapes={after} unmapped={sorted(unmapped)}", flush=True)


def capture_constants(model, src, spatial_hw, patches, threads):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto

    capture = copy.deepcopy(model)
    want = [(TILED_NAME, TensorProto.FLOAT, [1, patches, FOLD_HIDDEN]),
            (ATTN_BIAS_NAME, TensorProto.FLOAT, [1, FOLD_HEADS, patches, patches]),
            (GRID_INDICES_NAME, TensorProto.INT64, [patches, 3]),
            (FLAT_MASK_NAME, TensorProto.BOOL, [patches]),
            (OUT_MASK_NAME, TensorProto.BOOL, [patches // 4])]
    for name, dtype, dims in want:
        _producer(capture.graph, name)
        capture.graph.output.append(
            onnx.helper.make_tensor_value_info(name, dtype, dims))
    tmp = pathlib.Path(src).parent / "vision_capture_tmp.onnx"
    if tmp.exists():
        raise ValueError(f"temp capture model already exists: {tmp}")
    onnx.save_model(capture, str(tmp))
    del capture
    gc.collect()
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        session = ort.InferenceSession(
            str(tmp), so, providers=["CPUExecutionProvider"])
        feeds = {
            "pixel_values": np.zeros((1, patches, 768), dtype=np.float32),
            "pixel_attention_mask": np.ones((1, patches), dtype=np.int64),
            "spatial_shapes": np.array([list(spatial_hw)], dtype=np.int64),
        }
        started = time.perf_counter()
        values = session.run([name for name, _, _ in want], feeds)
        elapsed = time.perf_counter() - started
        del session
        gc.collect()
    finally:
        os.remove(tmp)
    return dict(zip(
        ["tiled", "attn_bias", "grid_indices", "flat_mask", "out_mask"],
        values)), elapsed


def check_captured(cap, h, w, patches):
    import numpy as np

    assert cap["tiled"].shape == (1, patches, FOLD_HIDDEN), cap["tiled"].shape
    assert np.isfinite(cap["tiled"]).all(), "pos_embed has non-finite values"
    assert cap["attn_bias"].shape == (1, FOLD_HEADS, patches, patches)
    if not bool((cap["attn_bias"] == 0.0).all()):
        raise ValueError("attention bias is not all zeros; cannot drop mask Add")
    expect = expected_grid_indices(h, w)
    if cap["grid_indices"].shape != expect.shape or not bool(
            (cap["grid_indices"] == expect).all()):
        raise ValueError("scatter grid is not the identity row-major grid")
    if not bool(cap["flat_mask"].all()):
        raise ValueError("projector input mask is not all valid")
    if not bool(cap["out_mask"].all()):
        raise ValueError("projector output mask is not all valid")


def onnx_get_int(attr):
    if attr.HasField("i"):
        return int(attr.i)
    return int(attr.ints[0])


def onnx_get_float(attr):
    if attr.HasField("f"):
        return float(attr.f)
    return float(attr.floats[0])


def collect_mha(graph):
    nodes = [n for n in graph.node if n.op_type == "MultiHeadAttention"]
    if len(nodes) != 27:
        raise ValueError(f"MultiHeadAttention count {len(nodes)} != 27")
    scales = []
    for node in nodes:
        if not (node.domain == "com.microsoft" and len(node.input) == 8
                and len(node.output) == 1):
            raise ValueError(
                f"unexpected MHA signature: {node.name} domain={node.domain}")
        if [node.input[3], node.input[4], node.input[6], node.input[7]] != [
            "", "", "", ""]:
            raise ValueError(f"MHA {node.name} has unexpected optional inputs")
        if node.input[5] != ATTN_BIAS_NAME:
            raise ValueError(f"MHA {node.name} mask input is not folded bias")
        attrs = {a.name: a for a in node.attribute}
        if set(attrs) != {"num_heads", "scale"}:
            raise ValueError(f"MHA {node.name} attrs {sorted(attrs)}")
        if onnx_get_int(attrs["num_heads"]) != FOLD_HEADS:
            raise ValueError(f"MHA {node.name} num_heads mismatch")
        scales.append(onnx_get_float(attrs["scale"]))
    if any(s != scales[0] for s in scales):
        raise ValueError(f"MHA scales differ: {sorted(set(scales))}")
    return nodes, scales[0]


def build_decomposed(model, cap, mha_nodes, mha_scale, shapes, patches):
    import numpy as np
    import onnx

    variant = copy.deepcopy(model)
    graph = variant.graph
    init_names = {t.name for t in graph.initializer}

    def add_const(name, array):
        if name in init_names:
            raise ValueError(f"initializer collision: {name}")
        init_names.add(name)
        tensor = onnx.numpy_helper.from_array(
            np.ascontiguousarray(array), name=name)
        graph.initializer.append(tensor)
        return name

    pos_name = add_const("v6b/pos_embed", cap["tiled"].astype(np.float32))
    embed = _by_name(graph, EMBEDDINGS_ADD)
    if embed.input[1] != TILED_NAME:
        raise ValueError(f"{EMBEDDINGS_ADD} input[1] is not the tiled pos_embed")
    embed.input[1] = pos_name

    qshape = add_const("v6b/mha_qshape",
                       np.array([1, patches, FOLD_HEADS, FOLD_HEAD_DIM],
                                dtype=np.int64))
    merge = add_const("v6b/mha_merge",
                      np.array([1, patches, FOLD_HIDDEN], dtype=np.int64))
    scale = add_const("v6b/mha_scale", np.array(mha_scale, dtype=np.float32))

    post = _by_name(graph, POST_NORM)
    scatter = _by_name(graph, SCATTER_NODE)
    if scatter.op_type != "ScatterND" or len(scatter.input) != 3:
        raise ValueError(f"{SCATTER_NODE} is not a 3-input ScatterND")
    canvas_shape = add_const("v6b/canvas_shape",
                             np.array(shapes["canvas"], dtype=np.int64))
    scatter_replacement = onnx.helper.make_node(
        "Reshape", [post.output[0], canvas_shape], [scatter.output[0]],
        name=SCATTER_NODE)

    step1 = _by_name(graph, STEP1_NODE)
    step3 = _by_name(graph, STEP3_NODE)
    if step1.op_type != "Reshape" or step3.op_type != "Reshape":
        raise ValueError("unshuffle step1/step3 are not Reshape nodes")
    r1 = add_const("v6b/unshuffle_shape1",
                   np.array(shapes["reshape1"], dtype=np.int64))
    r2 = add_const("v6b/unshuffle_shape2",
                   np.array(shapes["reshape2"], dtype=np.int64))
    step1.input[1] = r1
    step3.input[1] = r2

    final = _by_name(graph, FINAL_NODE)
    if final.op_type != "Compress":
        raise ValueError(f"{FINAL_NODE} is not Compress")
    flat_shape = add_const("v6b/final_shape",
                           np.array(shapes["final"], dtype=np.int64))
    final_replacement = onnx.helper.make_node(
        "Reshape", [final.input[0], flat_shape], [final.output[0]],
        name=FINAL_NODE)

    replacements = {SCATTER_NODE: scatter_replacement,
                    FINAL_NODE: final_replacement}
    for node in mha_nodes:
        prefix = "v6b/" + node.name.strip("/").replace("/", "_")
        specs = mha_block_spec(prefix, node.input[0], node.input[1],
                               node.input[2], node.output[0],
                               scale_init=scale, qshape_init=qshape,
                               merge_init=merge)
        made = []
        for spec in specs:
            made.append(onnx.helper.make_node(
                spec["op"], spec["inputs"], spec["outputs"],
                name=spec["name"], **dict(spec["attrs"])))
        replacements[node.name] = made

    new_nodes = []
    for node in graph.node:
        if node.name in replacements:
            rep = replacements[node.name]
            new_nodes.extend(rep if isinstance(rep, list) else [rep])
        else:
            new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)

    recs = _node_recs(graph)
    outputs = [o.name for o in graph.output]
    if outputs != [FINAL_NODE]:
        raise ValueError(f"graph outputs {outputs} != [{FINAL_NODE}]")
    kept, _ = eliminate_dead_nodes(recs, outputs)
    keep_names = {n["name"] for n in kept}
    survivors = [n for n in graph.node if n.name in keep_names]
    del graph.node[:]
    graph.node.extend(survivors)

    used = set()
    for node in graph.node:
        used.update(n for n in node.input if n)
    kept_inits = [t for t in graph.initializer if t.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)
    live_tensors = set(used)
    for node in graph.node:
        live_tensors.update(node.output)
    kept_infos = [v for v in graph.value_info if v.name in live_tensors]
    del graph.value_info[:]
    graph.value_info.extend(kept_infos)

    for gone in ["pixel_attention_mask", "spatial_shapes"]:
        still_used = [n.name for n in graph.node if gone in n.input]
        if still_used:
            raise ValueError(f"{gone} still used by {still_used[:5]}")
        remaining = [i for i in graph.input if i.name != gone]
        if len(remaining) == len(graph.input):
            raise ValueError(f"graph input {gone} was already missing")
        del graph.input[:]
        graph.input.extend(remaining)
    if [i.name for i in graph.input] != ["pixel_values"]:
        raise ValueError("graph inputs are not [pixel_values]")

    mha_left = sum(1 for n in graph.node if n.op_type == "MultiHeadAttention")
    if mha_left:
        raise ValueError(f"{mha_left} MultiHeadAttention nodes remain")
    kept_imports = [o for o in variant.opset_import if o.domain != "com.microsoft"]
    del variant.opset_import[:]
    variant.opset_import.extend(kept_imports)
    counts: dict = {}
    for node in graph.node:
        counts[(node.domain or "ai.onnx", node.op_type)] = \
            counts.get((node.domain or "ai.onnx", node.op_type), 0) + 1
    violations = op_policy_ok(counts, mha_left, True)
    if violations:
        raise ValueError("op policy violations: " + "; ".join(violations))
    return variant


def spill_to_external(graph, data_path, min_bytes=1 << 20):
    import onnx

    data_path = pathlib.Path(data_path)
    if data_path.exists():
        raise ValueError(f"data file already exists: {data_path}")
    spilled = []
    with open(data_path, "w+b") as handle:
        for tensor in graph.initializer:
            raw = bytes(tensor.raw_data) if tensor.raw_data else b""
            if not raw or len(raw) < min_bytes:
                continue
            offset = handle.tell()
            handle.write(raw)
            onnx.external_data_helper.set_external_data(
                tensor, location=data_path.name, offset=offset, length=len(raw))
            tensor.ClearField("raw_data")
            spilled.append({"name": tensor.name, "bytes": len(raw)})
    if not spilled:
        data_path.unlink()
    return spilled


def smoke_run(path, patches, threads):
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), so, providers=["CPUExecutionProvider"])
    if [i.name for i in session.get_inputs()] != ["pixel_values"]:
        raise ValueError("smoke session inputs are not [pixel_values]")
    got = session.run(
        None, {"pixel_values": np.zeros((1, patches, 768), dtype=np.float32)}
    )[0]
    arr = np.asarray(got)
    if list(arr.shape) != [patches // 4, FOLD_FEAT]:
        raise ValueError(f"smoke output shape {list(arr.shape)}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError("smoke output has non-finite values")
    del session
    gc.collect()


def split_mlp(src, dst, chunks=2):
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    if dst.exists():
        raise FileExistsError(dst)
    model = onnx.load(str(src))
    graph = model.graph
    init = {t.name: t for t in graph.initializer}
    by_name = {n.name: n for n in graph.node}
    for i in range(27):
        p = f"/model/layers.{i}/mlp"
        mm1, add1, gelu, mm2, add2 = (
            by_name[f"{p}/fc1/MatMul"], by_name[f"{p}/fc1/Add"],
            by_name[f"{p}/fc1/Gelu"], by_name[f"{p}/fc2/MatMul"],
            by_name[f"{p}/fc2/Add"])
        bias_name = add1.input[1] if add1.input[1] in init else add1.input[0]
        w1 = numpy_helper.to_array(init[mm1.input[1]])
        b1 = numpy_helper.to_array(init[bias_name])
        w2 = numpy_helper.to_array(init[mm2.input[1]])
        hidden = w1.shape[1]
        edges = np.linspace(0, hidden, chunks + 1).astype(int)
        new_nodes, outs = [], []
        for c in range(chunks):
            lo, hi = edges[c], edges[c + 1]
            t = f"v6split/{p}/c{c}"
            for name, arr in [(f"{t}/w1", w1[:, lo:hi]),
                              (f"{t}/b1", b1[lo:hi]),
                              (f"{t}/w2", w2[lo:hi])]:
                graph.initializer.append(numpy_helper.from_array(
                    np.ascontiguousarray(arr, dtype=np.float32), name))
            new_nodes += [
                helper.make_node("MatMul", [mm1.input[0], f"{t}/w1"],
                                 [f"{t}/h"], name=f"{t}/fc1/MatMul"),
                helper.make_node("Add", [f"{t}/h", f"{t}/b1"], [f"{t}/hb"],
                                 name=f"{t}/fc1/Add"),
                # The released model was built without copying the
                # original ``approximate="tanh"`` attribute, so these GELUs
                # use the exact (erf) form.
                helper.make_node("Gelu", [f"{t}/hb"], [f"{t}/g"],
                                 name=f"{t}/fc1/Gelu", domain=gelu.domain),
                helper.make_node("MatMul", [f"{t}/g", f"{t}/w2"], [f"{t}/o"],
                                 name=f"{t}/fc2/MatMul"),
            ]
            outs.append(f"{t}/o")
        acc = outs[0]
        for c, o in enumerate(outs[1:], 1):
            name = f"v6split/{p}/sum{c}"
            new_nodes.append(helper.make_node("Add", [acc, o], [name], name=name))
            acc = name
        idx = list(add2.input).index(mm2.output[0])
        add2.input[idx] = acc
        pos = list(graph.node).index(mm1)
        for n in (mm1, add1, gelu, mm2):
            graph.node.remove(n)
        for k, n in enumerate(new_nodes):
            graph.node.insert(pos + k, n)
        for name in {mm1.input[1], bias_name, mm2.input[1]}:
            if not any(name in n.input for n in graph.node):
                graph.initializer.remove(init[name])
    onnx.checker.check_model(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(dst), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=dst.name + "_data")
    print(json.dumps({"output": str(dst), "chunks": chunks,
                      "nodes": len(graph.node)}), flush=True)


def parity_check(original, final, threads, seed=0):
    """Cosine(final, original) on a random input must be >= 0.9999."""
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    pixel_values = rng.standard_normal((1, FOLD_PATCHES, 768)).astype(np.float32)

    def run(path, feeds):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        session = ort.InferenceSession(
            str(path), so, providers=["CPUExecutionProvider"])
        try:
            return np.asarray(session.run(None, feeds)[0])
        finally:
            del session
            gc.collect()

    want = run(original, {
        "pixel_values": pixel_values,
        "pixel_attention_mask": np.ones((1, FOLD_PATCHES), dtype=np.int64),
        "spatial_shapes": np.array([list(FIXED_SPATIAL)], dtype=np.int64),
    })
    got = run(final, {"pixel_values": pixel_values})
    while want.ndim > 2 and want.shape[0] == 1:
        want = want[0]
    while got.ndim > 2 and got.shape[0] == 1:
        got = got[0]
    report = metrics(got, want)
    print("PARITY", json.dumps(report), flush=True)
    if not report.get("same_shape") or report["cosine"] < 0.9999:
        raise ValueError(f"parity check failed: {report}")
    return report


def build(input_path, output_path, threads):
    import onnx

    src = pathlib.Path(input_path)
    dst = pathlib.Path(output_path)
    if dst.exists():
        raise FileExistsError(dst)
    h, w = FIXED_SPATIAL
    shapes = static_projector_shapes(h, w, FOLD_PATCHES)
    with tempfile.TemporaryDirectory(prefix="vision-npu-") as tmp:
        tmpdir = pathlib.Path(tmp)
        fixed = tmpdir / "fixed.onnx"
        freeze_shapes(src, fixed)
        model = onnx.load(str(fixed), load_external_data=False)
        in_shapes = {n: _input_shape(model, n) for n in
                     ["pixel_values", "pixel_attention_mask", "spatial_shapes"]}
        if in_shapes != {"pixel_values": [1, FOLD_PATCHES, 768],
                         "pixel_attention_mask": [1, FOLD_PATCHES],
                         "spatial_shapes": [1, 2]}:
            raise ValueError(f"unexpected fixed-model input shapes: {in_shapes}")
        cap, capture_s = capture_constants(
            model, fixed, (h, w), FOLD_PATCHES, threads)
        check_captured(cap, h, w, FOLD_PATCHES)
        print(f"CAPTURE s={capture_s:.1f}", flush=True)
        mha_nodes, mha_scale = collect_mha(model.graph)
        variant = build_decomposed(
            model, cap, mha_nodes, mha_scale, shapes, FOLD_PATCHES)
        del model
        gc.collect()
        decomp = tmpdir / "mhadecomp.onnx"
        spill_to_external(variant.graph, tmpdir / "mhadecomp.onnx_data")
        onnx.save_model(variant, str(decomp))
        del variant
        gc.collect()
        smoke_run(decomp, FOLD_PATCHES, threads)
        print("SMOKE ok", flush=True)
        split_mlp(decomp, dst, chunks=2)
    return dst


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="fp32 vision encoder onnx")
    parser.add_argument("--output", required=True,
                        help="vision_encoder_npu.onnx to write")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check", action="store_true",
                        help="CPU parity check vs the original (>= 0.9999)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    dst = build(args.input, args.output, args.threads)
    if args.check:
        parity_check(args.input, dst, args.threads, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
