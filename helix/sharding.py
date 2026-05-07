"""
Auto-sharding code generator.

Given a JAX function and example inputs, infers a Megatron-style 2-D
tensor-parallel + data-parallel sharding plan and emits ready-to-paste
Python code.

Usage
-----
    import helix
    import jax.numpy as jnp

    plan = helix.generate_sharding(
        my_model, x, wq, wk, wv, wo, w1, w2, norm,
        mesh_shape=(2, 4),            # (data_parallel, model_parallel)
        axis_names=('batch', 'model'),
        arg_names=['x','wq','wk','wv','wo','w1','w2','norm'],
    )
    print(plan.code)   # copy-paste into your training script
    print(plan)        # human-readable summary table

How it works
------------
1. Capture the JAXPR to see which arguments appear as the first (activation)
   vs second (weight) operand of every dot_general op.
2. Classify each argument:
     - batch-varying activation  → shard on 'batch' axis
     - fan-out weight (D → F>D) → column-shard on 'model' axis
     - fan-in  weight (F → D<F) → row-shard on 'model' axis (Megatron col→row)
     - 1-D weight (layer norm)  → replicate
     - anything else            → replicate
3. Generate the mesh definition + one `jax.device_put` call per argument.

The generated plan implements Megatron-LM column-row parallelism:
  - QKV / gate / up projections → column-parallel (shard output dim)
  - Out / down projections      → row-parallel    (shard input  dim)
This keeps the all-reduce to exactly one sync per transformer sub-layer.
"""
from __future__ import annotations
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TensorPlan:
    name: str
    shape: tuple[int, ...]
    role: str           # 'activation' | 'weight_col' | 'weight_row' | 'replicated'
    partition_spec: tuple  # e.g. ('batch', None) or (None, 'model')
    reason: str


@dataclass
class ShardingPlan:
    fn_name: str
    mesh_shape: tuple[int, int]
    axis_names: tuple[str, str]
    tensors: list[TensorPlan]
    code: str           # generated Python — ready to paste

    def __str__(self) -> str:
        lines = [
            f"ShardingPlan · {self.fn_name}",
            f"Mesh: {self.mesh_shape[0]}×{self.mesh_shape[1]}  "
            f"({self.axis_names[0]} × {self.axis_names[1]})",
            "",
            f"  {'Name':<16} {'Shape':<24} {'Role':<14} {'PartitionSpec'}",
            "  " + "─" * 72,
        ]
        for t in self.tensors:
            spec = str(tuple(
                (f"'{s}'" if isinstance(s, str) else "None") for s in t.partition_spec
            ))
            lines.append(
                f"  {t.name:<16} {str(t.shape):<24} {t.role:<14} P{spec}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "fn_name": self.fn_name,
            "mesh_shape": list(self.mesh_shape),
            "axis_names": list(self.axis_names),
            "tensors": [
                {
                    "name": t.name,
                    "shape": list(t.shape),
                    "role": t.role,
                    "partition_spec": list(t.partition_spec),
                    "reason": t.reason,
                }
                for t in self.tensors
            ],
            "code": self.code,
        }


# ---------------------------------------------------------------------------
# Inference logic
# ---------------------------------------------------------------------------

def _get_shape(aval: Any) -> tuple[int, ...]:
    return tuple(aval.shape) if hasattr(aval, "shape") else ()


def _classify_arg(
    arg_idx: int,
    shape: tuple[int, ...],
    batch_size: int,
    is_activation: bool,
    is_weight: bool,
    fan_out: bool,     # True when this weight's output_dim > input_dim
) -> TensorPlan:
    """Map a single argument to its sharding role + PartitionSpec."""
    name = f"arg{arg_idx}"

    if not shape:
        return TensorPlan(name, shape, "replicated", (None,), "scalar")

    # Treat any non-weight tensor whose first dim matches batch_size as an
    # activation — catches inputs fed through einsum rather than dot_general.
    if not is_weight and shape[0] == batch_size:
        spec = ("batch",) + (None,) * (len(shape) - 1)
        return TensorPlan(name, shape, "activation", spec,
                          f"first dim {shape[0]} == batch_size {batch_size}")

    if is_weight and len(shape) == 1:
        return TensorPlan(name, shape, "replicated", (None,),
                          "1-D weight (layer norm / bias) — replicate")

    if is_weight and len(shape) >= 2:
        if fan_out:
            # Column-parallel: shard output (last) dim on model axis
            spec = (None,) * (len(shape) - 1) + ("model",)
            return TensorPlan(name, shape, "weight_col", spec,
                              f"fan-out weight (→{shape[-1]}) — column-parallel")
        else:
            # Row-parallel: shard input (first) dim on model axis
            spec = ("model",) + (None,) * (len(shape) - 1)
            return TensorPlan(name, shape, "weight_row", spec,
                              f"fan-in weight ({shape[0]}→) — row-parallel")

    return TensorPlan(name, shape, "replicated",
                      (None,) * len(shape), "no clear role — replicated")


def _infer_roles(
    fn: Callable,
    args: tuple[Any, ...],
) -> dict[int, dict]:
    """
    Return per-argument role dict:
      {arg_idx: {'is_activation': bool, 'is_weight': bool, 'fan_out': bool}}
    """
    from jax._src.core import Var, Literal

    jaxpr = jax.make_jaxpr(fn)(*args)
    n_args = len(jaxpr.jaxpr.invars)

    # Map Var id → arg index (for the function's formal parameters)
    var_to_arg: dict[int, int] = {
        id(v): i for i, v in enumerate(jaxpr.jaxpr.invars)
    }

    # Initialise
    roles: dict[int, dict] = {
        i: {"is_activation": False, "is_weight": False, "fan_out": False}
        for i in range(n_args)
    }

    for eqn in jaxpr.jaxpr.eqns:
        if eqn.primitive.name not in ("dot_general", "dot"):
            continue

        invars = eqn.invars
        if len(invars) < 2:
            continue

        lhs, rhs = invars[0], invars[1]

        # LHS (first operand) = activation
        if isinstance(lhs, Var) and id(lhs) in var_to_arg:
            idx = var_to_arg[id(lhs)]
            roles[idx]["is_activation"] = True

        # RHS (second operand) = weight
        if isinstance(rhs, Var) and id(rhs) in var_to_arg:
            idx = var_to_arg[id(rhs)]
            roles[idx]["is_weight"] = True
            # fan_out: output_dim > input_dim (column-parallel)
            if eqn.outvars:
                out_shape = _get_shape(eqn.outvars[0].aval)
                in_shape  = _get_shape(rhs.aval)
                if out_shape and in_shape:
                    roles[idx]["fan_out"] = out_shape[-1] >= in_shape[0]

    return roles


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _spec_str(spec: tuple, axis_names: tuple[str, str]) -> str:
    """Render a PartitionSpec tuple as Python source."""
    parts = []
    for s in spec:
        if s is None:
            parts.append("None")
        elif s == "batch":
            parts.append(f"'{axis_names[0]}'")
        elif s == "model":
            parts.append(f"'{axis_names[1]}'")
        else:
            parts.append(repr(s))
    return f"P({', '.join(parts)})"


def _generate_code(
    plan: "ShardingPlan",
    arg_names: list[str] | None,
) -> str:
    dp, mp = plan.mesh_shape
    ba, ma = plan.axis_names
    names = arg_names or [t.name for t in plan.tensors]

    header = textwrap.dedent(f"""\
        # ──────────────────────────────────────────────────────
        # HelixIR Auto-Generated Sharding Plan
        # Function : {plan.fn_name}
        # Mesh     : {dp}×{mp}  ({ba} × {ma})
        # Strategy : Megatron-style column–row tensor parallelism
        # ──────────────────────────────────────────────────────
        import numpy as np
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        # Device mesh
        devices = np.array(jax.devices())
        assert len(devices) >= {dp * mp}, "Need at least {dp * mp} devices"
        mesh = Mesh(devices[:{dp * mp}].reshape({dp}, {mp}), axis_names=('{ba}', '{ma}'))

        # Shard tensors
    """)

    lines = [header]
    pad = max(len(n) for n in names) if names else 4
    for tensor, name in zip(plan.tensors, names):
        spec = _spec_str(tensor.partition_spec, plan.axis_names)
        sharding = f"NamedSharding(mesh, {spec})"
        comment = f"  # {tensor.role} — {tensor.reason}"
        lines.append(f"{name:<{pad}} = jax.device_put({name:<{pad}}, {sharding}){comment}")

    lines.append("\n# Run your function — JAX will handle the collective ops automatically")
    lines.append(f"# out = {plan.fn_name}({', '.join(names[:4])}{',...' if len(names)>4 else ''})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sharding(
    fn: Callable,
    *args: Any,
    mesh_shape: tuple[int, int] = (1, 8),
    axis_names: tuple[str, str] = ("batch", "model"),
    arg_names: list[str] | None = None,
    fn_name: str | None = None,
) -> ShardingPlan:
    """
    Analyse fn(*args) and return a ShardingPlan with generated device-placement code.

    Parameters
    ----------
    fn          : The JAX function to shard.
    *args       : Example inputs (used only for shape/dtype — not executed on GPU).
    mesh_shape  : (data_parallel_size, model_parallel_size).  Product must equal
                  the number of available devices (or the plan is still printed
                  even if devices aren't present yet).
    axis_names  : Names for the two mesh axes.  Default ('batch', 'model').
    arg_names   : Human-readable names for each argument (for the generated code).
    fn_name     : Name to use in comments.  Defaults to fn.__name__.

    Returns
    -------
    ShardingPlan with .code (copy-pasteable Python) and .tensors (per-arg specs).
    """
    name = fn_name or getattr(fn, "__name__", "fn")

    # Infer which args are activations / weights and their fan direction
    roles = _infer_roles(fn, args)

    # Determine batch size from first input's first dimension
    batch_size = args[0].shape[0] if args and hasattr(args[0], "shape") and args[0].shape else 1

    tensors: list[TensorPlan] = []
    for i, arg in enumerate(args):
        shape = tuple(arg.shape) if hasattr(arg, "shape") else ()
        role = roles.get(i, {})
        plan = _classify_arg(
            arg_idx=i,
            shape=shape,
            batch_size=batch_size,
            is_activation=role.get("is_activation", False),
            is_weight=role.get("is_weight", False),
            fan_out=role.get("fan_out", True),
        )
        if arg_names and i < len(arg_names):
            plan.name = arg_names[i]
        tensors.append(plan)

    # Build the plan (code filled in below)
    splan = ShardingPlan(
        fn_name=name,
        mesh_shape=mesh_shape,
        axis_names=axis_names,
        tensors=tensors,
        code="",
    )
    splan.code = _generate_code(splan, arg_names)
    return splan
