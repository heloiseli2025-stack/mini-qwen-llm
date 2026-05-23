"""Load weights from a HuggingFace checkpoint into the mini_qwen model."""
from __future__ import annotations

import torch

from mini_qwen.config import Qwen3Config
from mini_qwen.model.qwen3 import Qwen3ForCausalLM


def load_moe_from_hf(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    quantize_w4a16: bool = False,
    group_size: int = 128,
):
    """Build and load Qwen3MoEForCausalLM from an HF checkpoint.

    Args:
        model_name_or_path: HF model name or local path
        dtype:              dtype for loading weights (default bfloat16)
        quantize_w4a16:     True -> apply W4A16 quantization to all experts after loading
                            (Qwen3-30B-A3B BF16 requires ~60GB; quantization is required for 4090)
        group_size:         W4A16 quantization group size

    Returns:
        Qwen3MoEForCausalLM (weights loaded, optionally quantized)

    Memory strategy (to work within a 120GB cgroup limit):
      1. Create a zero-memory skeleton on meta device
      2. Load HF weights (~58GB BF16)
      3. assign=True takes over HF tensors directly, no copy
      Peak memory ~58GB throughout, far below ~116GB for a float32 empty model.
    """
    import gc
    from transformers import AutoConfig, AutoModelForCausalLM
    from mini_qwen.config import Qwen3MoEConfig
    from mini_qwen.model.qwen3_moe import Qwen3MoEForCausalLM

    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    our_config = Qwen3MoEConfig.from_hf_config(hf_config)

    # Step 1: Create skeleton on meta device to get all expected keys and shapes at zero memory cost
    # Must use state_dict().keys() instead of named_parameters(), which skips tied weight aliases
    with torch.device("meta"):
        meta_model = Qwen3MoEForCausalLM(our_config)
    meta_sd  = meta_model.state_dict()   # Contains all keys, including tied weight aliases
    our_keys = set(meta_sd.keys())
    del meta_model

    # Step 2: Load HF weights (~58GB BF16)
    print(f"Loading MoE weights from {model_name_or_path}...", flush=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    hf_state  = hf_model.state_dict()
    del hf_model
    gc.collect()

    matched = {k: hf_state[k] for k in our_keys if k in hf_state}

    # HF Qwen3MoE packs all expert weights into batched tensors:
    #   experts.gate_up_proj  [E, 2*D, H]  (fused gate + up)
    #   experts.down_proj     [E, H, D]
    # We need to split them into per-expert Linears: experts.{e}.gate_proj.weight, etc.
    moe_int = our_config.intermediate_size   # D = 768
    for layer_idx in range(our_config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}.mlp.experts"
        gu_key = f"{prefix}.gate_up_proj"   # [E, 2*D, H]
        d_key  = f"{prefix}.down_proj"      # [E, H, D]
        if gu_key in hf_state and d_key in hf_state:
            gu = hf_state[gu_key]           # [E, 2*D, H]
            dw = hf_state[d_key]            # [E, H, D]
            for e in range(our_config.num_experts):
                ep = f"{prefix}.{e}"
                matched[f"{ep}.gate_proj.weight"] = gu[e, :moe_int, :]
                matched[f"{ep}.up_proj.weight"]   = gu[e, moe_int:, :]
                matched[f"{ep}.down_proj.weight"]  = dw[e]

    missing = our_keys - set(matched)
    extra   = set(hf_state) - our_keys - {
        f"model.layers.{i}.mlp.experts.gate_up_proj"
        for i in range(our_config.num_hidden_layers)
    } - {
        f"model.layers.{i}.mlp.experts.down_proj"
        for i in range(our_config.num_hidden_layers)
    }

    # Step 3: Fill missing keys with empty tensors (rare cases, e.g. tied weight aliases)
    for k in missing:
        t = meta_sd[k]
        matched[k] = torch.empty(t.shape, dtype=t.dtype if t.dtype != torch.float32 else dtype)

    del hf_state
    del meta_sd
    gc.collect()

    if missing:
        print(f"⚠  Missing weights (randomly initialized): {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"   Ignoring extra HF keys: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    # Step 4: Rebuild the real model (meta device), assign=True takes over tensors directly with no copy
    with torch.device("meta"):
        our_model = Qwen3MoEForCausalLM(our_config)
    our_model.load_state_dict(matched, strict=True, assign=True)
    del matched
    gc.collect()
    print(f"✓  Load complete, matched {len(our_keys) - len(missing)}/{len(our_keys)} tensors", flush=True)

    # Step 5: Non-persistent buffers (inv_freq / cos_cached / sin_cached) are not in state_dict;
    #         after meta device they remain meta tensors and will crash on .to(). Recompute them.
    from mini_qwen.model.layers.rope import RotaryEmbedding
    inv_freq_cpu = 1.0 / (
        our_config.rope_theta
        ** (torch.arange(0, our_config.head_dim, 2, dtype=torch.float32) / our_config.head_dim)
    )
    for module in our_model.modules():
        if isinstance(module, RotaryEmbedding):
            module.register_buffer("inv_freq", inv_freq_cpu.clone(), persistent=False)
            module._build_cache(our_config.max_position_embeddings)

    if quantize_w4a16:
        print(f"Quantizing expert weights to W4A16 (group_size={group_size})...", flush=True)
        our_model.quantize_experts_to_w4a16(group_size=group_size)
        print("✓  Quantization complete", flush=True)

    return our_model


def load_from_hf(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float32,
) -> Qwen3ForCausalLM:
    """Build and load Qwen3ForCausalLM from an HF checkpoint.

    Strategy: load the HF model to obtain its state_dict, then map keys to our model.
    Since module names match HF, the vast majority of keys correspond directly.
    Extra HF keys (e.g. rotary_emb.inv_freq) are silently ignored.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    our_config = Qwen3Config.from_hf_config(hf_config)
    our_model = Qwen3ForCausalLM(our_config)
    our_keys = set(our_model.state_dict().keys())

    print(f"Loading weights from {model_name_or_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    hf_state = hf_model.state_dict()
    del hf_model  # Immediately free GPU/CPU memory

    matched = {k: v for k, v in hf_state.items() if k in our_keys}
    missing = our_keys - set(matched.keys())
    extra = set(hf_state.keys()) - our_keys

    if missing:
        print(f"⚠  Missing weights (kept randomly initialized): {sorted(missing)}")
    if extra:
        shown = sorted(extra)[:5]
        print(f"   Ignoring extra HF keys: {shown}{'...' if len(extra) > 5 else ''}")

    our_model.load_state_dict(matched, strict=False)
    print(f"✓  Load complete, matched {len(matched)}/{len(our_keys)} tensors")

    return our_model.to(dtype=dtype)

def load_moe_from_gptq(
    gptq_path: str,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 128,
    zero_plus_one: bool = True,
    device: str = "cuda",
):
    """Load a GPTQ-Int4 checkpoint into Qwen3MoEForCausalLM (CUDA only).

    GPTQ (checkpoint_format=gptq, desc_act=false) quantizes all attention projections
    (q/k/v/o_proj) and expert projections (gate/up/down_proj); router (mlp.gate), norms,
    embed, and lm_head remain in fp16. All quantized Linears are replaced with LinearW4A16.

    Args:
        gptq_path:     GPTQ checkpoint directory
        zero_plus_one: GPTQ v1 zero-point +1 correction (see LinearW4A16.from_gptq)
        device:        Target device (W4A16 kernel only supports cuda)
    """
    import os, json, gc
    import torch.nn as nn
    from transformers import AutoConfig
    from safetensors import safe_open
    from mini_qwen.config import Qwen3MoEConfig
    from mini_qwen.model.qwen3_moe import Qwen3MoEForCausalLM
    from mini_qwen.model.layers.linear_w4a16 import LinearW4A16
    from mini_qwen.model.layers.rope import RotaryEmbedding

    hf_config = AutoConfig.from_pretrained(gptq_path)
    our_config = Qwen3MoEConfig.from_hf_config(hf_config)

    with torch.device("meta"):
        model = Qwen3MoEForCausalLM(our_config)

    # Load all GPTQ tensors to CPU
    print(f"Loading GPTQ tensors from {gptq_path}...", flush=True)
    with open(os.path.join(gptq_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    sd = {}
    for fn in sorted(set(weight_map.values())):
        with safe_open(os.path.join(gptq_path, fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)

    # Which Linears are quantized (contain .qweight)
    quant_names = {k[: -len(".qweight")] for k in sd if k.endswith(".qweight")}

    # The MoE router (mlp.gate) is also GPTQ-quantized, but the router needs fp16 precision
    # and moe_router reads .weight directly; dequantize it to a plain weight, bypassing LinearW4A16.
    from mini_qwen.quantization.packing import unpack_int4
    router_names = {n for n in quant_names if n.endswith(".mlp.gate")}
    quant_names -= router_names

    def _dequant_gptq(name):
        qw = sd[name + ".qweight"]            # [K//8, N] int32, packed along K
        qz = sd[name + ".qzeros"]             # [G, N//8] int32, packed along N
        sc = sd[name + ".scales"].float()     # [G, N]
        K = qw.shape[0] * 8
        N = qw.shape[1]
        G = K // group_size
        w_int = unpack_int4(qw.t().contiguous()).t().float()   # [K, N]
        z_int = unpack_int4(qz).float()                         # [G, N]
        off = 1.0 if zero_plus_one else 0.0
        w = torch.empty(K, N)
        for g in range(G):
            s, e = g * group_size, (g + 1) * group_size
            w[s:e] = (w_int[s:e] - (z_int[g][None, :] + off)) * sc[g][None, :]
        return w.t().contiguous()             # [N, K] = nn.Linear weight layout (out, in)

    def set_submodule(root, name, new_mod):
        parts = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        last = parts[-1]
        if last.isdigit():
            parent[int(last)] = new_mod
        else:
            setattr(parent, last, new_mod)

    # Replace quantized Linears with LinearW4A16
    for name in quant_names:
        new = LinearW4A16.from_gptq(
            sd[name + ".qweight"], sd[name + ".qzeros"], sd[name + ".scales"],
            group_size, zero_plus_one,
        )
        set_submodule(model, name, new)

    # Non-quantized parameters (norm/gate/embed/lm_head/q_norm/k_norm) -> assign directly
    nonq = {
        k: v.to(dtype)
        for k, v in sd.items()
        if k.endswith(".weight") and k[: -len(".weight")] not in quant_names
    }
    for rn in router_names:
        nonq[rn + ".weight"] = _dequant_gptq(rn).to(dtype)
    model.load_state_dict(nonq, strict=False, assign=True)

    if our_config.tie_word_embeddings and "lm_head.weight" not in nonq:
        model.lm_head.weight = model.model.embed_tokens.weight

    del sd, nonq
    gc.collect()

    # Rebuild RoPE buffers (meta -> real)
    inv_freq = 1.0 / (
        our_config.rope_theta
        ** (torch.arange(0, our_config.head_dim, 2, dtype=torch.float32) / our_config.head_dim)
    )
    for m in model.modules():
        if isinstance(m, RotaryEmbedding):
            m.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
            m._build_cache(our_config.max_position_embeddings)

    model.to(device)
    print(f"✓  GPTQ load complete: quantized {len(quant_names)} Linears, device={device}", flush=True)
    return model

