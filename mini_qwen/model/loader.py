"""从 HuggingFace checkpoint 加载权重到 mini_qwen 模型。"""
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
    """从 HF checkpoint 构建并加载 Qwen3MoEForCausalLM。

    Args:
        model_name_or_path: HF 模型名或本地路径
        dtype:              加载权重的 dtype（默认 bfloat16）
        quantize_w4a16:     True → 加载后对所有 expert 做 W4A16 量化
                            （Qwen3-30B-A3B BF16 需 60GB，4090 必须量化）
        group_size:         W4A16 量化 group size

    Returns:
        Qwen3MoEForCausalLM（已加载权重，可选已量化）

    内存策略（解决 120GB cgroup 限制）：
      1. meta device 创建零内存骨架
      2. 加载 HF 权重（~58GB BF16）
      3. assign=True 直接接管 HF 张量，无拷贝
      全程峰值 ~58GB，远低于 float32 空模型的 ~116GB。
    """
    import gc
    from transformers import AutoConfig, AutoModelForCausalLM
    from mini_qwen.config import Qwen3MoEConfig
    from mini_qwen.model.qwen3_moe import Qwen3MoEForCausalLM

    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    our_config = Qwen3MoEConfig.from_hf_config(hf_config)

    # Step 1：用 meta device 创建骨架，获取所有期望 key 及形状，零内存开销
    # 必须用 state_dict().keys() 而非 named_parameters()，后者会跳过 tied weight 别名
    with torch.device("meta"):
        meta_model = Qwen3MoEForCausalLM(our_config)
    meta_sd  = meta_model.state_dict()   # 包含所有 key，含 tied weight 别名
    our_keys = set(meta_sd.keys())
    del meta_model

    # Step 2：加载 HF 权重（~58GB BF16）
    print(f"从 {model_name_or_path} 加载 MoE 权重...", flush=True)
    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    hf_state  = hf_model.state_dict()
    del hf_model
    gc.collect()

    matched = {k: hf_state[k] for k in our_keys if k in hf_state}

    # HF Qwen3MoE 将所有 expert 的权重打包成批量张量：
    #   experts.gate_up_proj  [E, 2*D, H]  （fused gate + up）
    #   experts.down_proj     [E, H, D]
    # 需要拆分成每个 expert 的单独 Linear：experts.{e}.gate_proj.weight 等。
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

    # Step 3：缺失 key 用空张量填充（极少数，e.g. tied weight 别名）
    for k in missing:
        t = meta_sd[k]
        matched[k] = torch.empty(t.shape, dtype=t.dtype if t.dtype != torch.float32 else dtype)

    del hf_state
    del meta_sd
    gc.collect()

    if missing:
        print(f"⚠  缺少权重（随机初始化）: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        print(f"   忽略 HF 多余 key: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")

    # Step 4：重新构建真实模型（meta device），assign=True 直接接管张量，无拷贝
    with torch.device("meta"):
        our_model = Qwen3MoEForCausalLM(our_config)
    our_model.load_state_dict(matched, strict=True, assign=True)
    del matched
    gc.collect()
    print(f"✓  加载完成，匹配 {len(our_keys) - len(missing)}/{len(our_keys)} 个张量", flush=True)

    # Step 5：非持久化 buffer（inv_freq / cos_cached / sin_cached）不在 state_dict 里，
    #         meta device 后仍为 meta tensor，调用 .to() 时会 crash。重新计算。
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
        print(f"量化 expert 权重为 W4A16（group_size={group_size}）...", flush=True)
        our_model.quantize_experts_to_w4a16(group_size=group_size)
        print("✓  量化完成", flush=True)

    return our_model


def load_from_hf(
    model_name_or_path: str,
    dtype: torch.dtype = torch.float32,
) -> Qwen3ForCausalLM:
    """从 HF checkpoint 构建并加载 Qwen3ForCausalLM。

    策略：先加载 HF 模型拿到 state_dict，再按 key 名映射到我们的模型。
    由于模块命名与 HF 保持一致，绝大多数 key 直接对应。
    HF 多余的 key（如 rotary_emb.inv_freq）被静默忽略。
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    our_config = Qwen3Config.from_hf_config(hf_config)
    our_model = Qwen3ForCausalLM(our_config)
    our_keys = set(our_model.state_dict().keys())

    print(f"从 {model_name_or_path} 加载权重...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    hf_state = hf_model.state_dict()
    del hf_model  # 立即释放显存/内存

    matched = {k: v for k, v in hf_state.items() if k in our_keys}
    missing = our_keys - set(matched.keys())
    extra = set(hf_state.keys()) - our_keys

    if missing:
        print(f"⚠  缺少权重（保持随机初始化）: {sorted(missing)}")
    if extra:
        shown = sorted(extra)[:5]
        print(f"   忽略 HF 多余 key: {shown}{'...' if len(extra) > 5 else ''}")

    our_model.load_state_dict(matched, strict=False)
    print(f"✓  加载完成，匹配 {len(matched)}/{len(our_keys)} 个张量")

    return our_model.to(dtype=dtype)

def load_moe_from_gptq(
    gptq_path: str,
    dtype: torch.dtype = torch.bfloat16,
    group_size: int = 128,
    zero_plus_one: bool = True,
    device: str = "cuda",
):
    """加载 GPTQ-Int4 checkpoint 到 Qwen3MoEForCausalLM（仅 CUDA）。

    GPTQ（checkpoint_format=gptq, desc_act=false）量化了所有 attention 投影
    (q/k/v/o_proj) 与 expert 投影 (gate/up/down_proj)；router(mlp.gate)/各 norm/
    embed/lm_head 保持 fp16。所有量化 Linear 替换为 LinearW4A16。

    Args:
        gptq_path:     GPTQ checkpoint 目录
        zero_plus_one: GPTQ v1 的 zero-point +1 修正（见 LinearW4A16.from_gptq）
        device:        运行设备（W4A16 kernel 仅支持 cuda）
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

    # 加载全部 GPTQ 张量到 CPU
    print(f"从 {gptq_path} 加载 GPTQ 张量...", flush=True)
    with open(os.path.join(gptq_path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    sd = {}
    for fn in sorted(set(weight_map.values())):
        with safe_open(os.path.join(gptq_path, fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)

    # 哪些 Linear 被量化（含 .qweight）
    quant_names = {k[: -len(".qweight")] for k in sd if k.endswith(".qweight")}

    # MoE router (mlp.gate) 也被 GPTQ 量化了，但 router 需 fp16 精度且 moe_router
    # 直接读 .weight；把它反量化成普通 weight，不走 LinearW4A16。
    from mini_qwen.quantization.packing import unpack_int4
    router_names = {n for n in quant_names if n.endswith(".mlp.gate")}
    quant_names -= router_names

    def _dequant_gptq(name):
        qw = sd[name + ".qweight"]            # [K//8, N] int32，沿 K 打包
        qz = sd[name + ".qzeros"]             # [G, N//8] int32，沿 N 打包
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
        return w.t().contiguous()             # [N, K] = nn.Linear weight (out, in)

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

    # 替换量化 Linear → LinearW4A16
    for name in quant_names:
        new = LinearW4A16.from_gptq(
            sd[name + ".qweight"], sd[name + ".qzeros"], sd[name + ".scales"],
            group_size, zero_plus_one,
        )
        set_submodule(model, name, new)

    # 非量化参数（norm/gate/embed/lm_head/q_norm/k_norm）→ 直接 assign
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

    # 重建 RoPE buffer（meta → real）
    inv_freq = 1.0 / (
        our_config.rope_theta
        ** (torch.arange(0, our_config.head_dim, 2, dtype=torch.float32) / our_config.head_dim)
    )
    for m in model.modules():
        if isinstance(m, RotaryEmbedding):
            m.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
            m._build_cache(our_config.max_position_embeddings)

    model.to(device)
    print(f"✓  GPTQ 加载完成：量化 {len(quant_names)} 个 Linear，device={device}", flush=True)
    return model

