"""M0 验收测试：mini_qwen 输出与 HF 的 max abs error < 1e-4。"""
import pytest
import torch


def test_output_matches_hf(model_name, device):
    """验收标准：与 AutoModelForCausalLM 输出 max abs error < 1e-4（fp32）。"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from mini_qwen.model.loader import load_from_hf
    except ImportError as e:
        pytest.skip(f"依赖缺失: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        our_model = load_from_hf(model_name, dtype=torch.float32)
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    except Exception as e:
        pytest.skip(f"模型未下载或不可用: {e}")

    our_model = our_model.to(device).eval()
    hf_model = hf_model.to(device).eval()

    inputs = tokenizer("你好", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        our_logits = our_model(input_ids)           # [1, seq, vocab]
        hf_logits = hf_model(input_ids).logits      # [1, seq, vocab]

    diff = (our_logits - hf_logits).abs()
    max_err = diff.max().item()
    print(f"\nmax abs error: {max_err:.6e}")
    assert max_err < 1e-4, f"max abs error {max_err:.6e} 超过 1e-4"


def test_greedy_generate(model_name, device):
    """验证 generate() 能跑完、返回合理 token 数。"""
    try:
        from transformers import AutoTokenizer
        from mini_qwen.model.loader import load_from_hf
    except ImportError as e:
        pytest.skip(f"依赖缺失: {e}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = load_from_hf(model_name, dtype=torch.float32)
    except Exception as e:
        pytest.skip(f"模型未下载或不可用: {e}")

    model = model.to(device).eval()
    inputs = tokenizer("你好", return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10)

    prompt_len = input_ids.shape[1]
    assert output.shape[1] == prompt_len + 10
    generated = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    print(f"\n生成内容: {generated}")
