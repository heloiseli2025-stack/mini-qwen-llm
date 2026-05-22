"""
对比 mini_qwen 与 HuggingFace 的 logits 输出。

用法:
    python scripts/compare_with_hf.py --model Qwen/Qwen3-0.6B --prompt "你好"
    python scripts/compare_with_hf.py --model ./weights/qwen3-0.6b --device cuda
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mini_qwen.model.loader import load_from_hf

    print(f"模型: {args.model}  |  Prompt: {args.prompt!r}  |  Device: {args.device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)

    print("加载 HF 模型...")
    hf_model = (
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        .to(args.device).eval()
    )

    print("加载 mini_qwen 模型...")
    our_model = load_from_hf(args.model, dtype=torch.float32)
    our_model = our_model.to(args.device).eval()

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        our_logits = our_model(input_ids)

    diff = (our_logits - hf_logits).abs()
    print(f"\nLogits shape : {our_logits.shape}")
    print(f"Max  abs err : {diff.max().item():.6e}")
    print(f"Mean abs err : {diff.mean().item():.6e}")
    print(f"Std  abs err : {diff.std().item():.6e}")

    top5_hf  = hf_logits[0, -1].topk(5)
    top5_our = our_logits[0, -1].topk(5)
    print(f"\nHF  top-5 next tokens: {[tokenizer.decode([i]) for i in top5_hf.indices.tolist()]}")
    print(f"Our top-5 next tokens: {[tokenizer.decode([i]) for i in top5_our.indices.tolist()]}")

    passed = diff.max().item() < 1e-4
    print(f"\n验收（max abs error < 1e-4）: {'✓ PASS' if passed else '✗ FAIL'}")


if __name__ == "__main__":
    main()
