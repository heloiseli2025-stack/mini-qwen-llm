"""
CLI 推理入口。

用法:
    python scripts/run_inference.py --prompt "你好，请介绍一下自己" --max_new_tokens 100
    python scripts/run_inference.py --model ./weights/qwen3-0.6b --device cuda
"""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from mini_qwen.model.loader import load_from_hf

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_from_hf(args.model, dtype=torch.float32).to(args.device).eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(args.device)

    print(f"Prompt: {args.prompt}")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )

    generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Output: {generated}")


if __name__ == "__main__":
    main()
