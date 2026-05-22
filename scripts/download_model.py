"""
从 HuggingFace 下载 Qwen3 模型权重。

用法:
    python scripts/download_model.py --model Qwen/Qwen3-0.6B
    python scripts/download_model.py --model Qwen/Qwen3-8B --local_dir ./weights/qwen3-8b
    # 国内镜像：
    python scripts/download_model.py --model Qwen/Qwen3-0.6B --endpoint https://hf-mirror.com
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--endpoint", default=None, help="HF 镜像 endpoint（国内用）")
    args = parser.parse_args()

    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    from huggingface_hub import snapshot_download

    local_dir = args.local_dir or f"./weights/{args.model.split('/')[-1].lower()}"
    print(f"下载 {args.model} -> {local_dir}")
    snapshot_download(repo_id=args.model, local_dir=local_dir)
    print("✓ 下载完成")


if __name__ == "__main__":
    main()
