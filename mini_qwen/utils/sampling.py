import torch
import torch.nn.functional as F


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """logits [B, vocab] -> next_token [B, 1]。"""
    return logits.argmax(dim=-1, keepdim=True)


def top_p_sample(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> torch.Tensor:
    """Nucleus sampling。logits [B, vocab] -> next_token [B, 1]。"""
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    # 移除累积概率超过 top_p 的 token
    sorted_probs[cumsum - sorted_probs > top_p] = 0.0
    sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx.gather(dim=-1, index=next_token)
