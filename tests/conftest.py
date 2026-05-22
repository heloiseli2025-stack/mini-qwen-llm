import pytest
import torch


@pytest.fixture(scope="session")
def model_name():
    return "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
