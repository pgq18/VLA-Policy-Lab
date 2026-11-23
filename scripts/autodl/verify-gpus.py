import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(1234)
    a = torch.ones((128, 64), device=device)
    b = torch.ones((64, 96), device=device)
    assert torch.all(a @ b == 64).item()
    dist.init_process_group("nccl", timeout=timedelta(seconds=600))
    x = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(x)
    assert x.item() == 10
    report = dict(rank=rank, torch=torch.__version__, cuda=torch.version.cuda,
                  nccl=list(torch.cuda.nccl.version()), capability=list(torch.cuda.get_device_capability(rank)),
                  gpu=torch.cuda.get_device_name(rank), matmul=True, all_reduce=x.item())
    output = Path(__file__).resolve().parents[2] / "outputs/autodl_deployment/gpu-validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / f"rank{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
