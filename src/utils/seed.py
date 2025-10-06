import os, random
import numpy as np
import torch


# src/utils/seed.py
import os, random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU single-thread math
    torch.set_num_threads(1)
    # Interop threads must be set before parallel work starts; skip if too late
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # It’s fine for determinism; this mainly affects perf, not numeric stability.
        pass

    # Disable oneDNN fused kernels (can vary by CPU)
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass

    # Enforce deterministic ops
    torch.use_deterministic_algorithms(True, warn_only=False)



def per_round_client_seed(global_seed: int, cid: str, rnd: int) -> int:
    """
    Stable per-(client,round) seed derived from a single global seed.
    Ensures each client’s loaders/dropout/etc. are reproducible every round.
    """
    return (hash(cid) & 0xFFFF_FFFF) ^ (global_seed + 1000 * int(rnd))