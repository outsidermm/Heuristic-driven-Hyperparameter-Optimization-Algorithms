"""Demo: run all three tuners without TensorFlow or dataset files.

Uses a synthetic training function that simulates realistic accuracy/time curves
so the search logic can be exercised end-to-end on any machine.

Usage:
    uv run python demo.py
"""

import math
import time


def synthetic_epoch_fn(epoch: int) -> tuple[float, float]:
    """Accuracy grows logarithmically; time grows linearly."""
    accuracy = 0.05 + 0.45 * math.log(epoch + 1) / math.log(251)
    elapsed = epoch * 0.01  # 10 ms per epoch (simulated)
    return elapsed, accuracy


def synthetic_batch_size_fn(batch_size: int) -> tuple[float, float]:
    """Accuracy stays flat up to batch 256 then degrades; time decreases."""
    # batch_size here is 2**log2_bs; acceptable range is 0.30 so flat region is wide
    log2_bs = math.log2(batch_size)
    accuracy = max(0.70 - max(log2_bs - 8, 0) * 0.15, 0.0)
    elapsed = 100.0 / log2_bs
    return elapsed, accuracy


def synthetic_lr_fn(lr: float) -> tuple[float, float]:
    """Accuracy peaks sharply at lr=1e-2 (exponent=2) — well above neighbours."""
    log_lr = -math.log10(lr)  # exponent: 1→lr=0.1, 2→lr=0.01, ...
    accuracy = max(0.80 - (log_lr - 2) ** 2 * 0.12, 0.0)
    elapsed = 5.0
    return elapsed, accuracy


def main() -> None:
    from algorithm import BatchSizeTuner, EpochTuner, LrTuner

    print("=== EpochTuner demo ===")
    t0 = time.perf_counter()
    tuner = EpochTuner(
        "cifar100",
        left_bound=10,
        right_bound=200,
        exploration_factor=5,
        training_fn=synthetic_epoch_fn,
    )
    best_epoch, acc, elapsed = tuner.binary_search_efficient_epoch()
    print(f"  Best epoch : {best_epoch}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Simulated  : {elapsed:.2f}s  (real wall time: {time.perf_counter()-t0:.3f}s)")

    print()
    print("=== BatchSizeTuner demo ===")
    t0 = time.perf_counter()
    bs_tuner = BatchSizeTuner(
        "cifar100",
        left_bound=4,   # 2^4  = 16
        right_bound=12, # 2^12 = 4096
        acceptable_range=0.30,
        training_fn=synthetic_batch_size_fn,
    )
    best_bs, acc, elapsed = bs_tuner.search()
    print(f"  Best batch size : 2^{best_bs} = {2**best_bs}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Simulated       : {elapsed:.2f}s  (real wall time: {time.perf_counter()-t0:.3f}s)")

    print()
    print("=== LrTuner demo ===")
    t0 = time.perf_counter()
    lr_tuner = LrTuner(
        "cifar100",
        left_bound=1,  # 10^-1 = 0.1
        right_bound=7, # 10^-7
        local_extrema_allowance=0.05,
        training_fn=synthetic_lr_fn,
    )
    lr_exp, acc, elapsed = lr_tuner.search()
    print(f"  Best lr exponent : {lr_exp}  (lr = 1e-{lr_exp} = {10**-lr_exp:.0e})")
    print(f"  Accuracy         : {acc:.4f}")
    print(f"  Simulated        : {elapsed:.2f}s  (real wall time: {time.perf_counter()-t0:.3f}s)")


if __name__ == "__main__":
    main()
