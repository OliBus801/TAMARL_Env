import gc

import torch


def analyze_tensor_memory(tag="SNAPSHOT", top_k=8):
    """
    Scans memory, calculates the total size of live PyTorch tensors,
    and displays the top_k largest objects.
    """
    # Force dead objects cleanup before counting
    gc.collect()

    tensor_stats = []
    total_size_mb = 0.0

    # Safe iteration over garbage collector objects
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                size_mb = obj.element_size() * obj.nelement() / (1024**2)
                total_size_mb += size_mb

                tensor_stats.append(
                    {"size_mb": size_mb, "shape": tuple(obj.shape), "dtype": str(obj.dtype)}
                )
        except Exception:
            pass

    # Sort by descending size
    tensor_stats.sort(key=lambda x: x["size_mb"], reverse=True)

    # Formatted display
    print(f"\n[{tag}] - Total Tensor Memory: {total_size_mb:.2f} MB")
    print(f"--- TOP {top_k} OBJECTS ---")
    for i, stat in enumerate(tensor_stats[:top_k]):
        print(
            f"  {i + 1}. Size: {stat['size_mb']:>8.2f} MB | Shape: {str(stat['shape']):<20} | Type: {stat['dtype']}"
        )
    print("-" * 50)
