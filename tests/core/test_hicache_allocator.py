from __future__ import annotations

import pytest
import torch
from sgl.kernel import hicache


def test_allocate_host_can_use_torch_allocator(monkeypatch):
    expected = torch.empty(4, dtype=torch.uint8)

    def fake_torch_host(*shape: int, dtype: torch.dtype):
        assert shape == (4,)
        assert dtype is torch.uint8
        return expected

    monkeypatch.setenv("SGLANG_HICACHE_HOST_ALLOCATOR", "torch")
    monkeypatch.setattr(hicache, "_allocate_torch_host", fake_torch_host)

    assert hicache.allocate_host(4, dtype=torch.uint8) is expected


def test_allocate_host_rejects_unknown_allocator(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_HOST_ALLOCATOR", "unknown")

    with pytest.raises(ValueError, match="SGLANG_HICACHE_HOST_ALLOCATOR"):
        hicache.allocate_host(4, dtype=torch.uint8)


def test_allocate_host_falls_back_to_torch_when_numa_unavailable(monkeypatch):
    expected = torch.empty(4, dtype=torch.uint8)

    def raise_numa_unavailable():
        raise RuntimeError("NUMA unavailable")

    def fake_torch_host(*shape: int, dtype: torch.dtype):
        assert shape == (4,)
        assert dtype is torch.uint8
        return expected

    monkeypatch.delenv("SGLANG_HICACHE_HOST_ALLOCATOR", raising=False)
    monkeypatch.setattr(hicache, "probe_numa_node", raise_numa_unavailable)
    monkeypatch.setattr(hicache, "_allocate_torch_host", fake_torch_host)

    assert hicache.allocate_host(4, dtype=torch.uint8) is expected
