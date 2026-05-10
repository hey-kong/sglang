from __future__ import annotations

import pytest
import sgl.core as core
import torch
from sgl.kvcache.hiradix_cache import HiRadixPrefixCache


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    core.set_global_ctx(core.Context(page_size=1))
    yield
    core._GLOBAL_CTX = old_ctx


def _insert(cache: HiRadixPrefixCache, token: int, index: int):
    cache.insert_prefix(
        torch.tensor([token], dtype=torch.int32),
        torch.tensor([index], dtype=torch.int32),
    )
    return cache.root_node.children[token]


def test_fifo_does_not_refresh_timestamp_on_hit():
    cache = HiRadixPrefixCache(device=torch.device("cpu"), hicache_policy="fifo")
    first = _insert(cache, token=1, index=11)
    second = _insert(cache, token=2, index=22)
    first.timestamp = 1
    second.timestamp = 2

    cache.match_prefix(torch.tensor([1], dtype=torch.int32))

    assert first.timestamp == 1
    assert cache.evict(1).tolist() == [11]


def test_lfu_evicts_least_frequently_hit_node():
    cache = HiRadixPrefixCache(device=torch.device("cpu"), hicache_policy="lfu")
    first = _insert(cache, token=1, index=11)
    second = _insert(cache, token=2, index=22)

    cache.match_prefix(torch.tensor([1], dtype=torch.int32))
    cache.match_prefix(torch.tensor([1], dtype=torch.int32))

    assert first.freq == 2
    assert second.freq == 0
    assert cache.evict(1).tolist() == [22]


def test_node_prefix_hash_tracks_full_prefix_ids_after_split():
    cache = HiRadixPrefixCache(device=torch.device("cpu"), hicache_policy="lru")
    cache.insert_prefix(
        torch.tensor([1, 2, 3], dtype=torch.int32),
        torch.tensor([11, 22, 33], dtype=torch.int32),
    )

    cache.insert_prefix(
        torch.tensor([1, 2, 4], dtype=torch.int32),
        torch.tensor([11, 22, 44], dtype=torch.int32),
    )

    prefix_node = cache.root_node.children[1]
    assert prefix_node.prefix_hash == hash((1, 2))
    assert prefix_node.children[3].prefix_hash == hash((1, 2, 3))
    assert prefix_node.children[4].prefix_hash == hash((1, 2, 4))
