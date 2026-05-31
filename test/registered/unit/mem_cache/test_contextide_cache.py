"""CPU-only unit tests for ContextIDe cache metadata and lifecycle helpers."""

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.contextide_cache import (
    ContextIDeHiRadixCache,
    _ContextIDeNodeList,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_node(token: int) -> TreeNode:
    node = TreeNode()
    node.key = RadixKey([token])
    node.value = torch.tensor([token])
    return node


def _make_metadata_cache() -> ContextIDeHiRadixCache:
    cache = ContextIDeHiRadixCache.__new__(ContextIDeHiRadixCache)
    cache.small_fifo = _ContextIDeNodeList()
    cache.main_fifo = _ContextIDeNodeList()
    cache.hbm_lru = _ContextIDeNodeList()
    cache.main_freq = {}
    cache.node_tier = {}
    cache._pending_write_through = OrderedDict()
    cache._contextide_demote_after_write = set()
    cache._contextide_write_back = set()
    return cache


def test_small_fifo_retains_blocked_radix_node_metadata():
    cache = _make_metadata_cache()
    cache.small_capacity_pages = 1
    blocked = _make_node(1)
    evictable = _make_node(2)
    cache.small_fifo.add_head(blocked)
    cache.small_fifo.add_head(evictable)
    cache._evict_host_node = MagicMock(
        side_effect=lambda node, ghost: node is evictable
    )

    cache._evict_small_fifo_if_needed()

    assert blocked in cache.small_fifo
    assert evictable not in cache.small_fifo
    assert len(cache.small_fifo) == 1


def test_main_fifo_frequency_decay_does_not_stop_capacity_recovery():
    cache = _make_metadata_cache()
    cache.main_capacity_pages = 1
    hot = _make_node(1)
    cold = _make_node(2)
    cache.main_fifo.add_head(hot)
    cache.main_fifo.add_head(cold)
    cache.main_freq[hot.id] = 2
    cache._evict_host_node = MagicMock(return_value=True)

    cache._evict_main_fifo_if_needed()

    assert hot in cache.main_fifo
    assert cold not in cache.main_fifo
    assert cache.main_freq[hot.id] == 1
    assert len(cache.main_fifo) == 1


def test_chunked_new_page_is_still_written_through():
    cache = _make_metadata_cache()
    cache.page_size = 1
    cache.evictable_size_ = 0
    cache.enable_storage = False
    cache.enable_kv_cache_events = False
    cache._update_leaf_status = MagicMock()
    cache._record_store_event = MagicMock()
    cache._ensure_write_through = MagicMock()
    cache._mark_hbm = MagicMock()
    parent = TreeNode()

    node = cache._add_page_node(
        parent=parent,
        key=RadixKey([1]),
        value=torch.tensor([1]),
        priority=0,
        chunked=True,
    )

    cache._ensure_write_through.assert_called_once_with(node)
    assert node.hit_count == 0


def test_pending_write_through_is_retried_after_temporary_failure():
    cache = _make_metadata_cache()
    node = _make_node(1)
    cache.write_backup = MagicMock(side_effect=[0, 1])

    cache._ensure_write_through(node)
    assert list(cache._pending_write_through) == [node.id]

    cache._retry_pending_write_through()
    assert not cache._pending_write_through
    assert node.id in cache._contextide_demote_after_write


def test_dec_lock_ref_demotes_idle_main_page_but_keeps_tail_in_hbm():
    cache = _make_metadata_cache()
    cache.root_node = TreeNode()
    main = _make_node(1)
    main.parent = cache.root_node
    main.host_value = torch.tensor([11])
    tail = _make_node(2)
    tail.parent = cache.root_node
    tail.host_value = torch.tensor([12])
    cache.node_tier[main.id] = "main"
    cache._evict_backuped = MagicMock()
    cache._mark_hbm = MagicMock()

    with patch.object(HiRadixCache, "dec_lock_ref", return_value=MagicMock()):
        cache.dec_lock_ref(main)
        cache.dec_lock_ref(tail)

    cache._evict_backuped.assert_called_once_with(main)
    cache._mark_hbm.assert_called_once_with(tail)


def test_complete_write_tracks_hbm_write_back_in_small_fifo_without_unlock():
    cache = _make_metadata_cache()
    node = _make_node(1)
    node.host_value = torch.tensor([11])
    cache._contextide_write_back.add(node.id)
    cache._record_store_event = MagicMock()
    cache.dec_lock_ref = MagicMock()
    cache._mark_small = MagicMock()
    cache._evict_backuped = MagicMock()
    cache.enable_storage = False

    cache._complete_write(node)

    cache.dec_lock_ref.assert_not_called()
    cache._mark_small.assert_called_once_with(node)
    assert node.id not in cache._contextide_write_back


def test_runtime_storage_attach_cannot_override_contextide_write_policy():
    cache = _make_metadata_cache()

    with patch.object(
        HiRadixCache, "attach_storage_backend", return_value=(True, "ok")
    ) as attach:
        result = cache.attach_storage_backend(
            storage_backend="file", hicache_write_policy="write_back"
        )

    assert result == (True, "ok")
    assert attach.call_args.kwargs["hicache_write_policy"] == "write_through"


def test_hbm_lru_can_demote_backed_internal_prefix_page():
    cache = _make_metadata_cache()
    cache.update_eviction_metrics = MagicMock()
    prefix = _make_node(1)
    prefix.host_value = torch.tensor([11])
    child = _make_node(2)
    prefix.children[child.key.child_key(1)] = child
    cache.hbm_lru.add_head(prefix)
    cache._evict_backuped = MagicMock(return_value=1)

    result = cache.evict(EvictParams(num_tokens=1))

    cache._evict_backuped.assert_called_once_with(prefix)
    assert result.num_tokens_evicted == 1
