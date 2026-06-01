from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.events import StorageMedium
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.utils import compute_node_hash_values

logger = logging.getLogger(__name__)


class _ContextIDeNodeList:
    """Ordered node container used as FIFO/LRU metadata.

    The values are radix TreeNode references; each node can appear at most once in
    a list.  The tail (last item) is the eviction candidate.
    """

    def __init__(self):
        self._nodes: OrderedDict[int, TreeNode] = OrderedDict()

    def __contains__(self, node: TreeNode) -> bool:
        return node.id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def add_head(self, node: TreeNode) -> None:
        self._nodes.pop(node.id, None)
        self._nodes[node.id] = node
        self._nodes.move_to_end(node.id, last=False)

    def add_tail(self, node: TreeNode) -> None:
        self._nodes.pop(node.id, None)
        self._nodes[node.id] = node

    def remove(self, node: Optional[TreeNode]) -> None:
        if node is not None:
            self._nodes.pop(node.id, None)

    def move_head(self, node: TreeNode) -> None:
        if node.id in self._nodes:
            self._nodes.move_to_end(node.id, last=False)

    def pop_tail(self) -> Optional[TreeNode]:
        while self._nodes:
            _, node = self._nodes.popitem(last=True)
            return node
        return None

    def tail_items(self):
        for node in reversed(self._nodes.values()):
            yield node


class ContextIDeHiRadixCache(HiRadixCache):
    """HiRadixCache variant with page-sized nodes and FIFO metadata.

    ContextIDe keeps one radix node per KV page and keeps host-resident pages in
    small/main FIFO lists inspired by S3-FIFO.  Device/HBM entries are tracked in
    an LRU list for page-granular HBM eviction.  The radix tree remains the source
    of truth; lists are eviction metadata and only point to existing TreeNode
    objects.
    """

    def __init__(self, params, server_args):
        if server_args.main_page_size % params.page_size != 0:
            raise ValueError(
                f"--main-page-size ({server_args.main_page_size}) must be a multiple "
                f"of --page-size ({params.page_size})."
            )
        super().__init__(params=params, server_args=server_args)
        # ContextIDe ignores the user supplied hicache write policy.
        self.cache_controller.write_policy = "write_through"
        self.write_through_threshold = 1

        self.main_page_size = server_args.main_page_size
        self.main_pages_per_entry = max(1, self.main_page_size // self.page_size)
        self.main_size_ratio = server_args.main_size_ratio
        self.ghost_size_ratio = server_args.ghost_size_ratio

        host_token_capacity = self.cache_controller.mem_pool_host.size
        host_page_capacity = max(1, host_token_capacity // self.page_size)
        self.main_capacity_pages = max(
            1, int(host_page_capacity * self.main_size_ratio)
        )
        self.small_capacity_pages = max(
            1, host_page_capacity - self.main_capacity_pages
        )
        self.ghost_capacity_pages = max(
            1, int(host_page_capacity * self.ghost_size_ratio)
        )

        self.hbm_lru = _ContextIDeNodeList()
        self.small_fifo = _ContextIDeNodeList()
        self.main_fifo = _ContextIDeNodeList()
        self.ghost_fifo: OrderedDict[str, None] = OrderedDict()
        self.main_freq: dict[int, int] = {}
        self.node_tier: dict[int, str] = {}
        self._contextide_demote_after_write: OrderedDict[int, TreeNode] = OrderedDict()
        self._contextide_main_after_write: set[int] = set()
        self._pending_write_through: OrderedDict[int, TreeNode] = OrderedDict()

        logger.info(
            "ContextIDe HiCache enabled: page_size=%d main_page_size=%d "
            "small_pages=%d main_pages=%d ghost_pages=%d",
            self.page_size,
            self.main_page_size,
            self.small_capacity_pages,
            self.main_capacity_pages,
            self.ghost_capacity_pages,
        )

    def attach_storage_backend(
        self,
        storage_backend: str,
        storage_backend_extra_config_json: Optional[str] = None,
        served_model_name: Optional[str] = None,
        hicache_storage_prefetch_policy: Optional[str] = None,
        hicache_write_policy: Optional[str] = None,
    ) -> tuple[bool, str]:
        # ContextIDe semantics are always write-through, including runtime storage
        # attachment requests that attempt to update the generic HiCache policy.
        return super().attach_storage_backend(
            storage_backend=storage_backend,
            storage_backend_extra_config_json=storage_backend_extra_config_json,
            served_model_name=served_model_name,
            hicache_storage_prefetch_policy=hicache_storage_prefetch_policy,
            hicache_write_policy="write_through",
        )

    def _ensure_write_through(self, node: TreeNode) -> None:
        if node.backuped or node.value is None:
            self._pending_write_through.pop(node.id, None)
            return
        written = self.write_backup(node)
        if written > 0:
            self._pending_write_through.pop(node.id, None)
            self._contextide_demote_after_write[node.id] = node
        else:
            # A write can be temporarily blocked by an unfinished parent-prefix
            # write or host pressure. Retry after the next write acknowledgement.
            self._pending_write_through[node.id] = node

    def _retry_pending_write_through(self) -> None:
        for node in list(self._pending_write_through.values()):
            self._ensure_write_through(node)

    def _retry_pending_hbm_demotions(self) -> None:
        """Release HBM copies after write-through once their locks are gone."""
        for node in list(self._contextide_demote_after_write.values()):
            if node.value is None:
                self._contextide_demote_after_write.pop(node.id, None)
                continue
            if node.lock_ref > 0:
                continue
            _, fully_evicted = self._evict_hbm_subtree(node)
            if fully_evicted or node.value is None:
                self._contextide_demote_after_write.pop(node.id, None)

    def _inc_hit_count(self, node: TreeNode, chunked=False):
        if chunked:
            return
        node.hit_count += 1
        if node.hit_count >= self.write_through_threshold:
            self._ensure_write_through(node)

    def _node_hash_key(self, node: TreeNode) -> str:
        if node.hash_value:
            return node.hash_value[-1]
        return str(hash(tuple(node.key.token_ids)))

    def _remove_node_from_lists(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        self.hbm_lru.remove(node)
        self.small_fifo.remove(node)
        self.main_fifo.remove(node)
        self.main_freq.pop(node.id, None)
        self.node_tier.pop(node.id, None)
        self._pending_write_through.pop(node.id, None)
        self._contextide_demote_after_write.pop(node.id, None)
        self._contextide_main_after_write.discard(node.id)

    def _mark_hbm(self, node: TreeNode) -> None:
        if (
            node is not None
            and node != self.root_node
            and node.value is not None
            and node.lock_ref == 0
        ):
            self.hbm_lru.add_head(node)

    def _order_fifo_path_leaf_first(self, node: TreeNode, tier: str) -> None:
        """Keep descendants closer to the FIFO tail than their ancestors.

        FIFO eviction pops from the tail.  Re-inserting the matching path from
        leaf to root with ``add_head`` therefore makes removable leaf nodes reach
        the eviction end before their prefix nodes.
        """
        fifo = self.small_fifo if tier == "small" else self.main_fifo
        while node is not None and node != self.root_node:
            if self.node_tier.get(node.id) == tier:
                fifo.add_head(node)
            node = node.parent

    def _mark_small(self, node: TreeNode) -> None:
        if node.host_value is None:
            return
        self.main_fifo.remove(node)
        self.main_freq.pop(node.id, None)
        self.node_tier[node.id] = "small"
        self._order_fifo_path_leaf_first(node, "small")
        self._evict_small_fifo_if_needed()

    def _promote_to_main(self, node: TreeNode) -> None:
        if node.host_value is None:
            return
        self.small_fifo.remove(node)
        self.node_tier[node.id] = "main"
        self.main_freq[node.id] = min(3, self.main_freq.get(node.id, 0) + 1)
        self._order_fifo_path_leaf_first(node, "main")
        self._evict_main_fifo_if_needed()

    def _touch_host_hit_chain(self, last_host_node: TreeNode) -> None:
        nodes = []
        node = last_host_node
        while node is not None and node != self.root_node and node.backuped:
            nodes.append(node)
            node = node.parent
        nodes.reverse()

        main_aligned_count = (
            len(nodes) // self.main_pages_per_entry * self.main_pages_per_entry
        )
        for node in nodes[:main_aligned_count]:
            self._promote_to_main(node)
        for node in nodes[main_aligned_count:]:
            # Tail pages stay in HBM for the active request.  If already loaded,
            # refresh HBM LRU; otherwise init_load_back will promote them.
            self._mark_hbm(node)

    def _is_host_evictable_leaf(self, node: TreeNode) -> bool:
        return (
            node is not None
            and node != self.root_node
            and node.host_value is not None
            and node.host_ref_counter == 0
            and node.evicted
            and len(node.children) == 0
        )

    def _evict_host_node(self, node: TreeNode, *, ghost: bool) -> bool:
        if not self._is_host_evictable_leaf(node):
            return False
        if ghost:
            self._add_ghost(self._node_hash_key(node))
        self._remove_node_from_lists(node)
        self.evictable_host_leaves.discard(node)
        self._record_remove_event(node, medium=StorageMedium.CPU)
        host_value = node.host_value
        node.host_value = None
        self.cache_controller.evict_host(host_value)
        # Host-only leaf with no device value can be deleted from the radix tree.
        key = node.key.child_key(self.page_size)
        node.parent.children.pop(key, None)
        self._update_host_leaf_status(node.parent)
        self._update_leaf_status(node.parent)
        return True

    def _evict_main_host_node(self, node: TreeNode) -> bool:
        # Main FIFO eviction drops the HBM subtree first. This keeps HBM eviction
        # leaf-first and prevents a DRAM prefix from being removed while device
        # pages below it are still resident.
        _, fully_evicted = self._evict_hbm_subtree(node)
        if not fully_evicted:
            return False
        return self._evict_host_node(node, ghost=False)

    def evict_host(self, num_tokens: int):
        target_pages = max(1, (num_tokens + self.page_size - 1) // self.page_size)
        evicted_pages = 0
        blocked_pages = 0
        while evicted_pages < target_pages and blocked_pages < len(self.small_fifo):
            node = self.small_fifo.pop_tail()
            if node is None:
                break
            if self._evict_host_node(node, ghost=True):
                evicted_pages += 1
                blocked_pages = 0
            else:
                self.small_fifo.add_head(node)
                blocked_pages += 1

        blocked_pages = 0
        while evicted_pages < target_pages and blocked_pages < len(self.main_fifo):
            node = self.main_fifo.pop_tail()
            if node is None:
                break
            freq = self.main_freq.get(node.id, 0)
            if freq > 0:
                self.main_freq[node.id] = freq - 1
                self.main_fifo.add_head(node)
                blocked_pages = 0
                continue
            if self._evict_main_host_node(node):
                evicted_pages += 1
                blocked_pages = 0
            else:
                self.main_fifo.add_head(node)
                blocked_pages += 1

    def _add_ghost(self, key: str) -> None:
        self.ghost_fifo.pop(key, None)
        self.ghost_fifo[key] = None
        self.ghost_fifo.move_to_end(key, last=False)
        while len(self.ghost_fifo) > self.ghost_capacity_pages:
            self.ghost_fifo.popitem(last=True)

    def _is_ghost_hit(self, node: TreeNode) -> bool:
        return self._node_hash_key(node) in self.ghost_fifo

    def _consume_ghost_hit(self, node: TreeNode) -> None:
        self.ghost_fifo.pop(self._node_hash_key(node), None)

    def _admit_new_pages(self, nodes: list[TreeNode]) -> None:
        """Admit a newly created suffix after checking contiguous ghost hits.

        Pages absent from the ghost FIFO follow the regular write-through path and
        enter the small FIFO after the host write completes.  A contiguous ghost
        run is promoted to the main FIFO only in complete ``main_page_size``
        groups.  Its shorter tail is not written through to DRAM and stays only in
        HBM, managed by the HBM LRU.
        """
        ghost_run = []

        def flush_ghost_run() -> None:
            main_count = (
                len(ghost_run) // self.main_pages_per_entry * self.main_pages_per_entry
            )
            for ghost_node in ghost_run:
                self._consume_ghost_hit(ghost_node)
            for ghost_node in ghost_run[:main_count]:
                self._contextide_main_after_write.add(ghost_node.id)
                self._ensure_write_through(ghost_node)
            ghost_run.clear()

        for node in nodes:
            if self._is_ghost_hit(node):
                ghost_run.append(node)
                continue
            flush_ghost_run()
            self._ensure_write_through(node)
        flush_ghost_run()

    def _evict_small_fifo_if_needed(self) -> None:
        blocked_pages = 0
        while len(self.small_fifo) > self.small_capacity_pages:
            node = self.small_fifo.pop_tail()
            if node is None:
                break
            if self._evict_host_node(node, ghost=True):
                blocked_pages = 0
                continue

            # Internal or referenced radix nodes cannot be physically removed yet.
            # Keep them tracked and inspect the next-oldest candidate instead.
            self.small_fifo.add_head(node)
            blocked_pages += 1
            if blocked_pages >= len(self.small_fifo):
                break

    def _evict_main_fifo_if_needed(self) -> None:
        blocked_pages = 0
        while len(self.main_fifo) > self.main_capacity_pages:
            node = self.main_fifo.pop_tail()
            if node is None:
                break
            freq = self.main_freq.get(node.id, 0)
            if freq > 0:
                self.main_freq[node.id] = freq - 1
                self.main_fifo.add_head(node)
                blocked_pages = 0
                continue
            if self._evict_main_host_node(node):
                blocked_pages = 0
                continue

            # Preserve metadata for radix nodes that cannot be removed yet.
            self.main_fifo.add_head(node)
            blocked_pages += 1
            if blocked_pages >= len(self.main_fifo):
                break

    def _add_page_node(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        priority: int,
        chunked: bool,
    ) -> TreeNode:
        child_key = key.child_key(self.page_size)
        new_node = TreeNode(priority=priority)
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        parent.children[child_key] = new_node
        self.evictable_size_ += len(value)
        self._update_leaf_status(parent)
        self._update_leaf_status(new_node)
        # A newly created page missed both HBM and DRAM. Compute its hash now so
        # insert() can classify the complete new suffix against the ghost FIFO.
        new_node.hash_value = compute_node_hash_values(new_node, self.page_size)
        self._record_store_event(new_node)
        if not chunked:
            new_node.hit_count += 1
        self._mark_hbm(new_node)
        return new_node

    def insert(self, params: InsertParams) -> InsertResult:
        key = params.key
        value = params.value
        chunked = params.chunked
        priority = params.priority or 0

        key, value = key.maybe_to_bigram_view(self.is_eagle, value)
        key = key.page_aligned(self.page_size)
        if value is not None:
            value = value[: len(key)]
        if len(key) == 0:
            return InsertResult(prefix_len=0)

        existing = self.match_prefix(MatchPrefixParams(key=key)).device_indices
        original_prefix_len = len(existing)

        node = self.root_node
        remaining_key = key
        remaining_value = value
        total_prefix_length = 0
        new_nodes = []
        while len(remaining_key) > 0:
            page_key = remaining_key[: self.page_size]
            page_value = remaining_value[: self.page_size]
            child_key = page_key.child_key(self.page_size)
            if child_key in node.children:
                child = node.children[child_key]
                prefix_len = child.key.match(page_key, page_size=self.page_size)
                if prefix_len != len(child.key):
                    child = self._split_node(child.key, child, prefix_len)
                child.priority = max(child.priority, priority)
                if child.evicted:
                    child.value = page_value.clone()
                    self.evictable_size_ += len(child.value)
                    self._update_leaf_status(child)
                    self._update_host_leaf_status(child)
                    self._update_leaf_status(child.parent)
                    self._mark_hbm(child)
                    self._inc_hit_count(child, chunked)
                else:
                    self._inc_hit_count(child, chunked)
                    self._mark_hbm(child)
                total_prefix_length += len(page_key)
                node = child
            else:
                node = self._add_page_node(
                    node, page_key, page_value, priority, chunked
                )
                new_nodes.append(node)
            remaining_key = remaining_key[self.page_size :]
            remaining_value = remaining_value[self.page_size :]

        self._admit_new_pages(new_nodes)
        return InsertResult(prefix_len=original_prefix_len)

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        self._remove_node_from_lists(child)
        new_node = super()._split_node(key, child, split_len)
        if new_node.value is not None:
            self._mark_hbm(new_node)
        if child.value is not None:
            self._mark_hbm(child)
        if new_node.host_value is not None:
            self._mark_small(new_node)
        if child.host_value is not None:
            self._mark_small(child)
        return new_node

    def dec_lock_ref(self, node: TreeNode, params=None):
        result = super().dec_lock_ref(node, params)
        cur = node
        while cur != self.root_node:
            if (
                self.node_tier.get(cur.id) == "main"
                and cur.backuped
                and cur.value is not None
                and cur.lock_ref == 0
            ):
                # Main FIFO pages are DRAM-resident after the active request.
                # Preserve HiRadixCache's contiguous device-prefix invariant:
                # descendants cannot remain in HBM below a DRAM-only main page.
                self._evict_hbm_subtree(cur)
            else:
                self._mark_hbm(cur)
            cur = cur.parent
        self._retry_pending_write_through()
        self._retry_pending_hbm_demotions()
        return result

    def _evict_backuped(self, node: TreeNode):
        # A DRAM-backed page can drop its HBM copy even when it is an internal
        # radix node. The host copy keeps the prefix available for future loads.
        self.hbm_lru.remove(node)
        num_evicted = super()._evict_backuped(node)
        if node.parent is not None:
            self._mark_hbm(node.parent)
        if self.node_tier.get(node.id) == "small":
            self._evict_small_fifo_if_needed()
        elif self.node_tier.get(node.id) == "main":
            self._evict_main_fifo_if_needed()
        return num_evicted

    def _evict_regular(self, node: TreeNode):
        self._remove_node_from_lists(node)
        num_evicted = super()._evict_regular(node)
        if node.parent is not None:
            self._mark_hbm(node.parent)
        return num_evicted

    def _insert_helper_host(
        self, node: TreeNode, key: RadixKey, host_value, hash_value
    ):
        node.last_access_time = time.monotonic()
        matched_length = 0
        remaining_key = key
        remaining_host_value = host_value
        remaining_hash_value = hash_value

        while len(remaining_key) > 0:
            page_key = remaining_key[: self.page_size]
            page_host_value = remaining_host_value[: self.page_size]
            page_hash_value = remaining_hash_value[:1]
            child_key = page_key.child_key(self.page_size)
            if child_key in node.children:
                child = node.children[child_key]
                child.last_access_time = time.monotonic()
                prefix_len = child.key.match(page_key, page_size=self.page_size)
                if prefix_len < len(child.key):
                    child = self._split_node(child.key, child, prefix_len)
                if not child.backuped:
                    child.host_value = page_host_value.clone()
                    child.hash_value = page_hash_value
                    self._update_host_leaf_status(child)
                    self._mark_small(child)
                matched_length += len(page_key)
                node = child
            else:
                new_node = TreeNode(priority=node.priority)
                new_node.parent = node
                new_node.key = page_key
                new_node.value = None
                new_node.host_value = page_host_value.clone()
                new_node.hash_value = page_hash_value
                node.children[child_key] = new_node
                self._update_host_leaf_status(new_node)
                self._update_leaf_status(node)
                self._update_host_leaf_status(node)
                self._mark_small(new_node)
                node = new_node
            remaining_key = remaining_key[self.page_size :]
            remaining_host_value = remaining_host_value[self.page_size :]
            remaining_hash_value = remaining_hash_value[1:]
        return matched_length

    def match_prefix(self, params: MatchPrefixParams):
        result = super().match_prefix(params)
        if result.device_indices.numel() > 0:
            node = result.last_device_node
            while node != self.root_node:
                self._mark_hbm(node)
                node = node.parent
        if result.host_hit_length > 0:
            self._touch_host_hit_chain(result.last_host_node)
        return result

    def init_load_back(self, params: InitLoadBackParams):
        values, node = super().init_load_back(params)
        if values is not None and values.numel() > 0:
            cur = node
            while cur != self.root_node and cur.value is not None:
                self._mark_hbm(cur)
                cur = cur.parent
        return values, node

    def _complete_write(self, backuped_node: TreeNode) -> None:
        self._record_store_event(backuped_node, medium=StorageMedium.CPU)
        self.dec_lock_ref(backuped_node)
        if backuped_node.id in self._contextide_main_after_write:
            self._contextide_main_after_write.discard(backuped_node.id)
            self._promote_to_main(backuped_node)
        else:
            self._mark_small(backuped_node)
        self._retry_pending_hbm_demotions()
        if self.enable_storage:
            self.write_backup_storage(backuped_node)

    def writing_check(self, write_back: bool = False):
        if len(self.ongoing_write_through) == 0:
            self._retry_pending_write_through()
            if len(self.ongoing_write_through) == 0:
                return

        finish_count = 0
        for _, finish_event, ack_list in self.cache_controller.ack_write_queue:
            if not finish_event.query():
                break
            finish_count += 1
        queue_size = torch.tensor(finish_count, dtype=torch.int, device="cpu")
        self._all_reduce_attn_groups(queue_size, torch.distributed.ReduceOp.MIN)
        finish_count = int(queue_size.item())

        while finish_count > 0:
            _, finish_event, ack_list = self.cache_controller.ack_write_queue.pop(0)
            finish_event.synchronize()
            for ack_id in ack_list:
                backuped_node = self.ongoing_write_through.pop(ack_id)
                self._complete_write(backuped_node)
            finish_count -= 1
        self._retry_pending_write_through()

    def _evict_hbm_subtree(self, node: TreeNode) -> tuple[int, bool]:
        """Evict an HBM subtree in leaf-first order without writing to DRAM.

        A DRAM-backed radix node may still have radix children after its HBM copy
        is released.  For HBM eviction purposes it is a leaf only after every HBM
        page below it has been released.  If any page in the subtree is locked,
        keep the ancestors resident so eviction never removes a non-leaf HBM page.
        """
        if node.lock_ref > 0:
            return 0, False

        num_evicted = 0
        for child in list(node.children.values()):
            child_evicted, child_subtree_evicted = self._evict_hbm_subtree(child)
            num_evicted += child_evicted
            if not child_subtree_evicted:
                return num_evicted, False

        if node.value is None:
            return num_evicted, True
        if node.backuped:
            return num_evicted + self._evict_backuped(node), True
        if len(node.children) == 0:
            return num_evicted + self._evict_regular(node), True

        # An unbacked radix node can be deleted only after all of its children
        # have been removed from the tree. Host-only children intentionally stay
        # in the radix tree, so keep this node resident instead of demoting it.
        return num_evicted, False

    def evict(self, params: EvictParams) -> EvictResult:
        start_time = time.perf_counter()
        num_tokens = params.num_tokens
        num_evicted = 0
        blocked_pages = 0
        while num_evicted < num_tokens and blocked_pages < len(self.hbm_lru):
            node = self.hbm_lru.pop_tail()
            if node is None:
                break
            if node.value is None:
                continue
            subtree_evicted, fully_evicted = self._evict_hbm_subtree(node)
            num_evicted += subtree_evicted
            if not fully_evicted:
                # Locked descendants and unbacked internal nodes cannot be
                # released safely. Keep the candidate in HBM and inspect another
                # LRU entry instead.
                self.hbm_lru.add_head(node)
                blocked_pages += 1
                continue
            blocked_pages = 0

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
