from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sgl.kvcache.hiradix_cache import HiRadixTreeNode


class LinkedNode:
    def __init__(self, radix_node: HiRadixTreeNode | None = None) -> None:
        self.prev: LinkedNode | None = None
        self.next: LinkedNode | None = None
        self.radix_node = radix_node


class S3FIFOLinkedNode(LinkedNode):
    def __init__(
            self,
            radix_node: HiRadixTreeNode | None = None
    ) -> None:
        super().__init__(radix_node)
        self.freq = 0
