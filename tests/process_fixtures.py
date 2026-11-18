"""Spawn-importable renderers used only by process boundary tests."""

import os
import time

from desktop.fake import FakeRenderer


class ControlledRenderer(FakeRenderer):
    def sites_fragment(self, dna, on_batch):
        if dna == "crash":
            os._exit(37)
        return super().sites_fragment(dna, on_batch)


class FailsOnceRenderer(FakeRenderer):
    def __init__(self):
        super().__init__(step_delay=0)
        self.attempts = 0

    def warm_up(self, on_batch):
        self.attempts += 1
        if self.attempts == 1:
            raise ValueError("<retry preparation>")
        return super().warm_up(on_batch)


class SlowReadyRenderer(FakeRenderer):
    def warm_up(self, on_batch):
        on_batch(1, 2)
        time.sleep(2)
        on_batch(2, 2)
        return '<p class="status">old ready</p>'
