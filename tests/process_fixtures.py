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


class SlowFamilyRenderer(SlowReadyRenderer):
    """Expose a cancellable start, followed by a noninterruptible operation."""

    @staticmethod
    def _pause(on_batch):
        on_batch(1, 2)
        time.sleep(2)

    def prediction_fragments(self, dna, on_batch):
        self._pause(on_batch)
        return super().prediction_fragments(dna, on_batch)

    def explanation_fragment(self, dna, site_key, head, method, params, size, on_batch):
        self._pause(on_batch)
        return super().explanation_fragment(dna, site_key, head, method, params, size, on_batch)

    def csv_table(self, dna, on_batch, filters=None):
        self._pause(on_batch)
        return super().csv_table(dna, on_batch, filters)


class HangsRenderer(FakeRenderer):
    """#218: a computation that reports once, then never answers again."""

    def explanation_fragment(self, dna, site_key, head, method, params, size, on_batch):
        on_batch(1, 10)
        time.sleep(3600)


class SteadyRenderer(FakeRenderer):
    """#219: a long computation that keeps reporting progress."""

    def explanation_fragment(self, dna, site_key, head, method, params, size, on_batch):
        for done in range(1, 11):
            time.sleep(0.2)
            on_batch(done, 10)
        return super().explanation_fragment(dna, site_key, head, method, params, size,
                                            lambda *_: None)


class HangsOnceWhileLoadingRenderer(FakeRenderer):
    """#219: the first child's model load never ends; later children load normally.

    The marker path comes from the environment the spawned child inherits.
    """

    def warm_up(self, on_batch):
        marker = os.environ["DASHBOARD_TEST_LOAD_MARKER"]
        if not os.path.exists(marker):
            open(marker, "w").close()
            time.sleep(3600)
        return super().warm_up(on_batch)
