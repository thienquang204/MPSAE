"""Feature extraction survives interruption at array publication boundaries."""
import os

import numpy as np
import pytest

from mpsae.embed.archive import FeatureSink


@pytest.mark.parametrize("renamed", [0, 1, 2])
def test_recover_durable_full_sink_after_interrupted_publication(tmp_path, renamed):
    features = np.arange(12, dtype=np.float32).reshape(4, 3) + 1
    labels = np.arange(4, dtype=np.int64)
    with FeatureSink(tmp_path, rows=4, dim=3, identity={"extract": "fixed"}) as sink:
        sink.write(features, labels)
        partials, finals = sink._partial, sink._final
    for partial, final in list(zip(partials, finals))[:renamed]:
        os.replace(partial, final)
    with FeatureSink(tmp_path, rows=4, dim=3, identity={"extract": "fixed"}) as resumed:
        assert resumed.complete
        assert resumed.remaining == 0
        recovered_features, recovered_labels = resumed.publish()
        try:
            np.testing.assert_array_equal(recovered_features, features)
            np.testing.assert_array_equal(recovered_labels, labels)
        finally:
            recovered_features._mmap.close()
            recovered_labels._mmap.close()


def test_failed_sink_open_releases_lock_for_retry(tmp_path):
    sink = FeatureSink(tmp_path, rows=4, dim=3, identity={"extract": "fixed"})
    sink.directory.mkdir(parents=True)
    # Simulate failure after acquiring the lock, e.g. damaged durable metadata.
    sink._receipt_path.write_text("invalid JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        sink.__enter__()
    assert sink._lock is None
    sink._receipt_path.unlink()
    with FeatureSink(tmp_path, rows=4, dim=3, identity={"extract": "fixed"}) as retry:
        retry.write(np.ones((4, 3), dtype=np.float32), np.arange(4))
        values, labels = retry.publish()
        values._mmap.close()
        labels._mmap.close()
