"""Reproduce the OrderedDict cleanup failure without GPU/FFCV dependencies."""
from collections import OrderedDict
import gc
from threading import RLock
from types import SimpleNamespace
import weakref

import pytest

from mpsae.ffcv_compat import _cleanup_factory, verify_finalizers


def _upstream_factory(self):
    overloads, targetctx = self.overloads, self.targetctx

    def finalizer(shutting_down=lambda: False):
        if shutting_down():
            return
        for result in overloads.values():
            try:
                targetctx.remove_user_function(result.entry_point)
            except KeyError:
                pass

    return finalizer


class Dispatcher:
    def __init__(self, overloads, remove):
        self.overloads = overloads
        self.targetctx = SimpleNamespace(remove_user_function=remove)


def test_mutated_signature_hash_reproduces_error_and_cleanup_still_releases_code():
    class SignatureType:
        value = 1

        def __hash__(self):
            return self.value

    key = SignatureType()
    entries = OrderedDict({(key,): SimpleNamespace(entry_point="jpeg")})
    removed = []
    dispatcher = Dispatcher(entries, removed.append)
    key.value = 2
    with pytest.raises(KeyError):
        _upstream_factory(dispatcher)()
    _cleanup_factory(_upstream_factory, RLock())(dispatcher)()
    assert removed == ["jpeg"]


def test_cleanup_snapshots_results_before_unregistration_changes_mapping():
    entries = OrderedDict((index, SimpleNamespace(entry_point=index)) for index in range(3))
    removed = []

    def remove(entry):
        removed.append(entry)
        entries.clear()

    _cleanup_factory(_upstream_factory, RLock())(Dispatcher(entries, remove))()
    assert removed == [0, 1, 2]


def test_cleanup_does_not_keep_dispatcher_alive():
    removed = []
    dispatcher = Dispatcher(OrderedDict({0: SimpleNamespace(entry_point="jpeg")}), removed.append)
    reference = weakref.ref(dispatcher)
    callback = _cleanup_factory(_upstream_factory, RLock())(dispatcher)
    finalize = weakref.finalize(dispatcher, callback)
    del dispatcher
    gc.collect()
    assert reference() is None
    assert not finalize.alive
    assert removed == ["jpeg"]


def test_shutdown_skips_cleanup_and_unrelated_cleanup_errors_are_not_hidden():
    def fail(entry):
        raise ValueError("native cleanup failed")

    dispatcher = Dispatcher(OrderedDict({0: SimpleNamespace(entry_point="jpeg")}), fail)
    callback = _cleanup_factory(_upstream_factory, RLock())(dispatcher)
    callback(shutting_down=lambda: True)
    with pytest.raises(ValueError, match="native cleanup failed"):
        callback()


def test_already_removed_entry_uses_numba_original_policy():
    def missing(entry):
        raise KeyError(entry)

    dispatcher = Dispatcher(OrderedDict({0: SimpleNamespace(entry_point="jpeg")}), missing)
    _cleanup_factory(_upstream_factory, RLock())(dispatcher)()


def test_unknown_private_api_is_rejected():
    with pytest.raises(RuntimeError, match="pinned Numba"):
        _cleanup_factory(lambda self: lambda: None, RLock())(object())


def test_verifier_fails_on_unraisable_cleanup_and_restores_hook(monkeypatch):
    import sys

    forwarded = []
    monkeypatch.setattr(sys, "unraisablehook", forwarded.append)
    with pytest.raises(RuntimeError, match="FFCV cleanup verification failed"):
        with verify_finalizers():
            sys.unraisablehook(SimpleNamespace(exc_type=KeyError, exc_value=KeyError("jpeg")))
    assert sys.unraisablehook == forwarded.append
    assert len(forwarded) == 1


def test_real_numba_compiled_dispatcher_can_be_collected():
    numba = pytest.importorskip("numba")
    from mpsae.ffcv_compat import prepare_ffcv

    prepare_ffcv()
    with verify_finalizers():
        compiled = numba.njit(lambda x: x + 1)
        assert compiled(4) == 5
        reference = weakref.ref(compiled)
        del compiled
        gc.collect()
        assert reference() is None
