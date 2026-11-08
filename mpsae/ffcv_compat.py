"""Narrow Numba cleanup repair used before importing FFCV.

Numba's dispatcher finalizer walks OrderedDict.values(), which re-hashes its
signature keys. A signature whose hash changed can raise KeyError during that
walk, outside Numba's handler for already-removed native functions. Read the
underlying dict values and snapshot them before unregistering any function.
This affects disposal only; compilation, JPEG decoding and pixels are intact.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import gc
import inspect
import sys

REPAIR_VERSION = "numba-dict-values-cleanup-v1"


def _cleanup_factory(original, compiler_lock):
    @functools.wraps(original)
    def make_finalizer(dispatcher):
        old_callback = original(dispatcher)
        captured = inspect.getclosurevars(old_callback).nonlocals
        overloads = captured.get("overloads")
        target = captured.get("targetctx")
        defaults = old_callback.__defaults__ or ()
        if (not isinstance(overloads, dict) or target is None or len(defaults) != 1
                or not callable(defaults[0])):
            raise RuntimeError("Unsupported Numba dispatcher cleanup API; run bash setup_server.sh "
                               "to restore the project's pinned Numba/llvmlite stack")
        shutting_down = defaults[0]

        # Capture the mapping/context, never the dispatcher: retaining the
        # dispatcher here would prevent weakref.finalize from running at all.
        def cleanup(shutting_down=shutting_down):
            if shutting_down():
                return
            with compiler_lock:
                # dict.values bypasses OrderedDict's signature-key lookup.
                # Snapshot before removal can release other compiled objects.
                compiled = tuple(dict.values(overloads))
                for result in compiled:
                    try:
                        target.remove_user_function(result.entry_point)
                    except KeyError:
                        # Same policy as Numba: an entry may already be gone.
                        pass

        return cleanup

    make_finalizer._mpsae_cleanup_repair = REPAIR_VERSION
    return make_finalizer


def prepare_ffcv() -> str:
    """Install once, before FFCV creates its temporary CPU JIT dispatchers.

    Keep site-packages untouched. A new process automatically uses this fix
    when it calls the project's FFCV writer, reader or native verification.
    """
    from numba.core.compiler_lock import global_compiler_lock
    from numba.core.dispatcher import _DispatcherBase

    with global_compiler_lock:
        original = _DispatcherBase._make_finalizer
        if getattr(original, "_mpsae_cleanup_repair", None) != REPAIR_VERSION:
            _DispatcherBase._make_finalizer = _cleanup_factory(original, global_compiler_lock)
    return REPAIR_VERSION


@contextmanager
def verify_finalizers():
    """Make ignored cleanup exceptions fail native verification visibly."""
    previous = sys.unraisablehook
    failures = []

    def record(event):
        failures.append(f"{event.exc_type.__name__}: {event.exc_value}")
        previous(event)

    sys.unraisablehook = record
    try:
        yield
    finally:
        try:
            gc.collect()
        finally:
            sys.unraisablehook = previous
    if failures:
        raise RuntimeError("FFCV cleanup verification failed: " + "; ".join(failures))
