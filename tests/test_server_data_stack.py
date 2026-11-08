"""Check early data dependency diagnostics without requiring CUDA."""
import builtins

import pytest

from mpsae.server import verify_data_stack


def test_arrow_binary_import_failure_has_repair_command_before_gpu_import(monkeypatch):
    original_import = builtins.__import__

    def incompatible_arrow(name, *args, **kwargs):
        if name == "pyarrow":
            raise ImportError("pyarrow requires NumPy 2.0 or newer, found 1.26.4")
        if name in ("torch", "cupy", "ffcv"):
            pytest.fail("data preflight must not require GPU imports")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", incompatible_arrow)
    with pytest.raises(RuntimeError, match="requirements-server.txt") as caught:
        verify_data_stack()
    assert isinstance(caught.value.__cause__, ImportError)
    assert "PyArrow 25.0.1" in str(caught.value)


def test_installed_data_stack_round_trip():
    pytest.importorskip("pyarrow", exc_type=ModuleNotFoundError)
    pytest.importorskip("datasets", exc_type=ModuleNotFoundError)
    versions = verify_data_stack()
    assert set(versions) == {"numpy", "pyarrow", "datasets"}
