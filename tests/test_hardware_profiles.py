"""Architecture validation runs without Torch or physical GPU access."""
import pytest

from mpsae.server import validate_cuda_runtime


@pytest.mark.parametrize("capability,cuda", [
    ((8, 0), "12.6"), ((8, 6), "12.6"), ((8, 9), "12.6"),
    ((9, 0), "12.6"), ((12, 0), "12.8"), ((10, 3), "12.9"),
    ((8, 0), "12.9"), ((10, 3), "13.0"),
])
def test_supported_gpu_runtime_pairs(capability, cuda):
    validate_cuda_runtime(capability, cuda)


@pytest.mark.parametrize("capability,cuda", [
    ((12, 0), "12.6"), ((10, 3), "12.8"), ((99, 0), "12.9"), ((8, 0), None),
])
def test_invalid_or_too_old_runtime_is_rejected(capability, cuda):
    with pytest.raises(RuntimeError):
        validate_cuda_runtime(capability, cuda)


def test_active_environment_must_match_selected_installation():
    with pytest.raises(RuntimeError, match="Selected profile expects CUDA 12.9"):
        validate_cuda_runtime((8, 0), "12.6", expected="12.9")
