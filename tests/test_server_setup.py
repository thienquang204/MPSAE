"""Run the native installer with fake hardware/package-manager boundaries.

These tests never install anything. They verify failure propagation and
environment/result preservation, not numerical compatibility on a real A100.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if os.name == "nt":
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    BASH = str(git_bash) if git_bash.exists() else None
pytestmark = pytest.mark.skipif(BASH is None, reason="Bash required for installer contracts")


def shell(program, *arguments, cwd=ROOT, env=None):
    return subprocess.run([BASH, "-c", program, "setup-test", *map(str, arguments)],
                          cwd=cwd, env=env, text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize("driver", ["560.28.03", "560.35.03", "570.86.15", "580.95.05", "600.1"])
def test_a100_stack_stays_on_pinned_cuda_for_newer_drivers(driver):
    result = shell('source scripts/setup_server.sh; select_a100_stack "$1"; '
                   'printf "%s|%s|%s|%s" "$CUDA_VERSION" "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCH_INDEX"', driver)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "12.6|2.9.1|0.24.1|https://download.pytorch.org/whl/cu126"


@pytest.mark.parametrize("driver", ["535.104.05", "560.1", "not-a-driver"])
def test_old_or_invalid_driver_fails_before_installing(driver):
    result = shell('source scripts/setup_server.sh; select_a100_stack "$1"; echo SHOULD_NOT_REACH', driver)
    assert result.returncode != 0
    assert "SHOULD_NOT_REACH" not in result.stdout


def _executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


@pytest.fixture
def installation(tmp_path):
    installer = tmp_path / "scripts" / "setup_server.sh"
    _executable(installer, (ROOT / "scripts/setup_server.sh").read_text(encoding="utf-8"))
    python = tmp_path / ".venv" / "bin" / "python"
    _executable(python, '''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_SETUP_CALLS"
if [[ "$*" == *"pip install --no-build-isolation -r requirements-server.txt"* && "${TEST_FAIL_INSTALL:-0}" == 1 ]]; then
    echo 'fixture FFCV build failed' >&2
    exit 17
fi
if [[ "$*" == '-m mpsae.server' && "${TEST_FAIL_VERIFY:-0}" == 1 ]]; then
    echo 'fixture GPU verification failed' >&2
    exit 23
fi
if [[ "$*" == '-m pip freeze' ]]; then printf 'fixture==1\n'; fi
''')
    cuda = tmp_path / "cuda"
    _executable(cuda / "bin" / "nvcc", '#!/usr/bin/env bash\necho "Cuda compilation tools, release 12.6, V12.6.85"\n')
    (tmp_path / ".env").write_text("HF_TOKEN=fixture-keep-existing\n")
    (tmp_path / ".env.example").write_text("HF_TOKEN=\n")
    checkpoint = tmp_path / "work" / "runs" / "existing.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserve source checkpoint")
    env = os.environ.copy()
    env.pop("MPSAE_TORCH_INDEX", None)
    env.update(MPSAE_CUDA_HOME=cuda.as_posix(), TEST_SETUP_CALLS=(tmp_path / "calls.txt").as_posix())
    harness = '''
source "$1"
ubuntu_repository() { echo ubuntu2204; }
uname() { if [[ "$1" == -s ]]; then echo Linux; else echo x86_64; fi; }
nvidia-smi() {
    if [[ "$1" == --query-gpu=driver_version ]]; then echo 580.95.05;
    else echo 'NVIDIA A100-SXM4-80GB, 8.0, 81920 MiB, 580.95.05'; fi
}
pkg-config() { return 0; }
apt-get() { echo 'UNEXPECTED_APT_CALL' >&2; exit 99; }
main "${@:2}"
'''
    return tmp_path, installer, python, env, harness


def test_dry_run_has_no_installation_side_effects(installation):
    root, installer, python, env, harness = installation
    result = shell(harness, installer.as_posix(), "--dry-run", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA Toolkit: 12.6" in result.stdout
    assert not (root / ".setup-logs").exists()
    assert not (root / "scripts/env.sh").exists()
    assert not (root / "calls.txt").exists()


@pytest.mark.parametrize("failure,expected_code", [("TEST_FAIL_INSTALL", 17), ("TEST_FAIL_VERIFY", 23)])
def test_dependency_or_gpu_failure_never_reports_setup_success(installation, failure, expected_code):
    root, installer, python, env, harness = installation
    env[failure] = "1"
    result = shell(harness, installer.as_posix(), "--skip-system-packages", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert "Setup complete." not in result.stdout
    assert not (root / ".venv/server-requirements.freeze.txt").exists()
    assert (root / "work/runs/existing.pt").read_bytes() == b"preserve source checkpoint"
    assert (root / ".env").read_text() == "HF_TOKEN=fixture-keep-existing\n"
    if failure == "TEST_FAIL_INSTALL":
        assert "-m mpsae.server" not in (root / "calls.txt").read_text()


def test_success_verifies_full_environment_and_keeps_existing_artifacts(installation):
    root, installer, python, env, harness = installation
    result = shell(harness, installer.as_posix(), "--skip-system-packages", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Setup complete." in result.stdout
    calls = (root / "calls.txt").read_text()
    assert "torch==2.9.1+cu126 torchvision==0.24.1+cu126" in calls
    assert "--no-build-isolation -r requirements-server.txt" in calls
    assert calls.index("-m pip check") < calls.index("-m mpsae.server") < calls.index("-m pip freeze")
    assert (root / "scripts/env.sh").is_file()
    assert (root / ".venv/server-requirements.freeze.txt").read_text().strip() == "fixture==1"
    assert (root / "work/runs/existing.pt").read_bytes() == b"preserve source checkpoint"
    assert (root / ".env").read_text() == "HF_TOKEN=fixture-keep-existing\n"
    assert "-m mpsae --config" not in calls  # Setup never launches a campaign.


def test_system_setup_installs_versioned_toolkit_without_driver_or_real_privileges(installation):
    root, installer, python, env, harness = installation
    (root / "cuda/bin/nvcc").unlink()
    env["TEST_SYSTEM_CALLS"] = (root / "system-calls.txt").as_posix()
    replacements = r'''
sudo() { printf 'sudo:%s\n' "$*" >> "$TEST_SYSTEM_CALLS"; if [[ "$1" != -v ]]; then "$@"; fi; }
env() { shift; "$@"; }
apt-get() {
    printf 'apt:%s\n' "$*" >> "$TEST_SYSTEM_CALLS"
    if [[ "$*" == *cuda-toolkit-12-6* ]]; then
        printf '#!/usr/bin/env bash\necho "release 12.6, fixture"\n' > "$MPSAE_CUDA_HOME/bin/nvcc"
        chmod +x "$MPSAE_CUDA_HOME/bin/nvcc"
    fi
}
curl() {
    printf 'curl:%s\n' "$*" >> "$TEST_SYSTEM_CALLS"
    while (( $# )); do
        if [[ "$1" == --output ]]; then printf 'fixture keyring' > "$2"; return; fi
        shift
    done
    return 1
}
dpkg() { printf 'dpkg:%s\n' "$*" >> "$TEST_SYSTEM_CALLS"; }
'''
    harness = harness.replace('main "${@:2}"', replacements + '\nmain "${@:2}"')
    result = shell(harness, installer.as_posix(), "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (root / "system-calls.txt").read_text()
    assert "libturbojpeg0-dev libopencv-dev" in calls
    assert "cuda-toolkit-12-6" in calls
    assert "ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" in calls
    assert "nvidia-driver" not in calls
    assert "Setup complete." in result.stdout


def test_shell_entrypoints_parse():
    for script in ("setup_server.sh", "activate_server.sh", "pipeline.sh"):
        result = subprocess.run([BASH, "-n", f"scripts/{script}"], cwd=ROOT,
                                text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr
