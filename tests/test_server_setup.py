"""Run the native installer with fake hardware/package-manager boundaries.

These tests never install anything. They verify failure propagation and
environment/result preservation, not numerical compatibility on real GPUs.
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


@pytest.mark.parametrize("driver", ["560.35.05", "570.86.15", "580.95.05", "600.1"])
def test_ampere_stack_stays_on_pinned_cuda_for_newer_drivers(driver):
    result = shell('source setup_server.sh; select_cuda_stack "$1" 8.0; '
                   'printf "%s|%s|%s|%s" "$CUDA_VERSION" "$TORCH_VERSION" "$TORCHVISION_VERSION" "$TORCH_INDEX"', driver)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "12.6|2.9.1|0.24.1|https://download.pytorch.org/whl/cu126"


@pytest.mark.parametrize("driver", ["535.104.05", "560.1", "not-a-driver"])
def test_old_or_invalid_driver_fails_before_installing(driver):
    result = shell('source setup_server.sh; select_cuda_stack "$1" 8.0; echo SHOULD_NOT_REACH', driver)
    assert result.returncode != 0
    assert "SHOULD_NOT_REACH" not in result.stdout


@pytest.mark.parametrize("gpu,capabilities,cuda", [
    ("A100", "8.0", "12.6"), ("RTX A6000", "8.6", "12.6"),
    ("RTX 4090", "8.9", "12.6"), ("H100", "9.0", "12.6"),
    ("H200", "9.0", "12.6"), ("RTX 5090", "12.0", "12.8"),
    ("GB300", "10.3", "12.9"), ("mixed", "8.0;12.0;10.3;8.0", "12.9"),
])
def test_gpu_selection_uses_capability_including_blackwell_and_mixed_hosts(gpu, capabilities, cuda):
    result = shell('source setup_server.sh; select_cuda_stack 580.95.05 "$1"; '
                   'printf "%s|%s" "$CUDA_VERSION" "$TORCH_ARCHES"', capabilities)
    assert result.returncode == 0, (gpu, result.stderr)
    assert result.stdout.split("|")[0] == cuda
    assert len(result.stdout.split("|")[1].split(";")) == len(set(capabilities.split(";")))


@pytest.mark.parametrize("driver,capability,override", [
    ("560.35.05", "12.0", "auto"), ("570.124.06", "10.3", "auto"),
    ("580.95.05", "12.0", "12.6"), ("580.95.05", "10.3", "12.8"),
    ("580.95.05", "99.0", "auto"), ("580.95.05", "8.0", "invalid"),
])
def test_incompatible_driver_override_or_unknown_gpu_is_rejected(driver, capability, override):
    result = shell('source setup_server.sh; select_cuda_stack "$1" "$2" "$3"; echo SHOULD_NOT_REACH',
                   driver, capability, override)
    assert result.returncode != 0
    assert "SHOULD_NOT_REACH" not in result.stdout


def test_newer_explicit_profile_is_allowed_for_older_gpu():
    result = shell('source setup_server.sh; select_cuda_stack 580.95.05 8.0 12.9; echo "$CUDA_VERSION"')
    assert result.returncode == 0 and result.stdout.strip() == "12.9"


def _executable(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)


@pytest.fixture
def installation(tmp_path):
    installer = tmp_path / "setup_server.sh"
    _executable(installer, (ROOT / "setup_server.sh").read_text(encoding="utf-8"))
    python = tmp_path / ".venv" / "bin" / "python"
    _executable(python, '''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$TEST_SETUP_CALLS"
if [[ "$*" == *"Missing {header}"* && "${TEST_FAIL_HEADERS:-0}" == 1 ]]; then
    echo 'fixture matching Python.h missing' >&2
    exit 29
fi
if [[ "$*" == *"pip install --no-build-isolation -r requirements-server.txt"* && "${TEST_FAIL_INSTALL:-0}" == 1 ]]; then
    echo 'fixture FFCV build failed' >&2
    exit 17
fi
if [[ "$*" == '-m mpsae.server'* && "${TEST_FAIL_VERIFY:-0}" == 1 ]]; then
    echo 'fixture GPU verification failed' >&2
    exit 23
fi
if [[ "$*" == '-m pip freeze' ]]; then printf 'fixture==1\n'; fi
''')
    cuda = tmp_path / "cuda"
    _executable(cuda / "bin" / "nvcc", '#!/usr/bin/env bash\necho "Cuda compilation tools, release ${TEST_CUDA_VERSION:-12.6}, fixture"\n')
    (tmp_path / ".env").write_text("HF_TOKEN=fixture-keep-existing\n")
    (tmp_path / ".env.example").write_text("HF_TOKEN=\n")
    checkpoint = tmp_path / "work" / "runs" / "existing.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"preserve source checkpoint")
    env = os.environ.copy()
    env.pop("MPSAE_TORCH_INDEX", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(MPSAE_CUDA_HOME=cuda.as_posix(), TEST_SETUP_CALLS=(tmp_path / "calls.txt").as_posix())
    harness = '''
source "$1"
ubuntu_repository() { echo "${TEST_UBUNTU_REPOSITORY:-ubuntu2204}"; }
uname() { if [[ "$1" == -s ]]; then echo Linux; else echo "${TEST_CPU_ARCH:-x86_64}"; fi; }
nvidia-smi() {
    printf '%s\n' "${TEST_GPU_INFO:-NVIDIA A100-SXM4-80GB, 8.0, 81920 MiB, 580.95.05}"
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
    assert not (root / "env.sh").exists()
    assert not (root / "calls.txt").exists()


@pytest.mark.parametrize("failure,expected_code", [("TEST_FAIL_INSTALL", 17), ("TEST_FAIL_VERIFY", 23),
                                                    ("TEST_FAIL_HEADERS", 29)])
def test_dependency_or_gpu_failure_never_reports_setup_success(installation, failure, expected_code):
    root, installer, python, env, harness = installation
    env[failure] = "1"
    result = shell(harness, installer.as_posix(), "--skip-system-packages", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert "Setup complete." not in result.stdout
    assert not (root / ".venv/server-requirements.freeze.txt").exists()
    assert (root / "work/runs/existing.pt").read_bytes() == b"preserve source checkpoint"
    assert (root / ".env").read_text() == "HF_TOKEN=fixture-keep-existing\n"
    if failure in ("TEST_FAIL_INSTALL", "TEST_FAIL_HEADERS"):
        assert "-m mpsae.server" not in (root / "calls.txt").read_text()
    if failure == "TEST_FAIL_HEADERS":
        assert "-m pip install" not in (root / "calls.txt").read_text()


def test_success_verifies_full_environment_and_keeps_existing_artifacts(installation):
    root, installer, python, env, harness = installation
    result = shell(harness, installer.as_posix(), "--skip-system-packages", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Setup complete." in result.stdout
    calls = (root / "calls.txt").read_text()
    assert "torch==2.9.1+cu126 torchvision==0.24.1+cu126" in calls
    assert "--no-build-isolation -r requirements-server.txt" in calls
    assert calls.index("-m pip install -c constraints/server.txt numpy") < calls.index("torch==2.9.1+cu126")
    assert calls.index("-m pip check") < calls.index("-m mpsae.server") < calls.index("-m pip freeze")
    assert (root / "env.sh").is_file()
    assert (root / ".venv/server-requirements.freeze.txt").read_text().strip() == "fixture==1"
    assert (root / "work/runs/existing.pt").read_bytes() == b"preserve source checkpoint"
    assert (root / ".env").read_text() == "HF_TOKEN=fixture-keep-existing\n"
    assert "-m mpsae --config" not in calls  # Setup never launches a campaign.


@pytest.mark.parametrize("name,capability,cpu,cuda,vision", [
    ("RTX 5090", "12.0", "x86_64", "12.8", "torchvision==0.24.1+cu128"),
    ("GB300", "10.3", "aarch64", "12.9", "torchvision==0.24.1"),
])
def test_installer_selects_correct_wheels_and_compiler_architecture(installation, name, capability, cpu, cuda, vision):
    root, installer, python, env, harness = installation
    env.update(TEST_GPU_INFO=f"NVIDIA {name}, {capability}, 81920 MiB, 580.95.05",
               TEST_CPU_ARCH=cpu, TEST_CUDA_VERSION=cuda)
    result = shell(harness, installer.as_posix(), "--skip-system-packages", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (root / "calls.txt").read_text()
    assert f"torch==2.9.1+cu{cuda.replace('.', '')} {vision}" in calls
    generated = (root / "env.sh").read_text()
    assert f"TORCH_CUDA_ARCH_LIST={capability}" in generated
    assert f"MPSAE_CUDA_VERSION={cuda}" in generated
    if cpu == "aarch64":
        assert "NVIDIA repository: sbsa" in result.stdout
        assert "--force-reinstall --no-deps" in calls


@pytest.mark.parametrize("cpu,gpu,capability,cuda,repo_arch,distro,glib", [
    ("x86_64", "A100", "8.0", "12.6", "x86_64", "ubuntu2204", "libglib2.0-0"),
    ("aarch64", "GB300", "10.3", "12.9", "sbsa", "ubuntu2404", "libglib2.0-0t64"),
])
def test_system_setup_installs_versioned_toolkit_without_driver_or_real_privileges(
        installation, cpu, gpu, capability, cuda, repo_arch, distro, glib):
    root, installer, python, env, harness = installation
    (root / "cuda/bin/nvcc").unlink()
    env["TEST_SYSTEM_CALLS"] = (root / "system-calls.txt").as_posix()
    env.update(TEST_CPU_ARCH=cpu, TEST_CUDA_VERSION=cuda, TEST_UBUNTU_REPOSITORY=distro,
               TEST_GPU_INFO=f"NVIDIA {gpu}, {capability}, 81920 MiB, 580.95.05")
    replacements = r'''
sudo() { printf 'sudo:%s\n' "$*" >> "$TEST_SYSTEM_CALLS"; if [[ "$1" != -v ]]; then "$@"; fi; }
env() { shift; "$@"; }
apt-get() {
    printf 'apt:%s\n' "$*" >> "$TEST_SYSTEM_CALLS"
    if [[ "$*" == *cuda-toolkit-* ]]; then
        printf '#!/usr/bin/env bash\necho "release %s, fixture"\n' "${TEST_CUDA_VERSION:-12.6}" > "$MPSAE_CUDA_HOME/bin/nvcc"
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
    assert f"cuda-toolkit-{cuda.replace('.', '-')}" in calls
    assert f"{distro}/{repo_arch}/cuda-keyring_1.1-1_all.deb" in calls
    assert f"libgl1 {glib}" in calls
    assert "nvidia-driver" not in calls
    assert "Setup complete." in result.stdout


def test_shell_entrypoints_parse():
    for script in ("setup_server.sh", "activate_server.sh", "pipeline.sh", "all_in_one.sh"):
        result = subprocess.run([BASH, "-n", f"{script}"], cwd=ROOT,
                                text=True, capture_output=True, timeout=10)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("options,command,install", [
    ([], "run", True), (["--skip-setup"], "run", False),
    (["--sync-only"], "sync", False), (["--check"], "check", False),
    (["--smoke"], "run --smoke --config configs/smoke.yaml", True),
    (["--cuda", "12.8", "--skip-system-packages"], "run", True),
])
def test_all_in_one_uses_native_python_and_expected_setup_mode(installation, options, command, install):
    root, installer, python, env, harness = installation
    script = root / "all_in_one.sh"
    _executable(script, (ROOT / "all_in_one.sh").read_text())
    _executable(installer, '#!/usr/bin/env bash\necho "SETUP:$*" >> "$TEST_SETUP_CALLS"\n')
    _executable(root / "activate_server.sh", '#!/usr/bin/env bash\nexport ACTIVATED_FIXTURE=1\n')
    result = subprocess.run([BASH, str(script), *options], cwd=root, env=env,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = (root / "calls.txt").read_text()
    assert ("SETUP" in calls) is install
    assert f"-m mpsae.remote {command}" in calls
    assert "docker" not in calls
    if "--cuda" in options:
        assert "SETUP:--cuda 12.8 --skip-system-packages" in calls


def test_all_in_one_rejects_ignored_install_options(installation):
    root, installer, python, env, harness = installation
    script = root / "all_in_one.sh"
    _executable(script, (ROOT / "all_in_one.sh").read_text())
    result = subprocess.run([BASH, str(script), "--skip-setup", "--cuda", "12.8"],
                            cwd=root, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2
    assert "Installer options cannot be combined" in result.stderr
    assert not (root / "calls.txt").exists()


@pytest.mark.parametrize("mode", ["--skip-setup", "--sync-only", "--check"])
def test_all_in_one_rejects_setup_only_when_setup_is_disabled(installation, mode):
    root, installer, python, env, harness = installation
    script = root / "all_in_one.sh"
    _executable(script, (ROOT / "all_in_one.sh").read_text())
    result = subprocess.run([BASH, str(script), "--setup-only", mode],
                            cwd=root, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 2
    assert "--setup-only cannot be combined" in result.stderr
    assert not (root / "calls.txt").exists()


def test_selected_gpu_mask_is_forwarded_to_nvidia_smi(installation):
    root, installer, python, env, harness = installation
    env["CUDA_VISIBLE_DEVICES"] = "GPU-fixture-uuid"
    env["TEST_GPU_QUERY"] = (root / "gpu-query.txt").as_posix()
    harness = harness.replace('nvidia-smi() {', 'nvidia-smi() {\nprintf "%s\\n" "$*" > "$TEST_GPU_QUERY"')
    result = shell(harness, installer.as_posix(), "--dry-run", "--python", python.as_posix(), cwd=root, env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--id=GPU-fixture-uuid" in (root / "gpu-query.txt").read_text()


def test_flexible_gpu_setup_leaves_scripts_at_repo_root():
    for name in ("setup_server.sh", "activate_server.sh", "all_in_one.sh", "pipeline.sh", "capture_golden.py"):
        assert (ROOT / name).is_file()
        assert not (ROOT / "scripts" / name).exists()
