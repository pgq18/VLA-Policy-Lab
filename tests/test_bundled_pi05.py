import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from textwrap import dedent
from types import ModuleType

import pytest

from pressb import model_sources
from pressb.model_sources import pi05 as source_loader
from pressb.models import pi05_source_identity


PROJECT = Path(__file__).resolve().parents[1]
ROOT = model_sources.builtin_source_root("pi05")
POLICY_PATH = Path("lerobot/policies/pi05")
LEGACY_REVISION = "7e9d93342f3db35fe10d0636fedf4d7aeebb3d804eeaed2098bca8112e201a32"


def run_python(code):
    result = subprocess.run(
        [sys.executable, "-c", dedent(code)],
        cwd=PROJECT, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_pi05_manifest_records_bundled_sources_and_license():
    manifest = model_sources.source_manifest("pi05")
    lock = json.loads((PROJECT / "repos.lock.json").read_text())["repositories"]["LeRobot"]
    assert ROOT == PROJECT / lock["bundled_path"]
    listed_manifest = PROJECT / lock["bundled_manifest"]
    assert hashlib.sha256(listed_manifest.read_bytes()).hexdigest() == lock["bundled_manifest_sha256"]
    assert manifest["commit"] == lock["runtime_commit"]
    assert manifest["license"] == lock["license"] == "Apache-2.0"
    assert "LICENSE" in manifest["files"]
    assert "lerobot/policies/pi_gemma.py" in manifest["files"]
    for filename in ("__init__.py", "configuration_pi05.py", "modeling_pi05.py", "processor_pi05.py"):
        assert (POLICY_PATH / filename).as_posix() in manifest["files"]
    for name, record in manifest["files"].items():
        data = (ROOT / name).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]


def test_pi05_source_selection_is_lightweight_and_idempotent():
    run_python("""
        import sys
        from pressb.model_sources import activate_source, builtin_source_root

        assert activate_source("pi05") == builtin_source_root("pi05")
        first = tuple(sys.meta_path)
        activate_source("pi05")
        assert tuple(sys.meta_path) == first
        assert {"lerobot", "torch", "transformers"}.isdisjoint(sys.modules)
    """)


def test_pi05_policy_and_factories_share_bundled_classes_with_installed_datasets():
    for package in ("lerobot", "transformers", "torch", "datasets", "draccus"):
        if find_spec(package) is None:
            pytest.skip(f"π0.5 integration requires {package}")
    run_python("""
        import inspect
        from pathlib import Path
        from pressb.training.pi05 import policy_config
        from pressb.model_sources import builtin_source_root

        config, _ = policy_config({"device": "cpu"})
        import lerobot
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import get_policy_class, make_policy_config
        from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PiGemmaForCausalLM
        from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
        from lerobot.processor import ProcessorStepRegistry

        policy_dir = builtin_source_root("pi05") / "lerobot/policies/pi05"
        for obj in (PI05Config, PI05Policy, make_pi05_pre_post_processors):
            assert Path(inspect.getfile(obj)).resolve().parent == policy_dir
        assert Path(inspect.getfile(PiGemmaForCausalLM)).resolve() == policy_dir.parent / "pi_gemma.py"
        assert type(config) is type(make_policy_config("pi05", device="cpu")) is PI05Config
        assert get_policy_class("pi05") is PI05Policy
        tokenizer_step = ProcessorStepRegistry.get("pi05_prepare_state_tokenizer_processor_step")
        assert tokenizer_step is Pi05PrepareStateTokenizerProcessorStep
        runtime_dir = Path(lerobot.__file__).resolve().parent
        assert Path(inspect.getfile(LeRobotDataset)).resolve().is_relative_to(runtime_dir / "datasets")
        assert not runtime_dir.is_relative_to(builtin_source_root("pi05"))
    """)


@pytest.fixture
def isolated_finder(monkeypatch):
    monkeypatch.setattr(sys, "meta_path", sys.meta_path.copy())
    monkeypatch.setattr(source_loader, "_finder", None)
    for name in tuple(sys.modules):
        if name in ("lerobot.policies.pi05", "lerobot.policies.pi_gemma") or name.startswith("lerobot.policies.pi05."):
            monkeypatch.delitem(sys.modules, name)


@pytest.mark.parametrize("changed", [None, "pi05/configuration_pi05.py", "pi_gemma.py"])
def test_preloaded_pi05_modules_preserve_class_identity_only_for_matching_source(
        tmp_path, monkeypatch, isolated_finder, changed):
    shutil.copytree(ROOT / "lerobot", tmp_path / "lerobot", ignore=shutil.ignore_patterns("__pycache__"))
    upstream = tmp_path / POLICY_PATH
    package_name = "lerobot.policies.pi05"
    package = ModuleType(package_name)
    package.__file__ = str(upstream / "__init__.py")
    package.__path__ = [str(upstream)]
    monkeypatch.setitem(sys.modules, package_name, package)
    config_module = ModuleType(package_name + ".configuration_pi05")
    config_path = upstream / "configuration_pi05.py"
    config_module.__file__ = str(config_path)
    config_module.PI05Config = type("PI05Config", (), {})
    registered = config_module.PI05Config
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    gemma_module = ModuleType("lerobot.policies.pi_gemma")
    gemma_module.__file__ = str(upstream.parent / "pi_gemma.py")
    monkeypatch.setitem(sys.modules, gemma_module.__name__, gemma_module)

    if changed:
        with (upstream.parent / changed).open("a") as stream:
            stream.write("\n# locally edited\n")
        with pytest.raises(RuntimeError, match="another source tree"):
            model_sources.activate_source("pi05")
        assert package.__path__ == [str(upstream)]
    else:
        model_sources.activate_source("pi05")
        assert package.__path__ == [str(ROOT / POLICY_PATH)]
        assert sys.modules[config_module.__name__].PI05Config is registered
        with pytest.raises(RuntimeError, match="fresh process"):
            model_sources.activate_source("pi05", tmp_path)


def test_verified_pi05_relocation_matches_the_existing_checkpoint_revision(monkeypatch):
    source = pi05_source_identity._BUNDLED_SOURCE
    adapter = pi05_source_identity._BUNDLED_ADAPTER
    monkeypatch.setattr(pi05_source_identity, "source_tree_sha256", lambda family: source)
    revision = pi05_source_identity.inference_source_revision
    assert revision("0.4.5", adapter) == LEGACY_REVISION
    assert revision("0.4.6", adapter) != LEGACY_REVISION
    assert revision("0.4.5", hashlib.sha256(b"updated adapter").hexdigest()) != LEGACY_REVISION
    monkeypatch.setattr(pi05_source_identity, "source_tree_sha256", lambda family: "0" * 64)
    assert revision("0.4.5", adapter) != LEGACY_REVISION


def test_pi05_checkpoint_revision_tracks_live_source_and_adapter(tmp_path, monkeypatch):
    adapter = PROJECT / "src/pressb/models/pi05.py"
    adapter_digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
    revision = pi05_source_identity.inference_source_revision
    original = revision("0.4.5", adapter_digest)

    relocated = tmp_path / "pi05"
    shutil.copytree(ROOT, relocated, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(model_sources, "builtin_source_root", lambda family: relocated)
    assert revision("0.4.5", adapter_digest) == original
    changed_adapter = hashlib.sha256(adapter.read_bytes() + b"\n# local change\n").hexdigest()
    assert revision("0.4.5", changed_adapter) != original
    model = relocated / POLICY_PATH / "modeling_pi05.py"
    with model.open("a") as stream:
        stream.write("\n# local change\n")
    changed_source = revision("0.4.5", adapter_digest)
    assert changed_source != original
    assert changed_source != revision("0.4.5", changed_adapter)
    with (model.parent.parent / "pi_gemma.py").open("a") as stream:
        stream.write("\n# edited action expert\n")
    assert revision("0.4.5", adapter_digest) != changed_source
