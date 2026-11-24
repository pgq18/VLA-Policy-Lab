"""YAML and JSON recipes share inheritance, expansion and resolved node files."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest

from pressb.experiments.cli import main
from pressb.experiments.config import load_experiment, PROJECT_ROOT


def test_elevator_examples_work_without_exports_and_use_the_training_outputs(monkeypatch, capsys):
    for key in ("PRESSB_QWEN_ROOT", "PRESSB_VJEPA_ROOT", "PRESSB_SFT_CHECKPOINT",
                "PRESSB_DSRL_CHECKPOINT", "PRESSB_DATASET_ROOT", "PRESSB_PRETRAINED_CHECKPOINT",
                "PRESSB_RUN_ROOT", "ELEVATOR_RUN", "RL_CHECKPOINT", "CFG", "METHOD"):
        monkeypatch.delenv(key, raising=False)
    recipes = PROJECT_ROOT / "examples/elevator"
    runs = {}
    for path in sorted(recipes.glob("*.yaml")):
        if path.name == "sft.yaml":
            continue
        assert main(["validate", "--config", str(path)]) == 0
        nodes = json.loads(capsys.readouterr().out)["nodes"]
        runs[path.stem] = nodes["trainer"]
        assert nodes["simulation"]["backend"]["config"]["num_envs"] == 1
        assert nodes["inference"]["backend"]["config"]["checkpoint"] == str(
            PROJECT_ROOT / "outputs/elevator/sft/pose9/checkpoints/best")
    dsrl_status = str(Path(runs["dsrl"]["output_dir"]) / "status.json")
    assert runs["train"] == runs["dsrl"]
    assert runs["eval"]["checkpoint"]["resume"] == dsrl_status
    assert runs["residual_after_dsrl"]["frozen_dsrl"]["checkpoint"] == dsrl_status
    assert runs["eval"]["learner"] == runs["train"]["learner"]
    assert runs["eval"]["seed"] == runs["base"]["seed"]
    assert len({cfg["output_dir"] for key, cfg in runs.items() if key != "train"}) == len(runs) - 1


def test_mixed_format_inheritance_and_prepare(tmp_path, monkeypatch, capsys):
    parents = tmp_path / "parents"
    parents.mkdir()
    base = parents / "base.yaml"
    base.write_text(dedent("""\
        schema_version: 1
        name: base
        task:
          plugin: pap
          config:
            scene: ${experiment_dir}/scene.json
            seed_list: [1, 2]
        model: {plugin: walloss}
        run:
          method: base
          mode: eval
          output_dir: ${experiment_dir}/$runs/${experiment_name}
    """))
    middle = tmp_path / "middle.json"
    middle.write_text(json.dumps({
        "extends": "parents/base.yaml", "task": {"config": {"seed_list": [3]}},
    }))
    recipe = tmp_path / "run.yml"
    recipe.write_text(dedent("""\
        extends: middle.json
        name: mixed-${env:PRESSB_TEST_SUFFIX}
        model:
          config:
            checkpoint: ${env:PRESSB_TEST_CHECKPOINT}
        collection:
          directory: ${experiment_dir}/trajectories
          include_images: false
    """))
    checkpoint = str(tmp_path / "${experiment_name}" / "${env:PRESSB_TEST_SUFFIX}" / "weights")
    monkeypatch.setenv("PRESSB_TEST_CHECKPOINT", checkpoint)
    monkeypatch.setenv("PRESSB_TEST_SUFFIX", "eval")
    assert load_experiment(middle)["nodes"]["trainer"]["output_dir"] == str(parents / "$runs/base")
    result = load_experiment(recipe)
    assert result["name"] == "mixed-eval"
    assert result["nodes"]["trainer"]["output_dir"] == str(parents / "$runs/mixed-eval")
    assert result["collection"] == {"directory": str(tmp_path / "trajectories"), "include_images": False}
    assert result["nodes"]["inference"]["backend"]["config"] == {"checkpoint": checkpoint}
    assert result["nodes"]["simulation"]["backend"]["config"] == {
        "scene": str(parents / "scene.json"), "seed_list": [3],
    }
    assert result["sources"] == [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (base, middle, recipe)
    ]

    assert main(["validate", "--config", str(recipe)]) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == result["nodes"]
    resolved = tmp_path / "resolved"
    assert main(["prepare", "--config", str(recipe), "--resolved-dir", str(resolved)]) == 0
    for node, config in result["nodes"].items():
        assert json.loads((resolved / f"{node}.json").read_text()) == config
    assert json.loads((resolved / "manifest.json").read_text())["sources"] == result["sources"]


@pytest.mark.parametrize(("content", "message"), [
    ("# empty\n", "root must be a mapping"),
    ("[a, b]", "root must be a mapping"),
    ("[unterminated", "Invalid experiment YAML"),
    ("name: !custom value", "Invalid experiment YAML"),
    ("run: {output_dir: 2025-09-01}", "JSON-compatible YAML values"),
    ("task: {config: {3: value}}", "mapping keys must be strings"),
    ("rnu: {}", "Unknown experiment sections.*'rnu'"),
    ("extends: [one.yaml, two.yaml]", "one parent JSON or YAML"),
    ("name: ${missing}", "Unresolved experiment variable"),
    ("name: run-${experiment_name}", "name defined independently"),
])
def test_invalid_yaml_identifies_source(tmp_path, content, message):
    path = tmp_path / "broken.yml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message) as err:
        load_experiment(path)
    assert str(path) in str(err.value)


def test_cross_format_cycle_identifies_both_sources(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.json"
    first.write_text("extends: second.json")
    second.write_text(json.dumps({"extends": "first.yml"}))
    with pytest.raises(ValueError, match="inheritance cycle") as err:
        load_experiment(first)
    assert f"{first} -> {second} -> {first}" in str(err.value)


def test_json_only_load_keeps_yaml_lazy(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1, "name": "test", "task": {"plugin": "pap"}, "model": {"plugin": "walloss"},
    }))
    script = dedent("""\
        import sys
        from pressb.experiments.config import load_experiment
        load_experiment(sys.argv[1])
        assert "yaml" not in sys.modules
    """)
    subprocess.run([sys.executable, "-c", script, str(recipe)], check=True,
                   env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
