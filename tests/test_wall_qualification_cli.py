"""Qualification CLIs must require real devices without weakening numeric gates."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ('qualify_wall_dppo_batch.py', 'qualify_wall_dppo_prefix.py')


def command(script, tmp_path, *options):
    config = tmp_path/'model.json'
    config.write_text(json.dumps({'device':'cpu','dtype':'bfloat16','num_inference_timesteps':10,'flow_prefix_cache':False}))
    return [sys.executable,str(ROOT/'scripts'/script),'--source-root',str(ROOT),
            '--model-config',str(config),'--observation',str(tmp_path/'missing-observation.json'),
            '--output',str(tmp_path/'result'),'--batch-size','16',*options]


@pytest.mark.parametrize('script', SCRIPTS)
def test_help_documents_explicit_device_without_loading_model(script):
    done=subprocess.run([sys.executable,str(ROOT/'scripts'/script),'--help'],capture_output=True,text=True)
    assert done.returncode==0 and '--device' in done.stdout


@pytest.mark.parametrize('script', SCRIPTS)
def test_cpu_cannot_be_mislabeled_accelerator_qualification(tmp_path,script):
    done=subprocess.run(command(script,tmp_path,'--device','cpu'),capture_output=True,text=True,
                        env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=30)
    assert done.returncode==1
    report=json.loads((tmp_path/'result/report.json').read_text())
    assert report['passed'] is False and report['stage']=='failed'
    assert 'available CUDA or NPU accelerator' in report['error']
    assert 'missing-observation' not in report['error']


@pytest.mark.parametrize('option,value', [('--raw-atol','0.051'),('--reduced-atol','0.00051')])
def test_density_gates_cannot_be_relaxed(tmp_path,option,value):
    done=subprocess.run(command(SCRIPTS[0],tmp_path,option,value),capture_output=True,text=True)
    assert done.returncode==2 and 'never loosened' in done.stderr
    assert not (tmp_path/'result').exists()


@pytest.mark.parametrize('option,value', [('--gradient-relative-atol','0.0011'),('--gradient-rows','0'),('--gradient-rows','33')])
def test_prefix_gradient_gate_and_explicit_subsample_size_are_validated(tmp_path,option,value):
    done=subprocess.run(command(SCRIPTS[1],tmp_path,option,value),capture_output=True,text=True)
    assert done.returncode==2
    assert not (tmp_path/'result').exists()
