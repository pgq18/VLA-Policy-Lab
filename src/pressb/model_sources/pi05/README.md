# LeRobot π0.5

The `lerobot/policies/pi05/` directory contains the complete upstream π0.5 policy package, including its configuration, PyTorch model and processors. Its PiGemma action expert and PaliGemma wrapper live in [pi_gemma.py](lerobot/policies/pi_gemma.py). The source manifest records the LeRobot commit, archive digest and file hashes. The Apache 2.0 license and upstream notices are included in [LICENSE](LICENSE).

PressB's inference adapter and supervised training integration live in [models/pi05.py](../../models/pi05.py) and [training/pi05.py](../../training/pi05.py). Install their shared LeRobot dependencies with the [π0.5 setup instructions](../../../../docs/pi05.md).

For direct use, select the source before importing LeRobot policies. Use the canonical `lerobot.policies.pi05` name so the LeRobot config and processor registries share one set of classes.

```python
from pressb.model_sources import activate_source

activate_source("pi05")
from lerobot.policies.pi05 import PI05Config, PI05Policy
```

The selector in `__init__.py` routes `lerobot.policies.pi05` and `lerobot.policies.pi_gemma` to this directory. LeRobot supplies the shared dataset, configuration and processor APIs. A process keeps its first selected source tree. If LeRobot has already imported either module, PressB checks the loaded files against the selected tree and reuses byte-identical modules.
