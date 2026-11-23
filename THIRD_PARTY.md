# Third-party software and asset sources

`scripts/fetch_assets.py` downloads the PiPER robot, wrist camera, and NVIDIA scene assets from pinned sources and verifies their SHA-256 hashes. The [asset manifest](assets/sources.lock.json) lists the sources and license notices.

The bundled FR3 URDF and the arm and gripper meshes are in `assets/robots/fr3/`. This directory preserves Franka Robotics' Apache-2.0 license and copyright notices for the `franka_description` files and documents the source of the controller URDF. [Asset notes](assets/robots/fr3/README.md) · [LICENSE](assets/robots/fr3/LICENSE) · [NOTICE](assets/robots/fr3/NOTICE)

| Source | Pinned version or commit | Usage and license notes |
|---|---|---|
| NVIDIA Isaac Sim | 5.0.0 in the current 5090 environment, 4.5.0 in the original data collection environment | Physics simulation and rendering are subject to NVIDIA's software license. |
| Isaac Lab | v2.2.0 in the current environment, v2.0.2 in earlier environments | Downloaded separately to `vendor/IsaacLab` under its upstream license. |
| AgileX `robot_lab` | `b868140eeb1459acefef24a865587ec39a5278c3` | PiPER URDF, USD, and meshes. The download includes the upstream Apache-2.0 LICENSE. |
| AgileX `piper_isaac_sim` | `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` | Manufacturer's camera mount and D435. This commit has no root LICENSE. The Piper package declaration alone does not establish a complete license grant. No separate license is assumed for the geometry. |
| Intel RealSense description | `_d435.urdf.xacro`, D435 meshes, and package declaration in the pinned `piper_isaac_sim` source tree | The source files retain Intel's license notices and attribution. |
| NVIDIA Isaac 4.5 content library | Per-file SHA-256 hashes | Tables, supports, and materials are subject to NVIDIA's asset terms of use. |
| ZPRL / ZPRL-PressB | Method source at `a34a1cba029702af82c0fed873be9e0e6bd274b3` | The [adapted SAC code](src/pressb/runtime/learner.py) retains the upstream MIT license notice. The reference repository can be cloned separately. |
| VLA-JEPA PiPER adaptation | Source revision `9c9dc71c8e199c7428b25d103ab666395832c001`, plus joint-space and multitask modules from the H200 workspace | Inference, training, and data code is bundled in `src/pressb/model_sources/vlajepa`. Its manifest records the sources and per-file hashes. Weights are not included. |
| Wall-X / Wall-OSS | `src/pressb/model_sources/walloss/source_manifest.json` records the sources and summarizes local changes | The bundled model, inference, training, data, and CUDA source code retains the original copyright headers. Weights and compiled binaries are not included. |
| NVIDIA CUTLASS | `e51efbfe18fe4f4cbb66ab814c55bf4aa0185491` | Downloaded to the cache only when building the Wall-OSS CUDA extension. The Wall-OSS source package preserves the original license text in `licenses/`. |
| DPPO / Diffusion Policy | `cc7234ad7ff39a8f32de3af903606723a16f0648` | MIT. `src/pressb/model_sources/diffusion_policy` bundles the conditional UNet, DDPM/DDIM equations, and their dependencies. It retains the LICENSE and per-file SHA-256 hashes for both original and adapted files. PressB adds visual encoding, LeRobot training, and node interfaces. Upstream model weights are not included. |
| Physical Intelligence, Hugging Face LeRobot / π₀.₅ | `d324ffe810d17264a0b1e628698aa1fa09aa639c` (package version 0.4.5) | The bundled policy, flow-matching objective, configuration, and π₀.₅-specific processors retain the upstream Apache-2.0 [LICENSE](src/pressb/model_sources/pi05/LICENSE) and [per-file sources and SHA-256 hashes](src/pressb/model_sources/pi05/source_manifest.json). `scripts/install_pi05.py` verifies the archive before installing the shared LeRobot runtime. Weights are obtained separately. |
| Ascend PyTorch extension / CANN | Verified with `torch_npu 2.10.0`, PyTorch 2.10.0, and CANN 9.0.0 | Runtime dependencies are installed separately. This repository does not distribute drivers, toolkits, or binaries. See `docs/ascend_pi05.md` for compatible environment components and the scope of validation. |

The VLA-JEPA manifest preserves the original file index so existing checkpoints can verify the source code they were built against. For two tabletop button-press training records, the manifest retains only their hashes. The framework does not distribute their contents.

`prepare_wrist_asset.py` extracts the manufacturer's mount and camera subtrees in an isolated OpenUSD 24.11 environment for compatibility with Isaac Sim 4.5. It preserves the mesh points and topology. Generated files and provenance records are stored in `assets/generated/official_wrist/` and are excluded from Git.

This repository has not specified an open-source license for its own code. Public access does not grant a license. Each third-party component remains subject to its own license.

Bundled model source files retain their existing per-file copyright and SPDX notices. Bundling grants no additional license. The VLA-JEPA source tree lacked the LICENSE file referenced in its `pyproject.toml`. The original MIT declaration is preserved separately as metadata. The Wall-X source tree also lacked a top-level LICENSE, and its original `setup.py` contained only a BSD license classifier. The repository does not treat these metadata declarations as a single license grant covering every file or supply reconstructed text for missing licenses. The two `source_manifest.json` files record each file's source, original hash, and local modifications.

The [repository manifest](repos.lock.json) lists the pinned revisions and uses of third-party repositories.

## Upstream links

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/v2.0.2)
- [AgileX robot_lab](https://github.com/agilexrobotics/robot_lab/tree/b868140eeb1459acefef24a865587ec39a5278c3)
- [AgileX piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim/tree/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2)
- [NVIDIA Omniverse software license](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html)
- [DPPO](https://github.com/irom-princeton/dppo/tree/cc7234ad7ff39a8f32de3af903606723a16f0648)
- [LeRobot π₀.₅](https://github.com/huggingface/lerobot/tree/d324ffe810d17264a0b1e628698aa1fa09aa639c/src/lerobot/policies/pi05)
