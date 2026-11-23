# FR3 v2.1 with Franka Hand

This asset uses the **actual FR3 v2.1 controller URDF** from the dataset capture
machine, not the Panda model bundled in older Isaac examples.

- `source/controller_fr3.urdf`: captured from `ssh franka`,
  `/home/pnp/Workspace/frankateleop/polymetis/polymetis/data/franka_fr3/fr3.urdf`.
- Arm meshes and the standard white Franka Hand come from the installed official
  [frankarobotics/franka_description](https://github.com/frankarobotics/franka_description)
  repository at commit `1aa4fd30e6e274cbf5e986a5af8004df32bad284`.
- `source/fr3v2_1_hand.urdf` is its expanded `fr3v2_1.urdf.xacro`, with
  `hand:=true with_sc:=false`. The hand's unmodified local origins, finger collision
  primitives and inertias are appended to the controller arm in `fr3_hand.urdf`.
- Only prefixes and mesh paths are changed in the combined URDF. The seven arm
  joint origins, limits and inertias remain the capture controller's values.
- Standard hand flange registration: yaw −π/4; `hand_tcp` is 0.1034 m along
  hand Z. Each finger travels +0.04 m along its own Y axis; the right finger
  frame is rotated π about Z, giving a total aperture of 0.08 m.
- Isaac conversion is cached under `assets/generated/fr3/`; it disables URDF
  mimic parsing and commands both fingers symmetrically through the robot driver.
- The retained fixed flange frames make `link7`, `link8`, `hand` and `hand_tcp`
  separate USD bodies in one rigid assembly. The authoring adapter filters
  internal collisions within every URDF fixed-joint component while retaining
  movable-link/finger collisions and general self-collision checking.
- Actuator drives explicitly use **force** semantics, so arm tensor gains are
  N·m/rad and the URDF effort caps remain 87 N·m for joints1–4 and 12 N·m for
  joints5–7. Importer-default acceleration drives are not used. Task configuration
  can override the defaults of 6000 N·m/rad and 100 N·m·s/rad. The default gripper
  gains are 1800 N/m and 60 N·s/m with a 20 N cap per simulated finger actuator;
  gripper force limits remain independent of arm gains.

The upstream Apache-2.0 `LICENSE` and `NOTICE` are retained here. Sensor mounting
and scene calibration belong to the task configuration, not this robot asset.
