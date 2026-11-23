"""FR3 task factory, including zero-tick pose publication for Isaac Sim 6."""
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pressb.runtime.parallel import SharedWorldBackend
from .parallel import Fr3PapTask


class _RenderPose:
    """USD ops of a cloned FR3 rigid body; its parent is fixed in the world."""

    def __init__(self, stage, path):
        from pxr import Usd, UsdGeom

        prim = stage.GetPrimAtPath(path)
        xf = UsdGeom.Xformable(prim)
        ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
        self.translate = ops[UsdGeom.XformOp.TypeTranslate]
        self.orient = ops.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp()
        self.point_type = type(self.translate.Get())
        from pxr import Gf
        self.quat_type = (Gf.Quatd if self.orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble else Gf.Quatf)
        self.parent_inverse = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).GetInverse()

    def publish(self, position, xyzw):
        from pxr import Gf

        world = Gf.Matrix4d(1)
        x, y, z, w = map(float, xyzw)
        world.SetRotate(Gf.Quatd(w, x, y, z))
        world.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
        local = world * self.parent_inverse
        self.translate.Set(self.point_type(*local.ExtractTranslation()))
        q = local.ExtractRotationQuat().GetNormalized()
        self.orient.Set(self.quat_type(q.GetReal(), *q.GetImaginary()))


class Isaac6Fr3PapTask(Fr3PapTask):
    def metadata(self):
        meta = super().metadata()
        data = Path(__file__).read_bytes()
        meta['source_identity']['tasks/fr3_pap/backend.py'] = {'sha256': sha256(data).hexdigest(), 'bytes': len(data)}
        meta['execution'] = {'reset_pose_publication': 'isaac6_measured_rigid_and_link_poses'}
        return meta

    def bind(self, engine):
        super().bind(engine)
        self.bowl_render_poses = [
            _RenderPose(engine.stage, engine.relocate(i, self.scene['bowl_path'])) for i in range(engine.num_envs)]
        robot = engine.robot
        paths = robot.arms.view.link_paths
        self.link_render_poses = [
            [_RenderPose(engine.stage, path) for path in paths[row]] for row in robot.arm_rows]

    def reset_task(self, env_ids, specs):
        from pxr import Sdf

        info = super().reset_task(env_ids, specs)
        # Isaac 6 tensor teleports update physics immediately, but a kinematic
        # bowl's USD pose can remain at authoring time indefinitely. Likewise,
        # before the first physics tick link poses driving the wrist camera and
        # gripper meshes may be stale. Publish measured poses before reset RGB.
        engine = self.engine
        engine.world.physics_sim_view.update_articulations_kinematic()
        links = engine.robot.arms.view.get_link_transforms()
        positions, quats = self.bodies.get_world_poses()
        with Sdf.ChangeBlock():
            for eid in env_ids:
                bowl = self.rows[eid, 1]
                self.bowl_render_poses[eid].publish(positions[bowl], quats[bowl, [1, 2, 3, 0]])
                for render_pose, pose in zip(self.link_render_poses[eid], links[engine.robot.arm_rows[eid]], strict=True):
                    render_pose.publish(pose[:3], pose[3:])
        return info


def create_backend(config, *, engine_factory=None):
    try:
        major = version('isaacsim').split('.')[0]
    except PackageNotFoundError:
        major = None
    task = Isaac6Fr3PapTask if major == '6' else Fr3PapTask
    return SharedWorldBackend(task(config), engine_factory=engine_factory)
