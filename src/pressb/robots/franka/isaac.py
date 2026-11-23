"""Lazy Isaac authoring for the controller-matched FR3 v2.1 asset."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET


def fixed_link_groups(urdf_path):
    """Rigid assemblies separated only by fixed joints cannot self-collide.

    The importer retains fixed flange/sensor frames to attach calibrated
    cameras. Adjacent-body filtering alone then misses e.g. link7 → link8 →
    hand, even though all three form one rigid assembly.
    """
    root = ET.parse(urdf_path).getroot()
    parents = {link.get("name"): link.get("name") for link in root.findall("link")}

    def find(name):
        while parents[name] != name:
            parents[name] = parents[parents[name]]
            name = parents[name]
        return name

    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            a, b = joint.find("parent").get("link"), joint.find("child").get("link")
            parents[find(b)] = find(a)
    groups = {}
    for link in parents:
        groups.setdefault(find(link), []).append(link)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def ensure_robot_usd(urdf_path, *, cache_directory=None):
    """Import the bundled URDF with native Isaac tooling, retaining named links.

    Must be called after SimulationApp creation. The shared simulation engine
    owns physics/rendering; this function only authors a cached robot asset.
    """
    import omni.kit.app
    import omni.kit.commands
    from pxr import Usd

    urdf = Path(urdf_path).resolve()
    identity = hashlib.sha256(urdf.read_bytes()+b"pressb-fr3-import-v1").hexdigest()[:16]
    directory = (Path(cache_directory) if cache_directory else
                 urdf.parents[2]/"generated/fr3") / identity
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/"fr3.usd"
    if destination.exists():
        candidate = Usd.Stage.Open(str(destination))
        if candidate and candidate.GetDefaultPrim():
            return destination
    # Accelerometer marker links have no dynamics and are not used here.
    # Fixed flange/TCP frames remain explicit so the cameras retain their
    # calibratable registration. Give massless marker frames negligible inertia
    # instead of letting an importer invent a 1 kg default rigid body.
    robot = ET.parse(urdf).getroot()
    for node in list(robot):
        if "accelerometer" in node.get("name", ""):
            robot.remove(node)
    for link in robot.findall("link"):
        if link.find("inertial") is None and link.get("name") != "base":
            inertia = ET.SubElement(link, "inertial")
            ET.SubElement(inertia, "mass", value="0.000001")
            ET.SubElement(inertia, "inertia", ixx="1e-9", iyy="1e-9", izz="1e-9", ixy="0", ixz="0", iyz="0")
    for mesh in robot.findall(".//mesh"):
        path = Path(mesh.get("filename"))
        if not path.is_absolute():
            mesh.set("filename", str((urdf.parent/path).resolve()))
    prepared = directory/"fr3_import.urdf"
    ET.ElementTree(robot).write(prepared, encoding="utf-8", xml_declaration=True)
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        raise RuntimeError("Isaac URDF importer is not available")
    config.merge_fixed_joints = False
    config.fix_base = True
    config.make_default_prim = True
    config.create_physics_scene = False
    config.import_inertia_tensor = True
    config.parse_mimic = False
    config.self_collision = True
    config.convex_decomp = True
    config.collision_from_visuals = False
    ok, root = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(prepared),
                                       import_config=config, dest_path=str(destination))
    if not ok or not destination.is_file():
        raise RuntimeError("FR3 URDF-to-USD import failed")
    imported = Usd.Stage.Open(str(destination))
    if not imported or not imported.GetDefaultPrim():
        raise RuntimeError("Imported FR3 asset has no default prim")
    (directory/"conversion.json").write_text(json.dumps({"source": str(urdf), "usd": str(destination),
        "root": root, "robot_model": "FR3 v2.1", "mimic_parsing": False,
        "fixed_marker_mass_kg": 1e-6, "accelerometer_markers_removed": True}, indent=2)+"\n")
    return destination


def author_robot(stage, scene_config, *, path="/World/FR3"):
    """Reference FR3, author actuator/contact settings, and return link paths."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

    usd = scene_config.get("robot_usd")
    if not usd:
        usd = ensure_robot_usd(scene_config["robot_urdf"], cache_directory=scene_config.get("robot_cache_directory"))
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(str(Path(usd).resolve()))
    base = scene_config.get("robot_base_position_world", [0., 0., 0.])
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, base)))
    material = UsdShade.Material.Define(stage, path+"/Looks/FingerContact")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(scene_config.get("finger_static_friction", 1.2)))
    physics.CreateDynamicFrictionAttr(float(scene_config.get("finger_dynamic_friction", 1.0)))
    physics.CreateRestitutionAttr(0.)
    for item in Usd.PrimRange(prim):
        if item.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(item)
            articulation.CreateSolverPositionIterationCountAttr(32)
            articulation.CreateSolverVelocityIterationCountAttr(8)
            articulation.CreateEnabledSelfCollisionsAttr(True)
        if item.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(item, "angular")
            drive.CreateTypeAttr(UsdPhysics.Tokens.force)
            # Explicit force drives make tensor gains N*m/rad and N*m*s/rad.
            # USD angular drive properties are per degree; the runtime driver
            # sets the same physical gains through the tensor API after reset.
            drive.CreateStiffnessAttr(float(scene_config.get("arm_stiffness", 6000.))*3.141592653589793/180)
            drive.CreateDampingAttr(float(scene_config.get("arm_damping", 100.))*3.141592653589793/180)
        if item.IsA(UsdPhysics.PrismaticJoint):
            drive = UsdPhysics.DriveAPI.Apply(item, "linear")
            drive.CreateTypeAttr(UsdPhysics.Tokens.force)
            drive.CreateStiffnessAttr(float(scene_config.get("gripper_stiffness", 1800.)))
            drive.CreateDampingAttr(float(scene_config.get("gripper_damping", 60.)))
            drive.CreateMaxForceAttr(float(scene_config.get("gripper_max_effort", 20.)))
            UsdPhysics.PrismaticJoint(item).CreateUpperLimitAttr(float(scene_config.get("gripper_max_width_m", .08))/2)
        if item.HasAPI(UsdPhysics.CollisionAPI):
            collider = PhysxSchema.PhysxCollisionAPI.Apply(item)
            collider.CreateContactOffsetAttr(.0005)
            collider.CreateRestOffsetAttr(0.)
            if any(f"/{name}/" in str(item.GetPath()) for name in ("leftfinger", "rightfinger")):
                UsdShade.MaterialBindingAPI.Apply(item).Bind(material, materialPurpose="physics")
        if item.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(item).CreateThresholdAttr(0.)
    fixed_filters = []
    for group in fixed_link_groups(scene_config["robot_urdf"]):
        existing = [path+"/"+name for name in group if stage.GetPrimAtPath(path+"/"+name)]
        for index, source in enumerate(existing):
            targets = existing[index+1:]
            if not targets:
                continue
            relation = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(source)).CreateFilteredPairsRel()
            for target in targets:
                relation.AddTarget(Sdf.Path(target))
                fixed_filters.append([source, target])
    result = {"robot_path": path, "robot_usd": str(usd), "link8_path": path+"/link8",
              "hand_path": path+"/hand", "tcp_path": path+"/hand_tcp",
              "finger_paths": [path+"/leftfinger", path+"/rightfinger"],
              "fixed_body_collision_filters": fixed_filters}
    for link in [result["link8_path"], result["hand_path"], *result["finger_paths"]]:
        if not stage.GetPrimAtPath(link):
            raise RuntimeError("FR3 imported asset lost required named link " + link)
    return result
