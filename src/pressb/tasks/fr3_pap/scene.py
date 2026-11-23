"""FR3 tabletop authoring; simulation clocks and rendering belong to runtime.

All distances are metres in the robot base frame. The bowl is an open shell
with compound convex wall colliders, never a filled convex hull over its mouth.
"""
from __future__ import annotations

import math
import numpy as np


def bowl_profile(spec):
    """Outside/inside meridian, bottom to rim (radius, height)."""
    height = float(spec["height_m"])
    bottom, top = spec["bottom_diameter_m"] / 2, spec["top_diameter_m"] / 2
    wall, floor = spec["wall_thickness_m"], spec["bottom_thickness_m"]
    # A measured silhouette can replace the linear taper without changing
    # collision authoring. Normalized radii interpolate the supplied diameters.
    t = np.array([0., .12, .36, .68, 1.])
    radii = bottom + (top-bottom)*t
    outer = np.c_[radii, height*t]
    inner_z = floor + (height-floor)*t
    inner = np.c_[bottom + (top-bottom)*(inner_z/height)-wall, inner_z]
    return outer, inner


def revolved_shell(spec, segments=96):
    outer, inner = bowl_profile(spec)
    # Closed radial polygon: outer wall -> lip -> inner wall -> bottom.
    profile = np.vstack(([0., 0.], outer, inner[::-1], [0., spec["bottom_thickness_m"]]))
    angle = np.arange(segments)*2*math.pi/segments
    points = np.array([[r*math.cos(a), r*math.sin(a), z] for r,z in profile for a in angle])
    faces = []
    for j in range(len(profile)-1):
        for i in range(segments):
            k = (i+1)%segments
            faces.extend([j*segments+i,j*segments+k,(j+1)*segments+k,(j+1)*segments+i])
    return points, faces


def build_template(cfg, execution):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema
    from isaacsim.core.api import World
    from pressb.runtime.parallel_contract import SceneTemplate
    from pressb.runtime.scene import _cube, _material, _bind
    from pressb.robots.franka.isaac import author_robot
    from pressb.sensors.camera import make_camera, set_intrinsics
    from pressb.sensors.color_response import normalize_camera_responses, author_camera_response

    world = World(stage_units_in_meters=1., physics_dt=execution["physics_dt"],
                  rendering_dt=1/execution["control_hz"], backend="numpy", device="cpu")
    stage = world.stage
    UsdGeom.Xform.Define(stage, "/World/Scene")
    table, cube, bowl = cfg["table"], cfg["cube"], cfg["bowl"]
    top_z = float(table["top_z"])
    looks = "/World/Scene/Looks"
    tabletop = _material(stage, looks+"/Table", table.get("color", [.57,.57,.55]),
                         roughness=table.get("roughness",.72), specular=table.get("specular",.22),
                         static_friction=.7, dynamic_friction=.6)
    matte = _material(stage, looks+"/Black", [.006,.007,.008], roughness=.95, specular=.08)
    metal = _material(stage, looks+"/Mount", [.36,.38,.39], roughness=.34, metallic=.65)
    blue = _material(stage, looks+"/BlueFoam", cube.get("color", [.10,.29,.41]),
                     roughness=.88, specular=.13, static_friction=.85, dynamic_friction=.75)
    compliance = cube.get("compliant_contact", {})
    if compliance.get("enabled", False):
        # This is a force-based contact-spring approximation, not deformable
        # volume simulation. Only the foam is compliant: robot/table/bowl keep
        # their ordinary hard contacts. Unloaded dimensions remain 50 mm.
        stiffness = float(compliance.get("stiffness_n_m", 1500.))
        damping = float(compliance.get("damping_n_s_m", 4.))
        if not math.isfinite(stiffness) or stiffness <= 0 or not math.isfinite(damping) or damping < 0:
            raise ValueError("Foam contact stiffness must be positive and damping nonnegative")
        soft = PhysxSchema.PhysxMaterialAPI.Apply(blue.GetPrim())
        soft.CreateCompliantContactStiffnessAttr(stiffness)
        soft.CreateCompliantContactDampingAttr(damping)
        soft.CreateCompliantContactAccelerationSpringAttr(False)
    brown = _material(stage, looks+"/Bowl", bowl.get("color", [.16,.105,.052]),
                      roughness=.29, specular=.65, static_friction=.65, dynamic_friction=.5)
    sx,sy = table["size_m"][:2]
    thick = table.get("thickness_m", .025)
    tx,ty = table["center_xy"]
    table_path = "/World/Scene/Table/Top"
    # The nominal square remains the reset/sampling workspace. An explicitly
    # fitted front margin covers the visible worktop edge beyond that workspace.
    front = float(table.get("front_extension_m",0.))
    if not math.isfinite(front) or not 0 <= front <= .3:
        raise ValueError("table.front_extension_m must lie in [0,.3] metres")
    _cube(stage, table_path, [tx+front/2,ty,top_z-thick/2], [sx+front,sy,thick], tabletop, collision=True)
    seam = _material(stage, looks+"/Seam", table.get("seam_color",[.30,.30,.29]), roughness=.9)
    # The two adjacent white boards in the dataset have one subtle seam.
    seam_x = table.get("seam_x_m", .69)
    _cube(stage, "/World/Scene/Table/Seam", [seam_x,ty,top_z+.00003],
          [table.get("seam_width_m",.0006),sy,.00006], seam, yaw_deg=table.get("seam_yaw_deg",0.))
    for ix in (-1,1):
        for iy in (-1,1):
            _cube(stage, f"/World/Scene/Table/Leg_{ix+1}_{iy+1}",
                  [tx+ix*(sx/2-.06),ty+iy*(sy/2-.06),top_z-.39], [.045,.045,.73], metal)
    # L-shaped black curtains visible behind the real table.
    curtain_bottom = table.get("curtain_bottom_z_m",-.35)
    curtain_top = table.get("curtain_top_z_m",1.65)
    curtain_mid,curtain_height = (curtain_top+curtain_bottom)/2,curtain_top-curtain_bottom
    if curtain_height <= 0:
        raise ValueError("Curtain top must be above its bottom")
    _cube(stage, "/World/Scene/BackdropRear", [table.get("rear_curtain_x_m", -.22),0.,curtain_mid], [.015,2.0,curtain_height], matte)
    _cube(stage, "/World/Scene/BackdropSide", [.40,table.get("side_curtain_y_m",-.67),curtain_mid], [1.5,.015,curtain_height], matte)
    _cube(stage, "/World/Scene/MountPlate", [0.,0.,-.001], [.20,.19,.002], metal)

    cube_path = "/World/Scene/Cube"
    edge = float(cube["edge_m"])
    shape = _cube(stage, cube_path, [*cube["center_xy"],top_z+edge/2+.0005], [edge]*3,
                  blue, collision=True, yaw_deg=cube.get("yaw_deg",0.))
    body = UsdPhysics.RigidBodyAPI.Apply(shape.GetPrim())
    body.CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(shape.GetPrim()).CreateMassAttr(cube.get("mass_kg", .025))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(shape.GetPrim())
    rb.CreateSolverPositionIterationCountAttr(16); rb.CreateSolverVelocityIterationCountAttr(4)
    rb.CreateSleepThresholdAttr(0.)
    PhysxSchema.PhysxContactReportAPI.Apply(shape.GetPrim()).CreateThresholdAttr(0.)

    bowl_path = "/World/Scene/Bowl"
    bowl_root = UsdGeom.Xform.Define(stage,bowl_path)
    bowl_root.AddTranslateOp().Set(Gf.Vec3d(*bowl["center_xy"],top_z))
    # Kinematic pose is changed only during indexed reset. The fixed bowl cannot
    # be pushed across a cloned scene by another environment's reset.
    bb = UsdPhysics.RigidBodyAPI.Apply(bowl_root.GetPrim())
    bb.CreateRigidBodyEnabledAttr(True); bb.CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxContactReportAPI.Apply(bowl_root.GetPrim()).CreateThresholdAttr(0.)
    UsdPhysics.MassAPI.Apply(bowl_root.GetPrim()).CreateMassAttr(.15)
    points, faces = revolved_shell(bowl)
    visual = UsdGeom.Mesh.Define(stage,bowl_path+"/Shell")
    visual.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    visual.CreateFaceVertexCountsAttr([4]*(len(faces)//4));visual.CreateFaceVertexIndicesAttr(faces)
    visual.CreateSubdivisionSchemeAttr("none");visual.CreateDoubleSidedAttr(True)
    _bind(visual.GetPrim(),brown)
    # An actual floor plus separately convex wall wedges preserve the opening.
    floor = UsdGeom.Cylinder.Define(stage,bowl_path+"/Collision/Floor")
    floor.CreateRadiusAttr(bowl["bottom_diameter_m"]/2)
    floor.CreateHeightAttr(bowl["bottom_thickness_m"])
    floor.AddTranslateOp().Set(Gf.Vec3d(0,0,bowl["bottom_thickness_m"]/2))
    _bind(floor.GetPrim(),brown,True)
    floor.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    outer, inner = bowl_profile(bowl)
    segments = 48
    for i in range(segments):
        a,b = i*2*math.pi/segments,(i+1)*2*math.pi/segments
        # Linear tapered wall: a closed convex eight-vertex wedge per segment.
        cross = [outer[0],outer[-1],inner[-1],inner[0]]
        verts = [[r*math.cos(ang),r*math.sin(ang),z] for ang in (a,b) for r,z in cross]
        wall = UsdGeom.Mesh.Define(stage,bowl_path+f"/Collision/Wall_{i:02d}")
        wall.CreatePointsAttr([Gf.Vec3f(*p) for p in verts])
        wall.CreateFaceVertexCountsAttr([4]*6)
        wall.CreateFaceVertexIndicesAttr([0,3,2,1,4,5,6,7,0,1,5,4,1,2,6,5,2,3,7,6,3,0,4,7])
        wall.CreateSubdivisionSchemeAttr("none");wall.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        _bind(wall.GetPrim(),brown,True)
        UsdPhysics.MeshCollisionAPI.Apply(wall.GetPrim()).CreateApproximationAttr("convexHull")

    robot = author_robot(stage, cfg, path="/World/FR3")
    robot_path = robot["robot_path"]
    if cfg.get("robot_appearance"):
        from pressb.robots.franka.appearance import apply_appearance
        apply_appearance(stage,robot_path,cfg["robot_appearance"])
    cameras = cfg["cameras"]
    base_path = make_camera(stage,"/World/Scene/BaseCamera",cameras["base"],np.zeros(3))
    wrist_spec = cameras["wrist"]
    parent_link = wrist_spec.get("parent_link","link8")
    parent_path = robot.get(parent_link+"_path",robot_path+"/"+parent_link)
    if not stage.GetPrimAtPath(parent_path):
        raise ValueError("Wrist camera parent link missing: "+parent_path)
    wrist_path = parent_path+"/WristCamera"
    camera = UsdGeom.Camera.Define(stage,wrist_path)
    transform = np.asarray(wrist_spec["link_from_camera_usd"],dtype=float)
    camera.AddTransformOp().Set(Gf.Matrix4d(*transform.T.ravel().tolist()))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.005,20.))
    set_intrinsics(camera.GetPrim(),wrist_spec)
    if wrist_spec.get("show_housing", True):
        # Visual D405 housing behind the calibrated optical centre. It does
        # not replace the sensor transform or introduce unmeasured collisions.
        housing = UsdGeom.Xform.Define(stage,parent_path+"/D405Housing")
        housing.AddTransformOp().Set(Gf.Matrix4d(*transform.T.ravel().tolist()))
        plastic = _material(stage,looks+"/CameraPlastic",[.66,.67,.67],roughness=.45)
        _cube(stage,str(housing.GetPath())+"/Body",[0.,0.,.012],[.042,.042,.023],plastic)
        _cube(stage,str(housing.GetPath())+"/Window",[0.,0.,.0005],[.034,.015,.0008],matte)
    lighting = cfg.get("lighting",{})
    dome = UsdLux.DomeLight.Define(stage,"/World/Scene/Ambient")
    dome.CreateIntensityAttr(lighting.get("ambient_intensity",250.))
    dome.CreateColorAttr(Gf.Vec3f(*lighting.get("ambient_color",[.92,.96,1.])))
    ceiling_lights = lighting.get("ceiling_lights")
    if ceiling_lights is None:
        ceiling_lights = [{"position_m":eye} for eye in lighting.get("ceiling_positions",[[.4,-.1,1.9],[.7,.7,1.9]])]
    for index,light in enumerate(ceiling_lights):
        eye = light["position_m"]
        lamp = UsdLux.RectLight.Define(stage,f"/World/Scene/Ceiling_{index}")
        lamp.AddTranslateOp().Set(Gf.Vec3d(*eye))
        lamp.AddRotateZOp().Set(float(light.get("yaw_deg",lighting.get("ceiling_yaw_deg",0.))))
        lamp.CreateWidthAttr(light.get("width_m",lighting.get("ceiling_width_m",1.2)))
        lamp.CreateHeightAttr(light.get("height_m",lighting.get("ceiling_height_m",.25)))
        lamp.CreateIntensityAttr(light.get("intensity",lighting.get("ceiling_intensity",550.)))
        lamp.CreateColorAttr(Gf.Vec3f(*light.get("color",lighting.get("ceiling_color",[1.,.98,.94]))))
    for name,response in normalize_camera_responses(cfg.get("camera_color_response"),("base","wrist")).items():
        author_camera_response(stage.GetPrimAtPath({"base":base_path,"wrist":wrist_path}[name]),response)
    scene = dict(cube_path=cube_path,bowl_path=bowl_path,table_path=table_path,table_top_z=top_z,
                 bowl_center_xy=list(bowl["center_xy"]),bowl_height_m=bowl["height_m"],
                 bowl_bottom_thickness_m=bowl["bottom_thickness_m"],robot=robot,
                 finger_paths=robot.get("finger_paths",[robot_path+"/leftfinger",robot_path+"/rightfinger"]))
    return SceneTemplate(world,stage,robot_path,{"base":base_path,"wrist":wrist_path},
                         data={"scene":scene,"robot":robot})
