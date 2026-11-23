"""Optional visual-only finger, palm-ridge and base-status-LED appearance.

The upstream finger mesh distinguishes its rubber pad and metal carrier with
GeomSubsets. Rebinding those subsets preserves geometry, joints, collision
material, friction and mass. Palm-ridge recoloring, when explicitly selected,
affects exactly two triangles; the remaining palm and arm retain their materials.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import permutations
import math


DEFAULTS = {
    "finger_pad": {"color": [.042, .048, .050], "roughness": .90, "metallic": 0., "specular": .025},
    "finger_metal": {"color": [.34, .35, .33], "roughness": .58, "metallic": .25, "specular": .22},
    "palm_bar": {"color": [.11, .12, .115], "roughness": .65, "metallic": .2, "specular": .15},
    # Only an explicit status_led entry enables this override. Intensity uses
    # the imported OmniPBR emission units (the upstream white LED uses 10000).
    "status_led": {"color": [.025, 1., .38], "emission_intensity": 5000.},
}
SUBSETS = {"Part__Feature_007": "finger_pad", "Part__Feature001_006": "finger_metal"}

# The two triangles forming the narrow palm ridge visible along the bottom of
# the real wrist image. Coordinates are in hand metres, from the upstream DAE;
# selecting by geometry avoids depending on importer-dependent face numbering.
PALM_BAR_TRIANGLES_HAND_M = (
    ((.01733583,-.08140993,.06283122),(.01825010,-.08402371,.05801284),(.01730364,.04331290,.06294345)),
    ((.01825010,-.08402371,.05801284),(.01827234,.08228188,.05795037),(.01730364,.04331290,.06294345)),
)


def normalize_appearance(options):
    if not isinstance(options, dict) or set(options)-set(DEFAULTS):
        raise ValueError("FR3 appearance supports finger_pad, finger_metal, palm_bar and status_led")
    result = {}
    for part, updates in options.items():
        if not isinstance(updates, dict) or set(updates)-set(DEFAULTS[part]):
            raise ValueError("Unknown properties for FR3 appearance part: "+part)
        item = {**deepcopy(DEFAULTS[part]), **deepcopy(updates)}
        color = item["color"]
        if not isinstance(color, (list,tuple)) or len(color)!=3:
            raise ValueError("Finger color must have three components")
        bounded = [*color] if part=="status_led" else [*color, item["roughness"],item["metallic"],item["specular"]]
        for value in bounded:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError("Finger appearance values must be finite numbers in [0,1]")
        if part=="status_led":
            value=item["emission_intensity"]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                raise ValueError("LED emission_intensity must be finite and nonnegative")
        result[part] = item
    return result


def _apply_palm_bar(stage, robot_path, settings):
    """Assign the empirical ridge appearance to exactly two matched triangles."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdShade
    from pressb.runtime.scene import _material

    hand = stage.GetPrimAtPath(robot_path+"/hand")
    visuals = stage.GetPrimAtPath(robot_path+"/hand/visuals")
    if not hand or not visuals:
        raise RuntimeError("FR3 hand visual root missing")
    visuals.SetInstanceable(False)
    cache = UsdGeom.XformCache()
    inv_hand = cache.GetLocalToWorldTransform(hand).GetInverse()
    targets = np.asarray(PALM_BAR_TRIANGLES_HAND_M)
    found = {0:[],1:[]}
    for prim in Usd.PrimRange(visuals):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        relative = cache.GetLocalToWorldTransform(prim)*inv_hand
        local = np.array([relative.Transform(Gf.Vec3d(*p)) for p in points])
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
        starts = np.r_[0,np.cumsum(counts)[:-1]]
        face_ids = np.flatnonzero(counts==3)
        triangles = local[indices[starts[face_ids,None]+np.arange(3)]]
        for ti,target in enumerate(targets):
            distances = np.stack([np.max(np.linalg.norm(triangles-target[list(order)],axis=-1),axis=-1)
                                  for order in permutations(range(3))])
            matches = face_ids[np.any(distances<=1e-5,axis=0)]
            found[ti].extend((prim,int(fid)) for fid in matches)
    if any(len(rows)!=1 for rows in found.values()):
        raise RuntimeError("Expected exactly two uniquely matched FR3 palm-bar triangles")
    rows = [found[ti][0] for ti in (0,1)]
    if rows[0][0] != rows[1][0] or rows[0][1]==rows[1][1]:
        raise RuntimeError("FR3 palm-bar faces must be distinct and belong to one mesh")
    prim = rows[0][0]
    selected = {rows[0][1],rows[1][1]}
    # Remove these faces from any existing material family subset before adding
    # the new subset, preserving a non-overlapping material partition.
    for child in prim.GetChildren():
        if child.IsA(UsdGeom.Subset):
            subset=UsdGeom.Subset(child)
            if subset.GetFamilyNameAttr().Get()=="materialBind":
                current=list(subset.GetIndicesAttr().Get() or [])
                if selected.intersection(current):
                    subset.GetIndicesAttr().Set([i for i in current if i not in selected])
    subset=UsdGeom.Subset.Define(stage,str(prim.GetPath())+"/PressB_PalmBar")
    subset.CreateElementTypeAttr("face")
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr(sorted(selected))
    material=_material(stage,robot_path+"/Looks/PressB_palm_bar",settings["color"],
        roughness=settings["roughness"],metallic=settings["metallic"],specular=settings["specular"])
    UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    return str(subset.GetPath())


def apply_appearance(stage, robot_path, options):
    """Author only explicitly selected visual materials, returning affected paths.

    Pass ``{"finger_pad": {}, "finger_metal": {}}`` for the measured-image
    starting values, or override either independently. Empty options is a no-op.
    Imported visual instances are made editable; collision instances are untouched.
    """
    config = normalize_appearance(options)
    if not config:
        return {"materials":{}, "subsets":[]}
    from pxr import Gf, Sdf, Usd, UsdShade
    from pressb.runtime.scene import _material

    materials = {key:_material(stage,robot_path+"/Looks/PressB_"+key,
        value["color"],roughness=value["roughness"],metallic=value["metallic"],specular=value["specular"])
        for key,value in config.items() if key in SUBSETS.values()}
    affected = []
    for finger in ("leftfinger","rightfinger"):
        if not materials:
            break
        root = stage.GetPrimAtPath(robot_path+"/"+finger+"/visuals")
        if not root:
            raise ValueError("Missing Franka Hand visual root: "+finger)
        # Bindings on instance-proxy subsets cannot be authored. This changes
        # only visual instancing, never the collision or articulation topology.
        root.SetInstanceable(False)
        matched = set()
        for prim in Usd.PrimRange(root):
            key = SUBSETS.get(prim.GetName())
            if key in materials and prim.GetTypeName()=="GeomSubset":
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[key])
                affected.append(str(prim.GetPath()));matched.add(key)
        if matched != set(materials):
            raise RuntimeError("Franka Hand visual subset mapping changed: "+finger)
    if "status_led" in config:
        # The imported material is uniquely assigned to the base-status LED
        # subset. Override its shader rather than duplicating the dense link0
        # visual mesh or touching any status/physics controller.
        material = stage.GetPrimAtPath(robot_path+"/Looks/LED_Lights_white")
        if not material:
            raise RuntimeError("FR3 base LED material mapping changed")
        shaders = [UsdShade.Shader(prim) for prim in Usd.PrimRange(material) if prim.IsA(UsdShade.Shader)]
        if not shaders:
            raise RuntimeError("FR3 base LED has no shader")
        led=config["status_led"]
        for shader in shaders:
            if shader.GetPrim().GetAttribute("info:mdl:sourceAsset").HasAuthoredValue():
                shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
                shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*led["color"]))
                shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(led["emission_intensity"])
            else:
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*led["color"]))
        affected.append(str(material.GetPath()))
    if "palm_bar" in config:
        affected.append(_apply_palm_bar(stage,robot_path,config["palm_bar"]))
    return {"materials":config,"subsets":affected}
