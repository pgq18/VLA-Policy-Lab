"""Different camera calibrations must not share Kit's first-camera projection."""
from types import ModuleType, SimpleNamespace
import math
import sys

import numpy as np
import pytest

from pressb.sensors.tiled import build_tiled_rgb, split_grouped_tiled_rgb
from pressb.runtime.parallel import execution_config


class Attr:
    def __init__(self, name, value): self.name, self.value = name, value
    def GetName(self): return self.name
    def Get(self): return self.value


class Prim:
    def __init__(self, values=None, targets=None): self.values, self.targets = values or {}, targets
    def IsA(self, kind): return self.targets is None
    def GetAttributes(self): return [Attr(k, v) for k, v in self.values.items()]
    def GetAttribute(self, name): return Attr(name, self.values.get(name))
    def GetRelationship(self, name): return SimpleNamespace(GetTargets=lambda: self.targets)


@pytest.fixture
def sdk(monkeypatch):
    prims = {}; products = []; annotators = []
    paths = ["/e0/base", "/e0/wrist", "/e1/base", "/e1/wrist"]
    for i, path in enumerate(paths):
        prims[path] = Prim({"focalLength": 2., "horizontalAperture": 2.11 if i % 2 == 0 else 3.27,
                            "verticalAperture": 1.58 if i % 2 == 0 else 2.46})
    stage = SimpleNamespace(GetPrimAtPath=lambda path: prims[str(path)])
    class Product:
        def __init__(self, cameras, tile_resolution, force_new):
            assert force_new
            self.path = f"/rp/{len(products)}"; self.cameras = cameras; self.resolution = tile_resolution
            self.destroyed = False; prims[self.path] = Prim(targets=cameras); products.append(self)
        def destroy(self): self.destroyed = True
    class Annotator:
        def __init__(self): self.attached = None; self.detached = None; annotators.append(self)
        def attach(self, paths): self.attached = paths
        def detach(self, paths): self.detached = paths
        def get_data(self):
            prod = next(p for p in products if p.path == self.attached[0]); n = len(prod.cameras)
            cols = math.ceil(math.sqrt(n)); rows = math.ceil(n/cols); w, h = prod.resolution
            buffer = np.zeros((rows*h, cols*w, 4), dtype=np.uint8)
            for i, path in enumerate(prod.cameras):
                y, x = divmod(i, cols); buffer[y*h:(y+1)*h, x*w:(x+1)*w, :3] = paths.index(path)
            return buffer
    omni = ModuleType("omni"); replicator = ModuleType("omni.replicator")
    rep = ModuleType("omni.replicator.core"); usd = ModuleType("omni.usd"); pxr = ModuleType("pxr")
    rep.create = SimpleNamespace(render_product_tiled=Product)
    rep.AnnotatorRegistry = SimpleNamespace(get_annotator=lambda *a, **k: Annotator())
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    pxr.UsdGeom = SimpleNamespace(Camera=object)
    omni.replicator = replicator; omni.usd = usd; replicator.core = rep
    for name, value in {"omni": omni, "omni.replicator": replicator, "omni.replicator.core": rep,
                        "omni.usd": usd, "pxr": pxr}.items(): monkeypatch.setitem(sys.modules, name, value)
    return SimpleNamespace(paths=paths, prims=prims, products=products, annotators=annotators)


def test_opt_in_groups_calibrations_and_restores_clone_camera_order(sdk):
    product, annotator, split = build_tiled_rgb(sdk.paths, (3, 2), group_by_intrinsics=True)
    assert [p.cameras for p in sdk.products] == [[sdk.paths[0], sdk.paths[2]], [sdk.paths[1], sdk.paths[3]]]
    frames = split(annotator.get_data())
    assert [int(image[0, 0, 0]) for image in frames] == [0, 1, 2, 3]
    assert product.group_count == 2
    annotator.detach([product.path]); product.destroy()
    assert all(a.detached == a.attached for a in sdk.annotators)
    assert all(p.destroyed for p in sdk.products)


def test_historical_default_retains_one_product_for_mixed_cameras(sdk):
    product, annotator, split = build_tiled_rgb(sdk.paths, (3, 2))
    assert len(sdk.products) == 1 and product is sdk.products[0]
    assert [int(image[0, 0, 0]) for image in split(annotator.get_data())] == [0, 1, 2, 3]


def test_identical_projection_uses_unchanged_single_product(sdk):
    for path in sdk.paths: sdk.prims[path].values = dict(sdk.prims[sdk.paths[0]].values)
    product, annotator, split = build_tiled_rgb(sdk.paths, (3, 2), group_by_intrinsics=True)
    assert len(sdk.products) == 1 and product is sdk.products[0]
    assert [int(image[0, 0, 0]) for image in split(annotator.get_data())] == [0, 1, 2, 3]


def test_principal_point_difference_requires_another_product(sdk):
    for path in sdk.paths: sdk.prims[path].values = dict(sdk.prims[sdk.paths[0]].values)
    sdk.prims[sdk.paths[1]].values["horizontalApertureOffset"] = .01
    build_tiled_rgb(sdk.paths, (3, 2), group_by_intrinsics=True)
    assert len(sdk.products) == 2


def test_execution_option_is_explicit_boolean():
    assert execution_config({})["camera_intrinsics_grouping"] is False
    assert execution_config({"camera_intrinsics_grouping": True})["camera_intrinsics_grouping"] is True
    with pytest.raises(ValueError): execution_config({"camera_intrinsics_grouping": "true"})


def test_grouped_split_rejects_incomplete_or_duplicate_assignments():
    data = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError): split_grouped_tiled_rgb([data, data], [[0], [0]], 2, (3, 2))
    with pytest.raises(ValueError): split_grouped_tiled_rgb([data], [[0]], 2, (3, 2))
