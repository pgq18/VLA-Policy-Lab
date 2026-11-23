"""Task-independent tiled RGB capture for existing simulation cameras."""

from __future__ import annotations

from functools import partial
import math

import numpy as np


def split_tiled_rgb(buffer, num_cameras: int, resolution=(640, 480)) -> list[np.ndarray]:
    """Copy RGB tiles in the original camera-list order (left-to-right, then down).

    Matches IsaacLab 2.0.2 ``reshape_tiled_image`` and installed Kit 106.5:
    columns=ceil(sqrt(N)), rows=ceil(N/columns). Camera i is at
    (row=i//columns, column=i%columns); no vertical flip is applied.
    Copies prevent the following render from mutating frames queued for writing.
    """
    if not isinstance(num_cameras, int) or num_cameras <= 0:
        raise ValueError("num_cameras must be a positive integer")
    width, height = (int(value) for value in resolution)
    if width <= 0 or height <= 0:
        raise ValueError("Tile resolution must be positive")
    if isinstance(buffer, dict):
        buffer = buffer["data"]
    data = np.asarray(buffer)
    columns = math.ceil(math.sqrt(num_cameras))
    rows = math.ceil(num_cameras / columns)
    if (data.ndim != 3 or data.shape[:2] != (rows * height, columns * width)
            or data.shape[2] not in (3, 4) or data.dtype != np.uint8):
        raise ValueError(f"Invalid tiled RGB buffer {data.shape}/{data.dtype}; expected "
                         f"({rows * height}, {columns * width}, 3 or 4) uint8")
    return [np.array(data[(i // columns) * height:(i // columns + 1) * height,
                          (i % columns) * width:(i % columns + 1) * width, :3], copy=True, order="C")
            for i in range(num_cameras)]


def split_grouped_tiled_rgb(buffers, groups, num_cameras, resolution):
    """Restore global camera order after independent projection groups render."""
    if not isinstance(buffers, (list, tuple)) or len(buffers) != len(groups):
        raise ValueError("Expected one RGB buffer per camera projection group")
    result = [None] * num_cameras
    for buffer, indices in zip(buffers, groups):
        tiles = split_tiled_rgb(buffer, len(indices), resolution)
        for index, tile in zip(indices, tiles):
            if not 0 <= index < num_cameras or result[index] is not None:
                raise ValueError("Repeated or invalid grouped camera index")
            result[index] = tile
    if any(tile is None for tile in result):
        raise ValueError("Projection groups omit a requested camera")
    return result


class _GroupedRenderProducts:
    """Keep the existing runtime's render-product lifetime interface."""
    def __init__(self, products):
        self.products = products
        self.paths = [product.path for product in products]
        self.path = self.paths[0]
        self.group_count = len(products)

    def destroy(self):
        error = None
        for product in self.products:
            try:
                product.destroy()
            except Exception as caught:
                error = error or caught
        if error is not None:
            raise error


class _GroupedRGBAnnotators:
    def __init__(self, annotators, products):
        self.annotators, self.products = annotators, products

    def get_data(self):
        return [annotator.get_data() for annotator in self.annotators]

    def detach(self, _paths=None):
        error = None
        for annotator, product in zip(self.annotators, self.products):
            try:
                annotator.detach([product.path])
            except Exception as caught:
                error = error or caught
        if error is not None:
            raise error


_PROJECTION_PROPERTIES = (
    "projection", "focalLength", "horizontalAperture", "verticalAperture",
    "horizontalApertureOffset", "verticalApertureOffset", "clippingRange",
    "clippingPlanes", "fStop", "focusDistance", "shutter:open", "shutter:close",
)


def _camera_projection_key(prim):
    # Pose is deliberately excluded: clones share optics, not position. Include
    # lens-model properties too, when a task authors an RTX distortion model.
    names = set(_PROJECTION_PROPERTIES)
    names.update(attr.GetName() for attr in prim.GetAttributes()
                 if attr.GetName().startswith(("omni:lensdistortion:", "ftheta")))
    return tuple((name, repr(prim.GetAttribute(name).Get())) for name in sorted(names))


def build_tiled_rgb(camera_paths, resolution=(640, 480), *, group_by_intrinsics=False):
    """Return (render_product, CPU_RGB_annotator, split_fn) for existing cameras.

    No Core Camera wrappers are created. Caller renders once, reads
    ``annotator.get_data()``, then invokes ``split_fn``.
    ``camera_paths`` order is preserved by the render product relationship.
    Kit 107.3 tiled products use their first camera's projection for every tile.
    ``group_by_intrinsics=True`` creates one tiled product per projection group,
    all with the same tile resolution, and restores the requested camera order.
    The default preserves historical single-product rendering for existing
    experiments. Same-projection cameras use the original path in either mode.
    """
    import omni.replicator.core as rep
    import omni.usd
    from pxr import UsdGeom

    paths = [str(path) for path in camera_paths]
    if type(group_by_intrinsics) is not bool:
        raise ValueError("group_by_intrinsics must be boolean")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Expected a nonempty list of distinct camera paths")
    dimensions = np.asarray(resolution)
    if (dimensions.shape != (2,) or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0) or np.any(dimensions != np.floor(dimensions))):
        raise ValueError("Camera resolution must contain two positive integers")
    resolution = tuple(int(value) for value in dimensions)
    stage = omni.usd.get_context().get_stage()
    for path in paths:
        if not stage.GetPrimAtPath(path).IsA(UsdGeom.Camera):
            raise ValueError(f"Missing Camera prim: {path}")
    grouped = {}
    for index, path in enumerate(paths):
        key = _camera_projection_key(stage.GetPrimAtPath(path)) if group_by_intrinsics else None
        grouped.setdefault(key, []).append(index)
    groups = list(grouped.values())
    products, annotators = [], []
    try:
        for indices in groups:
            selected = [paths[index] for index in indices]
            product = rep.create.render_product_tiled(cameras=selected, tile_resolution=resolution, force_new=True)
            products.append(product)
            authored = [str(path) for path in stage.GetPrimAtPath(product.path).GetRelationship("camera").GetTargets()]
            if authored != selected:
                raise RuntimeError("Tiled render product changed the requested camera order")
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu", do_array_copy=True)
            annotators.append(annotator)
            annotator.attach([product.path])
    except Exception:
        try:
            _GroupedRGBAnnotators(annotators, products).detach()
        finally:
            _GroupedRenderProducts(products).destroy()
        raise
    if len(groups) == 1:
        return products[0], annotators[0], partial(split_tiled_rgb, num_cameras=len(paths), resolution=resolution)
    return (_GroupedRenderProducts(products), _GroupedRGBAnnotators(annotators, products),
            partial(split_grouped_tiled_rgb, groups=groups, num_cameras=len(paths), resolution=resolution))
