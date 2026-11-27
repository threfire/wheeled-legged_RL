#!/usr/bin/env python3
"""Create a test-only compound-box contact candidate from the coarse USD.

The current base/front2 single boxes are disabled and replaced by boxes sized
around the major original convex-mesh components. Other coarse collisions and
the wheel/roller cylinders remain as authored. The production asset is untouched.
"""

import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_dir = os.path.join(
    repo_root,
    "source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration",
)
source_path = os.path.join(config_dir, "dogleg_floating_physics.usd")
output_path = os.path.join(config_dir, "dogleg_floating_physics_compound_candidate.usda")

# Each entry is (center xyz, full box dimensions xyz). Dimensions were chosen
# from the major convex-mesh component bounds in the preserved legacy layer.
compound_boxes = {
    "base_link": [
        ((0.0, 0.0, 0.0645), (0.565, 0.279, 0.147)),
        ((-0.1695, -0.0330, -0.0559), (0.1781, 0.1788, 0.0648)),
        ((-0.0917, 0.0020, -0.0439), (0.1578, 0.1705, 0.0582)),
        ((0.0, 0.2680, 0.0015), (0.141, 0.052, 0.1305)),
        ((0.0, -0.2680, 0.0015), (0.141, 0.052, 0.1305)),
    ],
    "left_front2_link": [
        ((0.0898, 0.0566, 0.0105), (0.314, 0.2409, 0.015)),
        ((0.0, 0.0, 0.022), (0.075, 0.075, 0.055)),
        ((0.202, 0.143, 0.022), (0.090, 0.090, 0.055)),
    ],
    "right_front2_link": [
        ((0.0898, -0.0566, -0.0105), (0.314, 0.2409, 0.015)),
        ((0.0, 0.0, -0.022), (0.075, 0.075, 0.055)),
        ((0.202, -0.143, -0.022), (0.090, 0.090, 0.055)),
    ],
}

if os.path.exists(output_path):
    os.remove(output_path)
layer = Sdf.Layer.CreateNew(output_path)
layer.subLayerPaths = [os.path.basename(source_path)]
layer.defaultPrim = "dogleg_floating_isaac"
stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadAll)
if stage is None:
    raise RuntimeError(f"Could not compose overlay with {source_path}")
stage.SetEditTarget(layer)

disabled = 0
for body in compound_boxes:
    prim = stage.GetPrimAtPath(f"/collisions/{body}/dogleg_simple_body_box")
    if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"Missing coarse body box for {body}")
    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    disabled += 1

count = 0
for body, boxes in compound_boxes.items():
    for index, (center, dimensions) in enumerate(boxes):
        prim_path = f"/collisions/{body}/dogleg_compound_box_{index}"
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        api.CreateCollisionEnabledAttr(True)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
        xform.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        count += 1

# Explicitly expose the per-link render instances; the source stores their
# prototype mesh library under a hidden /visuals scope.
visible_instances = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if (
        prim.GetName() == "visuals"
        and path.startswith("/dogleg_floating_isaac/base_link/")
        and prim.GetTypeName() == "Xform"
    ):
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.visible)
        visible_instances += 1

layer.Save()
print(f"Compound-box candidate: {output_path}")
print(f"Disabled coarse boxes: {disabled}; added compound boxes: {count}; visible render instances: {visible_instances}")
if (disabled, count, visible_instances) != (3, 11, 28):
    raise RuntimeError("Unexpected generated collision or render inventory")
app.close()
