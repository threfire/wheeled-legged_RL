#!/usr/bin/env python3
"""Build the Dogleg contact-fixed physics overlay from the preserved base asset.

It restores the original convex mesh colliders on the base and front2 links,
disabling only their coarse simple boxes. Wheel cylinders and all other current
simplifications remain unchanged. The source USD remains available as rollback.
"""

import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_dir = os.path.join(
    repo_root,
    "source/agent_world/agent_world/assets/usd_files/dogleg_floating/configuration",
)
source_path = os.path.join(config_dir, "dogleg_floating_physics.usd")
output_path = os.path.join(config_dir, "dogleg_floating_physics_contactfix.usda")
target_bodies = ("base_link", "left_front2_link", "right_front2_link")

if os.path.exists(output_path):
    os.remove(output_path)
layer = Sdf.Layer.CreateNew(output_path)
layer.subLayerPaths = [os.path.basename(source_path)]
layer.defaultPrim = "dogleg_floating_isaac"
stage = Usd.Stage.Open(layer, load=Usd.Stage.LoadAll)
if stage is None:
    raise RuntimeError(f"Could not compose overlay with {source_path}")
stage.SetEditTarget(layer)

restored = {name: 0 for name in target_bodies}
disabled_boxes = {name: 0 for name in target_bodies}
visible_instances = 0
for prim in stage.Traverse():
    path = str(prim.GetPath())
    # The imported USD stores render meshes in a root-level invisible prototype
    # library (/visuals). Force each per-link visual instance visible so the
    # referenced render meshes are not suppressed by the prototype visibility.
    if (
        prim.GetName() == "visuals"
        and path.startswith("/dogleg_floating_isaac/base_link/")
        and prim.GetTypeName() == "Xform"
    ):
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.visible)
        visible_instances += 1
    for body in target_bodies:
        prefix = f"/collisions/{body}/"
        if not path.startswith(prefix):
            continue
        if prim.GetTypeName() == "Mesh" and prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdPhysics.CollisionAPI(prim)
            api.GetCollisionEnabledAttr().Set(True)
            restored[body] += 1
        elif prim.GetName() == "dogleg_simple_body_box" and prim.HasAPI(UsdPhysics.CollisionAPI):
            api = UsdPhysics.CollisionAPI(prim)
            api.GetCollisionEnabledAttr().Set(False)
            disabled_boxes[body] += 1

layer.Save()
print(f"Contact-fixed production overlay: {output_path}")
print(f"Visible render instances: {visible_instances}")
if visible_instances != 28:
    raise RuntimeError(f"Expected 28 link visual instances, found {visible_instances}")
for body in target_bodies:
    print(f"{body}: restored mesh colliders={restored[body]}, disabled coarse boxes={disabled_boxes[body]}")
    if restored[body] == 0 or disabled_boxes[body] != 1:
        raise RuntimeError(f"Unexpected collider inventory for {body}")
app.close()
