# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RBY1 articulation wrapper with PhysX joint/motor configuration."""
from __future__ import annotations

import re
from typing import Optional, Sequence

import carb
import numpy as np
import omni
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.prims import GeometryPrim, SingleArticulation, SingleRigidPrim
from pxr import PhysxSchema, Sdf, UsdPhysics

from motor_profiles import get_profile_for_joint


# Wheel collision geometry prim-path patterns per RBY1 model.
WHEEL_COLLISION_PRIM_PATH_EXPRS = {
    "m": "/World/RBY1/wheel_.*_link/Roller_.*/Sphere_.*",
    "a": "/World/RBY1/wheel_.*/collisions/mesh_0/cylinder",
}

# Model-M mecanum roller joints are passive, but a tiny angular drive damping
# improves PhysX stability without trying to hold roller position.
MECANUM_ROLLER_REVOLUTE_JOINT_PATH_EXPR = "/World/RBY1/wheel_.*_link/Roller_.*/RevoluteJoint"
MECANUM_ROLLER_DRIVE_STIFFNESS = 0.0
MECANUM_ROLLER_DRIVE_DAMPING = 0.0005
MECANUM_ROLLER_TARGET_VELOCITY = 0.0

# Model-A only: extra base collision geometry that needs a frictionless material.
MODEL_A_EXTRA_GEOMETRY_PRIM_PATH_EXPR = "/World/RBY1/base/collisions/mesh_.*/cylinder"
MODEL_A_EXTRA_GEOMETRY_MATERIAL_CONFIG = {
    "prim_path": "/World/Physics_Materials/model_a_extra_geometry_material",
    "dynamic_friction": 0.3,
    "static_friction": 0.3,
    "restitution": 0.0,
}


class RBY1Robot(SingleArticulation):
    """Articulation wrapper that applies PhysX drive/motor properties at setup."""

    def __init__(
        self,
        prim_path: str,
        name: str = "rby1_robot",
        position: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        robot_model: str = "m",
    ) -> None:
        self.robot_model = str(robot_model).lower()
        if self.robot_model not in WHEEL_COLLISION_PRIM_PATH_EXPRS:
            valid_models = ", ".join(sorted(WHEEL_COLLISION_PRIM_PATH_EXPRS))
            raise ValueError(
                f"Unsupported RBY1 model '{robot_model}'. Valid models: {valid_models}"
            )
        self.base_prim_path = prim_path + "/base"
        self._base = None
        SingleArticulation.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            articulation_controller=None,
        )
        self._torso_5 = None
        self._link_left_arm_6 = None
        self._link_right_arm_6 = None

    def post_reset(self) -> None:
        SingleArticulation.post_reset(self)
        if self._torso_5 is not None:
            self._torso_5.post_reset()
        if self._link_left_arm_6 is not None:
            self._link_left_arm_6.post_reset()
        if self._link_right_arm_6 is not None:
            self._link_right_arm_6.post_reset()

    def initialize(self, physics_sim_view: omni.physics.tensors.SimulationView = None) -> None:
        SingleArticulation.initialize(self, physics_sim_view=physics_sim_view)

        # USD is fully spawned at initialize() time, so the rigid prims are safe to create.
        self._torso_5 = SingleRigidPrim(
            prim_path=self.prim_path + "/link_torso_5",
            name=self.name + "_torso_5",
        )
        self._torso_5.initialize(physics_sim_view=physics_sim_view)

        self._link_left_arm_6 = SingleRigidPrim(
            prim_path=self.prim_path + "/link_left_arm_6",
            name=self.name + "_link_left_arm_6",
        )
        self._link_left_arm_6.initialize(physics_sim_view=physics_sim_view)

        self._link_right_arm_6 = SingleRigidPrim(
            prim_path=self.prim_path + "/link_right_arm_6",
            name=self.name + "_link_right_arm_6",
        )
        self._link_right_arm_6.initialize(physics_sim_view=physics_sim_view)

    # ------------------------------------------------------------------
    # Rigid-prim accessors
    # ------------------------------------------------------------------

    @property
    def torso_5(self) -> SingleRigidPrim:
        return self._torso_5

    @property
    def base(self) -> SingleRigidPrim:
        return self._base

    @property
    def link_left_arm_6(self) -> SingleRigidPrim:
        return self._link_left_arm_6

    @property
    def link_right_arm_6(self) -> SingleRigidPrim:
        return self._link_right_arm_6


    # ------------------------------------------------------------------
    # PhysX property helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_float_attr(prim, attr_name: str, value: float) -> None:
        attr = prim.GetAttribute(attr_name)
        if not attr:
            attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
        attr.Set(float(value))

    def _apply_joint_motor_model(self, prim, joint_api, drive_type: str, profile: dict) -> None:
        joint_api.CreateJointFrictionAttr().Set(profile["joint_friction"])
        prim.ApplyAPI("PhysxJointAxisAPI", drive_type)

        viscous = profile["viscous_nm_s_per_rad"]
        if drive_type == "angular":
            viscous = np.deg2rad(viscous)

        prefix = f"physxJointAxis:{drive_type}"
        # Author armature on the axis API consumed by PhysX.
        self._set_float_attr(prim, f"{prefix}:armature", profile["armature"])
        self._set_float_attr(prim, f"{prefix}:staticFrictionEffort", profile["static_effort"])
        self._set_float_attr(prim, f"{prefix}:dynamicFrictionEffort", profile["dynamic_effort"])
        self._set_float_attr(prim, f"{prefix}:viscousFrictionCoefficient", viscous)

    def _is_mecanum_roller_revolute_joint(self, prim) -> bool:
        pattern = re.compile(f"^{MECANUM_ROLLER_REVOLUTE_JOINT_PATH_EXPR}$")
        return self.robot_model == "m" and pattern.fullmatch(str(prim.GetPath())) is not None

    def _configure_mecanum_roller_joint(self, prim, joint_api) -> None:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateStiffnessAttr().Set(MECANUM_ROLLER_DRIVE_STIFFNESS)
        drive.CreateDampingAttr().Set(MECANUM_ROLLER_DRIVE_DAMPING)
        drive.CreateTargetVelocityAttr(MECANUM_ROLLER_TARGET_VELOCITY)

        joint_api.CreateMaxJointVelocityAttr().Set(1000.0)
        joint_api.CreateJointFrictionAttr().Set(0.0)
        joint_api.CreateArmatureAttr().Set(0.005)

    def _apply_physics_material_to_geometry(
        self,
        stage,
        prim_paths_expr: str,
        geometry_name: str,
        physics_material: PhysicsMaterial,
    ) -> bool:
        """Apply a physics material to every collision geometry matching a regex."""
        if not prim_paths_expr:
            return False

        pattern = re.compile(f"^{prim_paths_expr}$")
        match_count = sum(
            1 for prim in stage.Traverse() if pattern.fullmatch(str(prim.GetPath()))
        )
        if match_count == 0:
            carb.log_warn(f"[RBY1Robot] No geometry prims matched: {prim_paths_expr}")
            return False

        geometry_prims = GeometryPrim(
            prim_paths_expr=prim_paths_expr,
            name=geometry_name,
            collisions=np.full(match_count, True, dtype=bool),
        )
        geometry_prims.apply_physics_materials(physics_material)
        return True

    # ------------------------------------------------------------------
    # Public configuration entry points
    # ------------------------------------------------------------------

    def set_properties(self, stage) -> None:
        """Apply default PhysX properties to every rigid body in the stage."""
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Xform":
                continue
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                continue
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPath())
            if not rb_api:
                continue
            rb_api.CreateDisableGravityAttr().Set(False)
            rb_api.CreateRetainAccelerationsAttr().Set(False)
            rb_api.CreateLinearDampingAttr().Set(0.0)
            rb_api.CreateAngularDampingAttr().Set(0.0)
            rb_api.CreateMaxLinearVelocityAttr().Set(200.0)
            rb_api.CreateMaxAngularVelocityAttr().Set(1000.0)

        # Apply a high-friction physics material to the wheel collision geometry.
        wheel_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/wheel_material",
            dynamic_friction=1.2,
            static_friction=1.2,
            restitution=0.0,
        )
        self._apply_physics_material_to_geometry(
            stage=stage,
            prim_paths_expr=WHEEL_COLLISION_PRIM_PATH_EXPRS[self.robot_model],
            geometry_name="wheel_collisions",
            physics_material=wheel_material,
        )

        # Model A has extra base collision geometry that needs a frictionless material.
        if self.robot_model == "a" and MODEL_A_EXTRA_GEOMETRY_PRIM_PATH_EXPR:
            model_a_extra_material = PhysicsMaterial(**MODEL_A_EXTRA_GEOMETRY_MATERIAL_CONFIG)
            self._apply_physics_material_to_geometry(
                stage=stage,
                prim_paths_expr=MODEL_A_EXTRA_GEOMETRY_PRIM_PATH_EXPR,
                geometry_name="model_a_extra_geometry",
                physics_material=model_a_extra_material,
            )

    def set_joint_properties(self, stage, stiffness: float = 5e4, damping: float = 1e2) -> None:
        """Apply PhysX/Drive properties to all revolute joints in the stage."""
        for prim in stage.Traverse():
            type_name = prim.GetTypeName()
            if type_name == "PhysicsRevoluteJoint":
                self._configure_revolute_joint(prim)

    def _configure_revolute_joint(self, prim) -> None:
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            carb.log_warn(f"[RBY1Robot] Failed to retrieve RevoluteJoint at: {prim.GetPath()}")
            return

        joint.CreateBreakForceAttr().Set(float("inf"))
        joint.CreateBreakTorqueAttr().Set(float("inf"))

        joint_api = PhysxSchema.PhysxJointAPI(prim)
        profile = get_profile_for_joint(prim.GetName())

        if self._is_mecanum_roller_revolute_joint(prim):
            self._configure_mecanum_roller_joint(prim, joint_api)
            return

        if prim.HasAPI(UsdPhysics.DriveAPI):
            drive = UsdPhysics.DriveAPI(prim, "angular")
            drive.CreateStiffnessAttr().Set(0.0)
            drive.CreateDampingAttr().Set(0.0)
            self._apply_joint_motor_model(prim, joint_api, "angular", profile)
        else:
            joint_api.CreateMaxJointVelocityAttr().Set(200.0)
            joint_api.CreateJointFrictionAttr().Set(0.0)
            joint_api.CreateArmatureAttr().Set(0.1)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_joint_properties(self, stage) -> None:
        """Print USD joint properties, using the active axis API for motor parameters."""
        for prim in stage.Traverse():
            joint_path = prim.GetPath()
            type_name = prim.GetTypeName()
            if type_name != "PhysicsRevoluteJoint":
                continue

            print(f"\n[INFO] Checking revolute joint: {joint_path}")
            joint = UsdPhysics.RevoluteJoint(prim)
            print(f"  [Joint]    break_force        = {joint.GetBreakForceAttr().Get()}")
            print(f"  [Joint]    break_torque       = {joint.GetBreakTorqueAttr().Get()}")
            joint_api = PhysxSchema.PhysxJointAPI(prim)
            print(f"  [Joint]    max_joint_velocity = {joint_api.GetMaxJointVelocityAttr().Get()}")
            print(f"  [Joint]    joint_friction     = {joint_api.GetJointFrictionAttr().Get()}")
            axis_prefix = "physxJointAxis:angular"
            if "PhysxJointAxisAPI:angular" in prim.GetAppliedSchemas():
                armature = prim.GetAttribute(f"{axis_prefix}:armature")
                static_friction = prim.GetAttribute(f"{axis_prefix}:staticFrictionEffort")
                dynamic_friction = prim.GetAttribute(f"{axis_prefix}:dynamicFrictionEffort")
                viscous_friction = prim.GetAttribute(f"{axis_prefix}:viscousFrictionCoefficient")
                print(f"  [Axis]     armature           = {armature.Get()} kg*m^2")
                print(f"  [Axis]     static_friction    = {static_friction.Get()} N*m")
                print(f"  [Axis]     dynamic_friction   = {dynamic_friction.Get()} N*m")
                viscous = viscous_friction.Get()
                print(f"  [Axis]     viscous_friction   = {viscous} N*m*s/deg (USD)")
                if viscous is not None:
                    print(f"  [Axis]     viscous_friction   = {np.rad2deg(viscous)} N*m*s/rad")
            else:
                # Passive joints without an axis API still use the legacy properties.
                print(f"  [Joint]    armature           = {joint_api.GetArmatureAttr().Get()} kg*m^2")
                print("  [Axis]     motor properties   = Not applied.")

            if prim.HasAPI(UsdPhysics.DriveAPI):
                drive = UsdPhysics.DriveAPI(prim, "angular")
                print(f"  [Drive]    stiffness          = {drive.GetStiffnessAttr().Get()}")
                print(f"  [Drive]    damping            = {drive.GetDampingAttr().Get()}")

    def debug_stage_prims(self, stage) -> None:
        """Traverse the stage and print each prim's type and applied APIs."""
        print("[INFO] Traversing stage and listing prim details...\n")
        for prim in stage.Traverse():
            print(f"[DEBUG] Prim: {prim.GetPath()}")
            print(f"        Type: {prim.GetTypeName()}")
            print(f"        Applied APIs: {prim.GetAppliedSchemas()}\n")
