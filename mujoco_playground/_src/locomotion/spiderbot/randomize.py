# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain randomization for the Spiderbot hexapod environment."""

import jax
import jax.numpy as jp
from mujoco import mjx

# These IDs should be verified against your actual spiderbot model
FLOOR_GEOM_ID = 0  # Verify this matches your floor geom
TORSO_BODY_ID = 1  # Verify this matches your torso body


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor friction: =U(0.3, 1.2) - wider range for more diverse terrain
    rng, key = jax.random.split(rng)
    geom_friction = model.geom_friction.at[FLOOR_GEOM_ID, 0].set(
        jax.random.uniform(key, minval=0.3, maxval=1.2)
    )

    # Scale actuator friction for the 12 actuated joints: *U(0.8, 1.2)
    # Spiderbot has actuated joints at DOF indices after the 6-DOF base
    # Based on your joystick.py, actuated DOFs are: [6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27]
    actuated_dof_indices = jp.array([6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27])
    
    rng, key = jax.random.split(rng)
    friction_multipliers = jax.random.uniform(
        key, shape=(12,), minval=0.8, maxval=1.2
    )
    dof_frictionloss = model.dof_frictionloss
    for i, idx in enumerate(actuated_dof_indices):
      dof_frictionloss = dof_frictionloss.at[idx].set(
          model.dof_frictionloss[idx] * friction_multipliers[i]
      )

    # Scale armature for actuated joints: *U(0.95, 1.1)
    rng, key = jax.random.split(rng)
    armature_multipliers = jax.random.uniform(
        key, shape=(12,), minval=0.95, maxval=1.1
    )
    dof_armature = model.dof_armature
    for i, idx in enumerate(actuated_dof_indices):
      dof_armature = dof_armature.at[idx].set(
          model.dof_armature[idx] * armature_multipliers[i]
      )

    # Jitter center of mass position: +U(-0.03, 0.03) - smaller for hexapod stability
    rng, key = jax.random.split(rng)
    dpos = jax.random.uniform(key, (3,), minval=-0.03, maxval=0.03)
    body_ipos = model.body_ipos.at[TORSO_BODY_ID].set(
        model.body_ipos[TORSO_BODY_ID] + dpos
    )

    # Scale all link masses: *U(0.9, 1.1)
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.9, maxval=1.1
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-0.5, 0.5) - smaller range for hexapod
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-0.5, maxval=0.5)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Jitter qpos0 for actuated joints only: +U(-0.03, 0.03)
    # Spiderbot has 30 qpos elements, actuated joints at [8,9,12,13,16,17,20,21,24,25,28,29]
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    actuated_qpos_indices = jp.array([8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29])
    joint_noise = jax.random.uniform(key, shape=(12,), minval=-0.03, maxval=0.03)
    
    for i, idx in enumerate(actuated_qpos_indices):
      qpos0 = qpos0.at[idx].set(qpos0[idx] + joint_noise[i])

    return (
        geom_friction,
        body_ipos,
        body_mass,
        qpos0,
        dof_frictionloss,
        dof_armature,
    )

  (
      friction,
      body_ipos,
      body_mass,
      qpos0,
      dof_frictionloss,
      dof_armature,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_ipos": 0,
      "body_mass": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
  })

  model = model.tree_replace({
      "geom_friction": friction,
      "body_ipos": body_ipos,
      "body_mass": body_mass,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
  })

  return model, in_axes


def verify_model_structure(model):
  """Helper function to verify the model structure and print key indices."""
  print(f"Model has {model.nbody} bodies")
  print(f"Model has {model.ngeom} geoms") 
  print(f"Model has {model.nq} qpos elements")
  print(f"Model has {model.nv} qvel elements")
  print(f"Model has {model.nu} actuators")
  print(f"DOF frictionloss shape: {model.dof_frictionloss.shape}")
  print(f"DOF armature shape: {model.dof_armature.shape}")
  print(f"qpos0 shape: {model.qpos0.shape}")
  print("First 10 qpos0 values:", model.qpos0[:10])
  print("Actuated qpos indices should be [8,9,12,13,16,17,20,21,24,25,28,29]")
