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
"""Stairs task for Hexapod."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from jax import debug
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.spiderbot import base as spiderbot_base
from mujoco_playground._src.locomotion.spiderbot import spiderbot_constants as consts


# Updated reward configuration
def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.08,
        sim_dt=0.004,
        episode_length=500,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),

        reward_config=config_dict.create(
            scales=config_dict.create(
                # Stairs-specific rewards - focused on global movement
                forward_progress=2.5,      # Strong reward for global forward movement
                y_deviation=-1.0,          # Strong penalty for sideways drift
                height_progress=2.5,       # Reward climbing higher
                
                # Basic locomotion - moderate weights  
                orientation=-0.5,          # Keep upright
                pose=0.02,                 # Stay near default pose
                
                # Standard penalties
                termination=-1.0,
                torques=-0.0002,
                action_rate=-0.02,
            ),
            tracking_sigma=0.25,
            max_foot_height=0.125,
            action_smoothness=-0.0035,
        ),

        pert_config=config_dict.create(
            enable=False,
            velocity_kick=[0.0, 3.0],
            kick_durations=[0.05, 0.2],
            kick_wait_times=[1.0, 3.0],
        ),
        impl="jax",
        nconmax=4 * 8192,
        njmax=40,
    )


class StairsClimbing(spiderbot_base.SpiderbotEnv): 
  def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
    if task.startswith("rough"):
      config.nconmax = 100 * 8192
      config.njmax = 12 + 100 * 4
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()


  def _actuator_joint_angles(self, qpos: jax.Array) -> jax.Array:
    indices = jax.numpy.array([8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29])
    return qpos[indices]

  def _actuator_joint_vels(self, qvel: jax.Array) -> jax.Array:
    indices = jax.numpy.array([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21])
    return qvel[indices]

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._actuator_joint_angles(self._mj_model.keyframe("home").qpos))

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    
    # FIX: Convert to JAX array for indexing
    actuated_indices = jp.array([0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21])
    self._lowers = self._lowers[actuated_indices]
    self._uppers = self._uppers[actuated_indices]

    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = jp.array([
        self._mj_model.site(name).id for name in consts.FEET_SITES
    ])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = jp.array([
        self._mj_model.geom(name).id for name in consts.FEET_GEOMS
    ])

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=0.0, maxval=0.0)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=0.0, maxval=0.0)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    # FIX: Use only the actuated joint positions for ctrl
    actuated_joint_positions = self._actuator_joint_angles(qpos)
    
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=actuated_joint_positions,  # Changed from qpos[7:] to match actuated joints
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Rest of the method remains the same...
    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(6),  # Changed to 6 for spiderbot
        "last_contact": jp.zeros(6, dtype=bool),  # Changed to 6 for spiderbot
        "swing_peak": jp.zeros(6),  # Changed to 6 for spiderbot
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
 
    state = mjx_env.State(data, obs, reward, done, metrics, info)
    
    return state
  
  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    sensor_addresses = jp.array([self._mj_model.sensor_adr[int(sid)] for sid in self._feet_floor_found_sensor])
    contact = data.sensordata[sensor_addresses] > 0

    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )

    rewards = {
        k: jp.clip(v * self._config.reward_config.scales[k], -10.0, 10.0) 
        for k, v in rewards.items()
    }
    
    reward = jp.clip(sum(rewards.values()) * self.dt, -5.0, 5.0)
    
    # And add an explicit print right before returning
    # debug.print("Final reward being returned: {}", reward)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)

    return state

    # Enhanced termination to prevent infinite episodes
  def _get_termination(self, data: mjx.Data) -> jax.Array:
        # Fall termination
        fall_termination = self.get_upvector(data)[-1] < -0.1
        
        # Boundary termination - robot went too far off course
        x_pos = data.qpos[0]
        y_pos = data.qpos[1]
        boundary_termination = (jp.abs(y_pos) > 2.0) | (x_pos < -2.0)
        
        # Optional: Success termination when reaching top
        # success_termination = data.qpos[2] > 1.5  # Adjust based on stair height
        
        return fall_termination | boundary_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = self._actuator_joint_angles(qpos=data.qpos)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = self._actuator_joint_vels(data.qvel[6:])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    # Adjusted state vector - assuming 12 joints for spiderbot (2 per leg × 6 legs)
    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12 for spiderbot # noisy_joint_vel,  # 12 for spiderbot
        info["last_act"],  # 12 for spiderbot
    ])

    # accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    # Adjusted privileged state for spiderbot
    privileged_state = jp.hstack([
        state,
        gyro,  # 3 # accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        joint_angles - self._default_pose,  # 12for spiderbot
        joint_vel,  # 12 for spiderbot
        data.actuator_force,  # 12 for spiderbot
        info["last_contact"],  # 6 for spiderbot
        feet_vel,  # 6*3 for spiderbot
        info["feet_air_time"],  # 6 for spiderbot
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
    # Compute basic rewards only, with NaN protection
    rewards = {
        "forward_progress": self._reward_forward_progress(data, info),
        "y_deviation": self._cost_y_deviation(data),
        "height_progress": self._reward_height_progress(data),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "pose": self._reward_pose(self._actuator_joint_angles(data.qpos)),
        "termination": self._cost_termination(done),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
    }
    
    # Clip each reward to avoid extreme values and NaNs
    rewards = {k: jp.clip(jp.where(jp.isnan(v), 0.0, v), -10.0, 10.0) for k, v in rewards.items()}
    return rewards

  # Tracking rewards.

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        """Penalize non flat base orientation"""
        return jp.sum(jp.square(torso_zaxis[:2]))

  def _reward_pose(self, qpos: jax.Array) -> jax.Array:
        """Stay close to the default pose."""
        weight = jp.array([1.0, 1.0] * 6)  # 12 joints for hexapod
        pose_error = jp.sum(jp.square(qpos - self._default_pose) * weight)
        return jp.exp(-pose_error / (1.0 + 1e-8))

  def _cost_termination(self, done: jax.Array) -> jax.Array:
        """Penalize early termination."""
        return jp.where(jp.isnan(done), 1.0, done)

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
        """Penalize torques."""
        torque_sq_sum = jp.sum(jp.square(torques))
        return jp.sqrt(torque_sq_sum + 1e-8) + jp.sum(jp.abs(torques))

  def _cost_action_rate(self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array) -> jax.Array:
        """Penalize action rate changes."""
        del last_last_act  # Unused
        return jp.sum(jp.square(act - last_act))

  def _reward_forward_progress(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    """Reward forward movement in global X direction."""
    # Use global velocity in X direction (world frame)
    global_velocity = self.get_global_linvel(data)
    forward_vel = global_velocity[0]  # Global X velocity
    
    # Reward forward movement (positive X velocity)
    forward_reward = jp.clip(forward_vel * 2.0, 0.0, 5.0)
    
    return jp.clip(jp.where(jp.isnan(forward_reward), 0.0, forward_reward), 0.0, 5.0)

  def _reward_height_progress(self, data: mjx.Data) -> jax.Array:
    """Reward climbing higher (global Z position)."""
    current_height = data.qpos[2]
    
    # Reward height gain above base height (0.3m is roughly the robot's initial height)
    height_reward = jp.clip((current_height - 0.3) * 5.0, 0.0, 3.0)
    
    return jp.clip(jp.where(jp.isnan(height_reward), 0.0, height_reward), 0.0, 3.0)

  def _cost_y_deviation(self, data: mjx.Data) -> jax.Array:
        """Penalize deviation from stair centerline in global Y."""
        y_pos = data.qpos[1]
        return jp.square(y_pos)

    # Perturbation and command sampling.

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )
