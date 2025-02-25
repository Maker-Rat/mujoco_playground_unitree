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
"""Defines Spiderbot Hexapod constants."""

from etils import epath

from mujoco_playground._src import mjx_env

ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "spiderbot"
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "Spiderbot_V2_full_position.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  return {
      "flat_terrain": FEET_ONLY_FLAT_TERRAIN_XML,
      "rough_terrain": FEET_ONLY_ROUGH_TERRAIN_XML,
  }[task_name]


FEET_SITES = [
    "foot_site_1",
    "foot_site_2",
    "foot_site_3",
    "foot_site_4",
    "foot_site_5",
    "foot_site_6",
]

FEET_GEOMS = [
    "foot_1",
    "foot_2",
    "foot_3",
    "foot_4",
    "foot_5",
    "foot_6",
]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "base_link"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
# ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
