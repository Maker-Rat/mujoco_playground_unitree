import numpy as np
import mujoco
import mujoco.viewer
import time
from joystick import Joystick  # Import your environment
from mujoco_playground import wrapper
from mujoco_playground import registry
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
import functools
import jax
from jax import numpy as jp
from brax.io import model
import keyboard
from mujoco_playground.config import locomotion_params


def Imitation_Model():
    env_name = "Go2JoystickFlatTerrain"
    env = registry.load(env_name)
    env_cfg = registry.get_default_config(env_name)

    ppo_params = locomotion_params.brax_ppo_config(env_name)

    ppo_training_params = dict(ppo_params)

    randomizer = registry.get_domain_randomizer(env_name)
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )

    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        randomization_fn=randomizer
    )

    make_inference_fn_new, _, _ = train_fn(
        environment=env,
        num_timesteps=0,
        eval_env=registry.load(env_name, config=env_cfg),
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    params_new = model.load_params("hexapod_PPO.npz")
    
    return jax.jit(make_inference_fn_new(params_new, deterministic=True))