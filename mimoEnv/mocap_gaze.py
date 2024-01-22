""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""

import gymnasium as gym
import time
import numpy as np
import mimoEnv
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY, DEFAULT_VISION_PARAMS
import os

SELFBODY_XML = os.path.join(SCENE_DIRECTORY, "selfbody_scene_mocapgaze.xml")



def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """

    env = gym.make("MIMoMocapGaze-v0", model_path=SELFBODY_XML, vision_params=DEFAULT_VISION_PARAMS)

    max_steps = 20000

    _ = env.reset()

    start = time.time()
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        # if done or trunc:
        #     env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()


if __name__ == "__main__":
    main()
