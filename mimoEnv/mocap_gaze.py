""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""

import gymnasium as gym
import time
import numpy as np
import mimoEnv
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY, DEFAULT_VISION_PARAMS
from mimoVision.vision import LogPolarVision
import os
from cv2 import (imshow, waitKey, warpPolar, 
                 cvtColor, COLOR_RGB2BGR,
                 WARP_INVERSE_MAP, WARP_POLAR_LOG, WARP_FILL_OUTLIERS,
                 INTER_NEAREST)

SELFBODY_XML = os.path.join(SCENE_DIRECTORY, "selfbody_scene_mocapgaze.xml")

LOGPOLAR_VISION_PARAMS = {
    "eye_left":     {"width": 256, "height": 256, 'maxRadius': 128}, # , 'logFraction': 0.125
    "eye_right":    {"width": 256, "height": 256, 'maxRadius': 128},
}

def show_eye(obs, left=True):
    eye = 'left' if left else 'right'
    imshow(f'eye_{eye}_lp', cvtColor(obs[f'eye_{eye}'], COLOR_RGB2BGR))
    imshow(f'eye_{eye}_cart', 
        cvtColor(warpPolar(obs[f'eye_{eye}'], (256, 256), (128, 128), 128, 
                WARP_INVERSE_MAP+WARP_POLAR_LOG+WARP_FILL_OUTLIERS+INTER_NEAREST), COLOR_RGB2BGR)
    )

def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """

    env = gym.make("MIMoCustom-v0", model_path=SELFBODY_XML, vision_class=LogPolarVision, vision_params=LOGPOLAR_VISION_PARAMS)

    max_steps = 20000

    _ = env.reset()


    start = time.time()
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        env.render()

        show_eye(obs, left=True)
        show_eye(obs, left=False)
        waitKey(1)
        # if done or trunc:
        #     env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()


if __name__ == "__main__":
    main()
