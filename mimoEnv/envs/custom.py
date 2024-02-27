""" This module contains a simple experiment where MIMo is tasked with touching parts of his own body.

The scene is empty except for MIMo, who is sitting on the ground. The task is for MIMo to touch a randomized target
body part with his right arm. MIMo is fixed in the initial sitting position and can only move his right arm.
Sensory inputs consist of touch and proprioception. Proprioception uses the default settings, but touch excludes
several body parts and uses a lowered resolution to improve runtime.
The body part can be any of the geoms constituting MIMo.

MIMos initial position is constant in all episodes. The target body part is randomized. An episode is completed
successfully if MIMo touches the target body part with his right arm.

The reward structure consists of a large fixed reward for touching the right body part, a shaping reward for touching
another body part, depending on the distance between the contact and the target body part, and a penalty for each time
step.

The class with the environment is :class:`~mimoEnv.envs.selfbody.MIMoSelfBodyEnv` while the path to the scene XML is
defined in :data:`SELFBODY_XML`.
"""

import os
import numpy as np

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS, SCENE_DIRECTORY
import mimoEnv.utils as env_utils
from mimoActuation.actuation import SpringDamperModel


from mimoTouch.touch import TrimeshTouch, Touch
from mimoVision.vision import SimpleVision, Vision
from mimoVestibular.vestibular import SimpleVestibular, Vestibular
from mimoProprioception.proprio import SimpleProprioception, Proprioception
from mimoEnv.envs.selfbody import TOUCH_PARAMS
from mimoEnv.envs.dummy import MIMoDummyEnv


SELFBODY_XML = os.path.join(SCENE_DIRECTORY, "explore_scene.xml")

SITTING_POSITION = {
    "robot:hip_lean1": np.array([0.039088]), "robot:hip_rot1": np.array([0.113112]),
    "robot:hip_bend1": np.array([0.5323]), "robot:hip_lean2": np.array([0]), "robot:hip_rot2": np.array([0]),
    "robot:hip_bend2": np.array([0.5323]),
    "robot:head_swivel": np.array([0]), "robot:head_tilt": np.array([0]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    
    "robot:right_hip1": np.array([-1.51997]), "robot:right_hip2": np.array([-0.397578]),
    "robot:right_hip3": np.array([0.0976615]), "robot:right_knee": np.array([-1.85479]),
    "robot:right_foot1": np.array([-0.585865]), "robot:right_foot2": np.array([-0.358165]),
    "robot:right_foot3": np.array([0]), "robot:right_toes": np.array([0]),
    "robot:left_hip1": np.array([-1.23961]), "robot:left_hip2": np.array([-0.8901]),
    "robot:left_hip3": np.array([0.7156]), "robot:left_knee": np.array([-2.531]),
    "robot:left_foot1": np.array([-0.63562]), "robot:left_foot2": np.array([0.5411]),
    "robot:left_foot3": np.array([0.366514]), "robot:left_toes": np.array([0.24424]),
}
SITTING_POSITION_UNLOCK = {
    "robot:left_shoulder_horizontal": np.array([0.683242]), "robot:left_shoulder_ad_ab": np.array([0.3747]),
    "robot:left_shoulder_rotation": np.array([-0.62714]), "robot:left_elbow": np.array([-0.756016]),
    "robot:left_hand1": np.array([0.28278]), "robot:left_hand2": np.array([0]), "robot:left_hand3": np.array([0]),
    "robot:left_fingers": np.array([-0.461583]),
}
""" Initial position of MIMo. Specifies initial values for all joints.
We grabbed these values by posing MIMo using the MuJoCo simulate executable and the positional actuator file.
We need these not just for the initial position but also resetting the position (excluding the right arm) each step.

:meta hide-value:
"""

class MIMoCustomEnv(MIMoDummyEnv):
    """
    A sitting Dummy MIMo with customizeable sensor classes.
    """

    def __init__(self,
                 model_path=SELFBODY_XML,
                 initial_qpos=SITTING_POSITION,
                 frame_skip=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=SpringDamperModel,
                 goals_in_observation=True,
                 done_active=True,
                 proprio_class=SimpleProprioception,
                 touch_class=TrimeshTouch,
                 vision_class=SimpleVision,
                 vestibular_class=SimpleVestibular,
                 **kwargs,
                 ):
    

        self._ProprioClass = proprio_class
        self._TouchClass = touch_class
        self._VisionClass = vision_class
        self._VestibularClass = vestibular_class

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         **kwargs)
        

        env_utils.set_joint_qpos(self.model,
                                 self.data,
                                 "mimo_location",
                                 np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029]))
        #  "mimo_location": np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029]),
        for joint_name in SITTING_POSITION:
            env_utils.lock_joint(self.model, joint_name, joint_angle=SITTING_POSITION[joint_name][0])
        for joint_name in SITTING_POSITION_UNLOCK:
            env_utils.set_joint_locking_angle(self.model, joint_name, angle=SITTING_POSITION_UNLOCK[joint_name][0])
        # Let sim settle for a few timesteps to allow weld and locks to settle
        self.do_simulation(np.zeros(self.action_space.shape), 25)
        self.init_sitting_qpos = self.data.qpos.copy()
        


    
    def proprio_setup(self, proprio_params):
        """ Perform the setup and initialization of the proprioceptive system.

        This should be overridden if you want to use another implementation!

        Args:
            proprio_params (dict): The parameter dictionary.
        """
        self.proprioception = self._ProprioClass(self, proprio_params)

    def touch_setup(self, touch_params):
        """ Perform the setup and initialization of the touch system.

        This should be overridden if you want to use another implementation!

        Args:
            touch_params (dict): The parameter dictionary.
        """
        self.touch = self._TouchClass(self, touch_params)

    def vision_setup(self, vision_params):
        """ Perform the setup and initialization of the vision system.

        This should be overridden if you want to use another implementation!

        Args:
            vision_params (dict): The parameter dictionary.
        """
        self.vision = self._VisionClass(self, vision_params)

    def vestibular_setup(self, vestibular_params):
        """ Perform the setup and initialization of the vestibular system.

        This should be overridden if you want to use another implementation!

        Args:
            vestibular_params (dict): The parameter dictionary.
        """
        self.vestibular = self._VestibularClass(self, vestibular_params)