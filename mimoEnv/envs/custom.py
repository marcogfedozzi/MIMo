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
from mimoEnv.envs.selfbody import TOUCH_PARAMS, SITTING_POSITION, SELFBODY_XML
from mimoEnv.envs.dummy import MIMoDummyEnv



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