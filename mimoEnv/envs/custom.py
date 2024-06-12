"""
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
from mimoEnv.envs.dummy import MIMoDummyEnv


SELFBODY_XML = os.path.join(SCENE_DIRECTORY, "explore_scene.xml")


TOUCH_PARAMS = {
    "scales": {
        "upper_body": 0.1,
        "head": 0.1,
        "left_upper_arm": 0.1,
        "left_lower_arm": 0.1,
        "right_hand": 0.01,
        "left_hand": 0.01,
        "right_fingers": 0.01,
        "left_fingers": 0.01
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" List of possible target bodies."""


TOUCH_PARAMS_V2 = {
    "scales": {
        "upper_body": 0.1,
        "head": 0.1,
        "left_upper_arm": 0.1,
        "left_lower_arm": 0.1,
        "right_hand": 0.01,
        "left_hand": 0.01,

        "left_ffknuckle": 0.1,
        "left_ffmiddle": 0.1,
        "left_ffdistal": 0.01,
        "left_mfknuckle": 0.1,
        "left_mfmiddle": 0.1,
        "left_mfdistal": 0.01,
        "left_rfknuckle": 0.1,
        "left_rfmiddle": 0.1,
        "left_rfdistal": 0.01,
        "left_lfmetacarpal": 0.1,
        "left_lfknuckle": 0.1,
        "left_lfmiddle": 0.1,
        "left_lfdistal": 0.01,
        "left_thbase": 0.1,
        "left_thhub": 0.1,
        "left_thdistal": 0.01,

        "right_ffknuckle": 0.1,
        "right_ffmiddle": 0.1,
        "right_ffdistal": 0.01,
        "right_mfknuckle": 0.1,
        "right_mfmiddle": 0.1,
        "right_mfdistal": 0.01,
        "right_rfknuckle": 0.1,
        "right_rfmiddle": 0.1,
        "right_rfdistal": 0.01,
        "right_lfmetacarpal": 0.1,
        "right_lfknuckle": 0.1,
        "right_lfmiddle": 0.1,
        "right_lfdistal": 0.01,
        "right_thbase": 0.1,
        "right_thhub": 0.1,
        "right_thdistal": 0.01,

    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" List of possible target bodies."""


SITTING_POSITION = {
    "robot:hip_bend1": np.array([0.533]),
    "robot:hip_lean2": np.array([0.0272]),
    "robot:hip_rot2": np.array([-0.101]),
    "robot:hip_bend2": np.array([0.519]),
    
    "robot:right_hip1": np.array([-1.39]), "robot:right_hip2": np.array([-0.891]),
    "robot:right_hip3": np.array([0.546]), "robot:right_knee": np.array([-2.07]),
    "robot:right_foot1": np.array([-0.496]), "robot:right_foot2": np.array([0.01]),
    "robot:right_foot3": np.array([0.048]), "robot:right_toes": np.array([0.01]),

    "robot:left_hip1": np.array([-0.725]), "robot:left_hip2": np.array([-0.006]),
    "robot:left_hip3": np.array([0.7156]), "robot:left_knee": np.array([-0.352]),
    "robot:left_foot1": np.array([-0.468]), "robot:left_foot2": np.array([0.03]),
    "robot:left_foot3": np.array([-0.033]), "robot:left_toes": np.array([0.0]),
    
    #"robot:left_shoulder_horizontal": np.array([1.2]), "robot:left_shoulder_ad_ab": np.array([0.8]),
    #"robot:left_shoulder_rotation": np.array([-1.0]), 
    #"robot:left_elbow": np.array([-0.8]),
}
SITTING_POSITION_UNLOCK = {

    "robot:head_swivel": np.array([0.385]), "robot:head_tilt": np.array([0.219]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    
    "robot:hip_lean1": np.array([0.024]), 
    "robot:hip_rot1": np.array([-0.124]),

    "robot:right_shoulder_horizontal": np.array([1.32]), "robot:right_shoulder_ad_ab": np.array([0.421]),
    "robot:right_shoulder_rotation": np.array([-1.11]), "robot:right_elbow": np.array([-0.756016]),
    "robot:right_hand1": np.array([0.157]), "robot:right_hand2": np.array([-0.698]), "robot:right_hand3": np.array([-0.211]),
    "robot:right_fingers": np.array([-0.698]),

    #"robot:left_shoulder_horizontal": np.array([1.0]), "robot:left_shoulder_ad_ab": np.array([0.4]),
    #"robot:left_shoulder_rotation": np.array([0.0]), 
    #"robot:left_elbow": np.array([-0.349]),
    "robot:left_hand1": np.array([-0.349]), "robot:left_hand2": np.array([0]), "robot:left_hand3": np.array([0]),
    "robot:left_fingers": np.array([-0.698]),
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
        for joint_name in initial_qpos:
            env_utils.lock_joint(self.model, joint_name, joint_angle=initial_qpos[joint_name][0])
        # for joint_name in SITTING_POSITION_UNLOCK:
        #     env_utils.set_joint_locking_angle(self.model, joint_name, angle=SITTING_POSITION_UNLOCK[joint_name][0])
        # Let sim settle for a few timesteps to allow weld and locks to settle
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

    