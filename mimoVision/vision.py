""" This module defines the vision interface and provides a simple implementation.

The interface is defined as an abstract class in :class:`~mimoVision.vision.Vision`.
A simple implementation treating each eye as a single camera is in :class:`~mimoVision.vision.SimpleVision`.

"""
import mujoco
import os
import matplotlib
from typing import Dict
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from mujoco import MjrRect
from typing import Callable, Dict

from numpy import ndarray
from functools import partial

from numpy.typing import NDArray
from math import floor

class Vision:
    """ Abstract base class for vision.

    This class defines the functions that all implementing classes must provide.
    The constructor takes two arguments: `env`, which is the environment we are working with, and `camera_parameters`,
    which can be used to supply implementation specific parameters.

    There is only one function that implementations must provide:
    :meth:`.get_vision_obs` should produce the vision outputs that will be returned to the environment. These outputs
    should also be stored in :attr:`.sensor_outputs`.

    Attributes:
        env (MujocoEnv): The environment to which this module will be attached
        camera_parameters: A dictionary containing the configuration. The exact from will depend on the specific
            implementation.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. Shape will depend on the specific
            implementation. This should be populated by :meth:`.get_vision_obs`

    """
    def __init__(self, env, camera_parameters):
        self.env = env
        self.camera_parameters = camera_parameters
        self.sensor_outputs = {}

    def get_vision_obs(self):
        """ Produces the current vision output.

        This function should perform the whole sensory pipeline and return the vision output as defined in
        :attr:`.camera_parameters`. Exact return value and functionality will depend on the implementation, but should
        always be a dictionary containing images as values.

        Returns:
            Dict[str, np.ndarray]: A dictionary of numpy arrays with the output images.

        """
        raise NotImplementedError


class SimpleVision(Vision):
    """ A simple vision system with one camera for each output.

    The output is simply one RGB image for each camera in the configuration. The constructor takes two arguments: `env`,
    which is the environment we are working with, and `camera_parameters`, which provides the configuration for the
    vision system.
    The parameter `camera_parameters` should be a dictionary with the following structure::

        {
            'camera_name': {'width': width, 'height': height},
            'other_camera_name': {'width': width, 'height': height},
        }

    The default MIMo model has two cameras, one in each eye, named `eye_left` and `eye_right`. Note that the cameras in
    the dictionary must exist in the scene xml or errors will occur!

    Attributes:
        env: The environment to which this module should be attached
        camera_parameters: A dictionary containing the configuration.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. This is populated by
            :meth:`.get_vision_obs`

    """
    def __init__(self, env, camera_parameters):
        """ Constructor.

        Args:
            env: The environment to which this module should be attached
            camera_parameters: A dictionary containing the configuration.

        """
        super().__init__(env, camera_parameters)
        self._viewports = {}
        for camera in camera_parameters:
            viewport = MjrRect(0, 0, camera_parameters[camera]["width"], camera_parameters[camera]["height"])
            self._viewports[camera] = viewport

    def get_vision_obs(self):
        """ Produces the current vision output.

        This function renders each camera with the resolution as defined in :attr:`.camera_parameters` using an
        off-screen render context. The images are also stored in :attr:`.sensor_outputs` under the name of the
        associated camera.

        Returns:
            Dict[str, np.ndarray]: A dictionary with camera names as keys and the corresponding rendered images as
            values.
        """
        # We have to cycle render modes, camera names, camera ids and viewport sizes
        old_mode = self.env.render_mode
        old_cam_name = self.env.camera_name
        old_cam_id = self.env.camera_id

        # Ensure that viewer is initialized
        if not self.env.mujoco_renderer._viewers.get("rgb_array"):
            self.env.mujoco_renderer.render(render_mode="rgb_array")

        rgb_viewer = self.env.mujoco_renderer._viewers["rgb_array"]
        old_viewport = rgb_viewer.viewport

        self.env.render_mode = "rgb_array"
        self.env.camera_id = None

        imgs = {}
        for camera in self.camera_parameters:
            self.env.camera_name = camera
            rgb_viewer.viewport = self._viewports[camera]
            imgs[camera] = self.env.render()
        self.sensor_outputs = imgs

        self.env.render_mode = old_mode
        self.env.camera_name = old_cam_name
        self.env.camera_id = old_cam_id
        rgb_viewer.viewport = old_viewport

        return imgs

    def save_obs_to_file(self, directory, suffix=""):
        """ Saves the output images to file.

        Everytime this function is called all images in :attr:`.sensor_outputs` are saved to separate files in
        `directory`. The filename is determined by the camera name and `suffix`. Saving large images takes a long time!

        Args:
            directory (str): The output directory. It will be created if it does not already exist.
            suffix (str): Optional file suffix. Useful for a step counter. Empty by default.
        """
        os.makedirs(directory, exist_ok=True)
        if self.sensor_outputs is None or len(self.sensor_outputs) == 0:
            raise RuntimeWarning("No image observations to save!")
        for camera_name in self.sensor_outputs:
            file_name = camera_name + suffix + ".png"
            matplotlib.image.imsave(os.path.join(
                directory, file_name), self.sensor_outputs[camera_name], vmin=0.0, vmax=1.0)
        
    def get_3D_point(self, x, y, camera_name):
        """
        Returns the 3D point in the world coordinates corresponding to the pixel (x, y) in the image of the camera with name camera_name.
        """
        
        old_mode = self.env.render_mode
        old_cam_name = self.env.camera_name
        old_cam_id = self.env.camera_id

        if not self.env.mujoco_renderer._viewers.get("rgb_array"):
            self.env.mujoco_renderer.render(render_mode="rgb_array")

        rgb_viewer = self.env.mujoco_renderer._viewers["rgb_array"]

        old_viewport = rgb_viewer.viewport

        rgb_viewer.viewport = self._viewports[camera_name]

        self.env.render_mode = "rgb_array"
        self.env.camera_id = None
        self.env.camera_name = camera_name

        w = rgb_viewer.viewport.width
        h = rgb_viewer.viewport.height

        y = h-y # move center of the image to the bottom left corner

        point  = np.zeros(3, dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)

        selid = mujoco.mjv_select(self.env.model, self.env.data,  rgb_viewer.vopt,
            aspectratio=w/h, relx=x/w, rely=y/h,
            scn=rgb_viewer.scn, selpnt=point, geomid=geomid, skinid=skinid,
        )
        
        self.env.render_mode = old_mode
        self.env.camera_name = old_cam_name
        self.env.camera_id = old_cam_id
        rgb_viewer.viewport = old_viewport

        return selid, point

    def get_2d_from_3d(self, point: NDArray, camera_name: str):

        old_mode = self.env.render_mode
        old_cam_name = self.env.camera_name
        old_cam_id = self.env.camera_id

        if not self.env.mujoco_renderer._viewers.get("rgb_array"):
            self.env.mujoco_renderer.render(render_mode="rgb_array")

        rgb_viewer = self.env.mujoco_renderer._viewers["rgb_array"]

        old_viewport = rgb_viewer.viewport

        rgb_viewer.viewport = self._viewports[camera_name]

        self.env.render_mode = "rgb_array"
        self.env.camera_id = self.env.model.camera(camera_name).id
        self.env.camera_name = camera_name

        point_homogeneus = np.ones(4, dtype=np.float64)
        point_homogeneus[:3] = point

        self.env.mujoco_renderer.render(render_mode="rgb_array", camera_name=camera_name)
        m = self._compute_camera_matrix()
        xs, ys, s = m @ point_homogeneus
        x = xs / s
        y = ys / s
        
        self.env.render_mode = old_mode
        self.env.camera_name = old_cam_name
        self.env.camera_id = old_cam_id
        rgb_viewer.viewport = old_viewport


        

        return np.array([floor(x), floor(y)], dtype=np.int32)

    
    def _compute_camera_matrix(self):
        """Returns the 3x4 camera matrix."""
        # FROM: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=sDYwClpxaxab
        # If the camera is a 'free' camera, we get its position and orientation
        # from the scene data structure. Note: we call `self.update()` in order to
        # ensure that the contents of `scene.camera` are correct.

        rgb_viewer = self.env.mujoco_renderer._viewers["rgb_array"]
        cam = rgb_viewer.scn.camera[self.env.camera_id]

        #mujoco.mjv_updateScene(
        #    m=self.env.model,
        #    d=self.env.data,
        #    opt=rgb_viewer.vopt,
        #    pert=mujoco.MjvPerturb(),
        #    cam=rgb_viewer.cam,
        #    catmask=mujoco.mjtCatBit.mjCAT_ALL.value,
        #    scn=rgb_viewer.scn,
        #)

        #self.env.mujoco_renderer.render(render_mode="rgb_array") # TEST: rerendering to ensure it's the correct camera... I guess?


        pos = cam.pos
        z   = -cam.forward
        y   = cam.up
        rot = np.vstack((np.cross(y, z), y, z))
        fov = self.env.model.cam_fovy[self.env.camera_id]

        h = self._viewports[self.env.camera_name].height
        w = self._viewports[self.env.camera_name].width

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * h / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (w - 1) / 2.0
        image[1, 2] = (h - 1) / 2.0
        return image @ focal @ rotation @ translation



class EditVision(SimpleVision):
    """A class that edits the iamges returned by the cameras before returning them to the environment.

    Args:
        env: The environment to which this module should be attached
        camera_parameters: A dictionary containing the configuration.
            it should contain the entries:
            - "warp_function" which is a function that takes an image and returns an edited version of it.
            - "warp_function_args" which is a dictionary of arguments to be passed to the function
        func: A function that takes an image and returns an edited version of it.
    """


    def __init__(self, env, camera_parameters): #, func: Callable[[np.ndarray, Dict], np.ndarray] = lambda x, _: x, func_args: Dict = {}):

        
        super().__init__(env, camera_parameters)

        self._image_warp_func = {}

        for camera in camera_parameters:
            self._image_warp_func[camera] = partial(camera_parameters[camera]["warp_function"], **camera_parameters[camera]["warp_function_args"])
        

        # check if func_args already specifies one set of args for each camera; otherwise copy the same args for each camera
    
    def _set_image_warp_func_args(self, **kwargs):
        """
        kwargs: a dictionary of arguments to be passed to the function
        expected either a list of arguments for each camera or a single set of arguments for all cameras
        
        e.g.

        _set_func_args(eye_left_args = {"arg1": 1, "arg2": 2}, eye_right_args = {"arg1": 3, "arg2": 4})
        or
        _set_func_args(arg1=1, arg2=2)
        """

        _c = True
        for camera in self._image_warp_func:
            _c = _c and camera in kwargs

        if not _c:
            for camera in self._image_warp_func: # one set of parameters shared by all cameras
                self._image_warp_func[camera] = self._image_warp_func[camera].update(**kwargs)
        else:
            for camera in self._image_warp_func: # one set of parameters for each cameras
                self._image_warp_func[camera] = self._image_warp_func[camera].update(**kwargs[camera])

        self._image_warp_func

    def get_vision_obs(self):
        imgs =  super().get_vision_obs()

        for camera, img in imgs.items():
            imgs[camera] = self._image_warp_func[camera](img)
        
        return imgs

class LogPolarVision(EditVision):
    """
    Like the SimpleVision class, but the image is transformed into
    logpolar coordinates before being returned.

    Optionally can return the cartesian reprojection of the logpolar image.
    """
    def __init__(self, env, camera_parameters):
        """
        Args:
            env: The environment to which this module should be attached
            camera_parameters: A dictionary containing the configuration.
                max_radius: The maximum radius of the logpolar image. If not provided,
                    the minimum of the width and height of the image will be used.
                return_cartesian: Whether to return the cartesian reprojection of the logpolar image.
        """

        # see  https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga49481ab24fdaa0ffa4d3e63d14c0d5e4

        try:
            cv.__version__
        except NameError:
            import cv2 as cv

        self.camera_transform_parameters = {}

        for camera, params in camera_parameters.items():
            max_radius = params.get("maxRadius", min(params["width"] / 2, params["height"] / 2))
            log_fraction = params.get("logFraction", 1)

            self.camera_transform_parameters[camera] = {
                'Klog': params["width"]*log_fraction / np.log(max_radius),
                'Kangle': params["height"] / (2*np.pi),
                'xc': params["width"] / 2,
                'yc': params["height"] / 2,
            }
            
            params.update(
                dict(
                    warp_function=cv.warpPolar,
                    warp_function_args=dict(
                        maxRadius=max_radius,
                        dsize=(int(params["width"]*log_fraction), params["height"]),
                        center=(params["width"] / 2, params["height"] / 2),
                        flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG
                    ) # arguments to be passed to the function
                )
            )
        
        

        super().__init__(env, camera_parameters)
    
    def get_3D_point(self, rho, phi, camera_name): 
        
        d = np.exp(rho / self.camera_transform_parameters[camera_name]['Klog'])
        th = phi / self.camera_transform_parameters[camera_name]['Kangle']

        x = d * np.cos(th) + self.camera_transform_parameters[camera_name]['xc']
        y = d * np.sin(th) + self.camera_transform_parameters[camera_name]['yc']

        return super().get_3D_point(x, y, camera_name)



class IncreasingActuityVision(EditVision):
    """
    Like the SimpleVision class, but the image is blurred before being returned.

    The blur is a gaussian blur with decreasing standard deviation.
    The std can decrease at fixed intervals or linearly (step size must be passed
    to the std_update function).
    """
    pass