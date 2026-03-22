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

try:
    cv.__version__
except NameError:
    import cv2 as cv

from abc import ABC

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

    
    def get_cartesian_image(self, image: NDArray, camera_name: str) -> NDArray:
        """
        Returns the cartesian reprojection of the edited image, if any.
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
        #flexid = np.zeros(1, dtype=np.int32)
        skinid = np.zeros(1, dtype=np.int32)

        # Set the desired camera as the one from which MuJoCo will update the scene
        cam_id = self.env.model.camera(camera_name).id
        rgb_viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        rgb_viewer.cam.fixedcamid = cam_id

        # Update the mathematical scene state, without rendering or passing data to the GPU
        mujoco.mjv_updateScene(
            self.env.model,
            self.env.data,
            rgb_viewer.vopt,
            mujoco.MjvPerturb(), # Dummy perturb object
            rgb_viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            rgb_viewer.scn
        )

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

        return np.array([round(x), round(y)], dtype=np.int32)

    
    def _compute_camera_matrix(self):
        """Returns the 3x4 camera matrix."""
        # FROM: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb#scrollTo=sDYwClpxaxab
        # If the camera is a 'free' camera, we get its position and orientation
        # from the scene data structure. Note: we call `self.update()` in order to
        # ensure that the contents of `scene.camera` are correct.

        pos = self.env.data.cam_xpos[self.env.camera_id]

        rot_mat = self.env.data.cam_xmat[self.env.camera_id].reshape(3, 3)
        rot = rot_mat.T

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
        image[0, 2] = w  / 2.0
        image[1, 2] = h  / 2.0
        return image @ focal @ rotation @ translation

    
    def get_cartesian_image(self, image: NDArray, camera_name: str) -> NDArray:
        """
        Returns the cartesian reprojection of the logpolar image.
        """
        return image

    @property
    def width(self) -> int:
        """ Returns the width of the camera with name camera_name. """
        return {k: _v.width - _v.left for k, _v in self._viewports.items()}
    
    @property
    def height(self) -> int:
        """ Returns the width of the camera with name camera_name. """
        return {k: _v.height - _v.bottom for k, _v in self._viewports.items()}




class EditVision(SimpleVision):
    """A class that edits the images returned by the cameras before returning them to the environment.

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

        self._image_warp_func   = {}
        self._image_dewarp_func = {}

        for camera in camera_parameters:
            self._image_warp_func[camera]   = partial(camera_parameters[camera]["warp_function"], **camera_parameters[camera]["warp_function_args"])
            self._image_dewarp_func[camera] = partial(camera_parameters[camera]["dewarp_function"], **camera_parameters[camera]["dewarp_function_args"])
        

        # check if func_args already specifies one set of args for each camera; otherwise copy the same args for each camera
    
    def get_vision_obs(self):
        imgs =  super().get_vision_obs()

        for camera, img in imgs.items():
            imgs[camera] = self._image_warp_func[camera](img)
        
        return imgs
    
class ILogPolarVision(ABC):
    pass

class LogPolarVision(EditVision, ILogPolarVision):
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



        self.camera_transform_parameters = {}

        for camera, params in camera_parameters.items():
            out_params, self.camera_transform_parameters[camera] = self._get_logpolar_params(camera, params)
            params.update(out_params)

        super().__init__(env, camera_parameters)
    
    def _get_logpolar_params(self, camera, params):
        out_params= dict(params)

        max_radius = out_params.get("maxRadius", min(out_params["width"] / 2, out_params["height"] / 2))
        log_fraction = out_params.get("logFraction", 1)

        camera_transform_parameters = {
            'Klog': out_params["width"]*log_fraction / np.log(max_radius),
            'Kangle': out_params["height"] / (2*np.pi),
            'xc': out_params["width"] / 2,
            'yc': out_params["height"] / 2,
        }
        
        out_params.update(
            dict(
                warp_function=cv.warpPolar,
                warp_function_args=dict(
                    maxRadius=max_radius,
                    dsize=(int(out_params["width"]*log_fraction), out_params["height"]),
                    center=(out_params["width"] / 2, out_params["height"] / 2),
                    flags=cv.INTER_LINEAR + cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG
                ), # arguments to be passed to the function

                dewarp_function=cv.warpPolar,
                dewarp_function_args=dict(
                    maxRadius=max_radius,
                    dsize=(out_params["width"], out_params["height"]),
                    center=(out_params["width"] / 2, out_params["height"] / 2),
                    flags=cv.WARP_FILL_OUTLIERS + cv.WARP_POLAR_LOG + cv.WARP_INVERSE_MAP
                )
            )
        )
        return out_params, camera_transform_parameters
    
    def get_3D_point(self, rho, phi, camera_name): 
        
        d = np.exp(rho / self.camera_transform_parameters[camera_name]['Klog'])
        th = phi / self.camera_transform_parameters[camera_name]['Kangle']

        x = d * np.cos(th) + self.camera_transform_parameters[camera_name]['xc']
        y = d * np.sin(th) + self.camera_transform_parameters[camera_name]['yc']

        return super().get_3D_point(x, y, camera_name)

    def get_logpolar_from_cartesian_2D(self, x, y, camera_name):
        """
        Returns the logpolar coordinates corresponding to the pixel (x, y) in the image of the camera with name camera_name.
        """
        xc = self.camera_transform_parameters[camera_name]['xc']
        yc = self.camera_transform_parameters[camera_name]['yc']
        Klog = self.camera_transform_parameters[camera_name]['Klog']
        Kangle = self.camera_transform_parameters[camera_name]['Kangle']

        dx = x - xc
        dy = y - yc

        rho = np.log(np.sqrt(dx**2 + dy**2)) * Klog
        phi = np.arctan2(dy, dx) * Kangle

        return rho, phi
    
    def get_cartesian_image(self, image: NDArray, camera_name: str) -> NDArray:
        """
        Returns the cartesian reprojection of the logpolar image.
        """
        return self._image_dewarp_func[camera_name](image)

    @property
    def width(self) -> int:
        """ Returns the width of the camera with name camera_name. """
        return {k: int((_v.width - _v.left) * self.camera_parameters[k]['logFraction']) for k, _v in self._viewports.items()}
    
class LogPolarNCartesianVision(SimpleVision, ILogPolarVision):
    """
    Returns both the logpolar and the original cartesian images.
    """
    
    def __init__(self, env, camera_parameters): #, func: Callable[[np.ndarray, Dict], np.ndarray] = lambda x, _: x, func_args: Dict = {}):

        cartesian_camera_parameters = {}
        logpolar_camera_parameters = {}
        self.is_camera_logpolar = {}

        self.l2c_cameramap = {} # map logpolar camera names used for convenience to real camera sensor names
        self.c2l_cameramap = {} # map logpolar camera names used for convenience to real camera sensor names

        for camera, params in camera_parameters.items():
            is_logpolar = params.pop("is_logpolar", False)
            self.is_camera_logpolar[camera] = is_logpolar

        for camera in camera_parameters: # logpolar
            if self.is_camera_logpolar[camera]:
                for camera2 in camera_parameters: # cartesian
                    if self.is_camera_logpolar[camera2] == False  and camera2 in camera:
                        self.l2c_cameramap[camera] = camera2
                        self.c2l_cameramap[camera2] = camera
                        self.c2l_cameramap[camera] = camera
                        break
            else:
                self.l2c_cameramap[camera] = camera


        for camera, params in camera_parameters.items():
            is_logpolar = self.is_camera_logpolar[camera]
            camera

            if is_logpolar:
                logpolar_camera_parameters[camera] = params
            else:
                cartesian_camera_parameters[camera] = params    

        self.cartesian_vision   = SimpleVision(env, cartesian_camera_parameters)

        remapped_logpolar_camera_parameters = {}
        for camera, params in logpolar_camera_parameters.items():
            remapped_logpolar_camera_parameters[self.l2c_cameramap[camera]] = params
        self.logpolar_vision    = LogPolarVision(env, remapped_logpolar_camera_parameters)

        self.camera_parameters = {}
        self.camera_parameters.update(cartesian_camera_parameters)
        self.camera_parameters.update(logpolar_camera_parameters)
   
    def get_vision_obs(self):
        imgs = {}

        imgs.update(self.cartesian_vision.get_vision_obs())
        for lp_camera, lp_img in self.logpolar_vision.get_vision_obs().items():
            imgs[self.c2l_cameramap[lp_camera]] = lp_img


        return imgs
    
    def get_cartesian_image(self, image, camera_name):
        if self.is_camera_logpolar[camera_name]:
            return self.logpolar_vision.get_cartesian_image(image, self.l2c_cameramap[camera_name])
        else:
            return self.cartesian_vision.get_cartesian_image(image, camera_name)
        
    def get_3D_point(self, a, b, camera_name):
        if self.is_camera_logpolar[camera_name]:
            return self.logpolar_vision.get_3D_point(rho=a, phi=b, camera_name=self.l2c_cameramap[camera_name])
        else:
            return self.cartesian_vision.get_3D_point(x=a, y=b, camera_name=camera_name)
        
    def get_logpolar_from_cartesian_2D(self, x, y, camera_name):
        
        if self.is_camera_logpolar[camera_name]:
            return self.logpolar_vision.get_logpolar_from_cartesian_2D(x, y, self.l2c_cameramap[camera_name])
        else:
            raise ValueError(f"The camera {camera_name} is already in cartesian space")
    
    def get_2d_from_3d(self, point: NDArray, camera_name: str):

        x, y = self.cartesian_vision.get_2d_from_3d(point, camera_name)

        if self.is_camera_logpolar[camera_name]:
            return self.logpolar_vision.get_logpolar_from_cartesian_2D(x, y, self.l2c_cameramap[camera_name])
        else:
            return x, y
    

    @property
    def width(self) -> int:
        """ Returns the width of the camera with name camera_name. """

        w = {}

        for k, _v in self.logpolar_vision._viewports.items():
            klp = self.c2l_cameramap[k]
            w[klp] = int((_v.width - _v.left) * self.camera_parameters[self.c2l_cameramap[k]]['logFraction'])
        
        
        for k, _v in self.cartesian_vision._viewports.items():
            w[k] = _v.width - _v.left

        return w    
    
    
    @property
    def height(self) -> int:
        """ Returns the width of the camera with name camera_name. """
        h = {}
        
        for k, _v in self.logpolar_vision._viewports.items():
            klp = self.c2l_cameramap[k]
            h[klp] = _v.height - _v.bottom
        
        
        for k, _v in self.cartesian_vision._viewports.items():
            h[k] = _v.height - _v.bottom

        return h    

class IncreasingActuityVision(EditVision):
    """
    Like the SimpleVision class, but the image is blurred before being returned.

    The blur is a gaussian blur with decreasing standard deviation.
    The std can decrease at fixed intervals or linearly (step size must be passed
    to the std_update function).
    """
    pass