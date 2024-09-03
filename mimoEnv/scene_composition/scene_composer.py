# This script is designed to generate an XML scene.
# Certain elements in the scene (floor, walls, etc.) present multiple variations that can be sampled.
# Each variation is defined by an XML chunk, which is then inserted into the scene XML.

# Pipeline
# 1. Load the scene XML template.
# 2. Sample the scene variations.
#   a. Go into each folder (floor, wall, ceil) and sample one of the files.
#   b. Read the params from the JSON
# 3. Compose the placeholder assets for the room
# 4. Compose the placeholder body of the room
#   a. sample the dimensions of the room
#   b. generate the proper dimensions and positions for the walls, floor, and ceiling
# 5. Insert the sampled variations into the scene XML.

import json
import os
import random
import numpy as np
from scipy.spatial.transform import Rotation as R

from mimoEnv.envs.mimo_env import SCENE_DIRECTORY
from mimoEnv.scene_composition.utils import DoubleCosine, InvDist
import re
import mimoEnv.utils as me_utils


class SceneComposer:
    XYAXES = {"left": "1 0 0 0 0 1", "right": "-1 0 0 0 0 1", "front": "0 -1 0 0 0 1", "back": "0 1 0 0 0 1"}
    COLORS = ("red", "green", "blue", "yellow", "orange")
    INIT_MIMO_ROT = [0.892294, -0.0284863, -0.450353, -0.0135029]
    INIT_MIMO_POS = [0.0579584, -0.00157173, 0.0566738]

    KWDS = {
        "room":{
            "floor":{
                "asset": {
                    "texture": {"filename":"FLOORTEXFILENAME"},
                    "material": {"params": "FLOORMATPARAMS"},
                },
                "geom": {"size": "FLOORSIZE"}
            },
            "wall": {
                "asset":{
                    "texture": {"filename":"WALLTEXFILENAME"},
                    "material": {"params": "WALLMATPARAMS"},
                },
                "geom": {
                        "size": {
                            "left": "WALLSIZE_L",
                            "right": "WALLSIZE_R",
                            "front": "WALLSIZE_F",
                            "back": "WALLSIZE_B",
                        },
                        "pos": {
                            "left": "WALLPOS_L",
                            "right": "WALLPOS_R",
                            "front": "WALLPOS_F",
                            "back": "WALLPOS_B",
                        },
                        "xyaxes": {
                            "left": "WALLXYAXES_L",
                            "right": "WALLXYAXES_R",
                            "front": "WALLXYAXES_F",
                            "back": "WALLXYAXES_B",
                        } 
                }
            },
            "ceil": {
                "asset": {
                    "texture": {"filename":"CEILTEXFILENAME"},
                    "material": {"params": "CEILMATPARAMS"},
                },
                "geom": {"pos": "CEILPOS"},
            },
        },
        "lights": {
            "static": {"pos": "LIGHTSPOS"},
            "follow": {"pos": "LIGHTFPOS"},
        },
        "deco":
        {
            "asset": {
                "placeholder": "DECOASSET",
                "texture": {"name": "DECOTEXNAME", "filename":"DECOTEXFILENAME"},
                "material": {"name": "DECOMATNAME", "params": "DECOMATPARAMS"},
            },
            "geom": {
                "placeholder": "DECOGEOM",
                "name": "DECOGEOMNAME",
                "size": "GEOMDECOSIZE",
                "pos": "GEOMDECOPOS",
                "xyaxes": "GEOMDECOXYAXES"
            },
        },
        "toys": {
            "asset": {
                "placeholder": "TOYASSET",
                "material": {"name": "TOYMATNAME", "rgba": "TOYMATRGBA", "specular": "TOYMATSPEC", "shininess": "TOYMATSHINE"},
                "mesh": {"name": "TOYMESHNAME", "file": "TOYMESHFILE", "scale": "TOYMESHSCALE"},
            },
            "body": {
                "placeholder": "TOYBODY",
                "name": "TOYBODYNAME",
                "pos": "TOYBODYPOS",
                "euler": "TOYBODYEULER",
            },
            "geom": {
                "size": "TOYGEOMSIZE",
            },
        }
    }
    def __init__(self, mimo_version, room_size_min, room_size_max,
                 template_dir="templates", template_scene_file="template_scene.xml", 
                 output_scene_file="random_explore_scene.xml",
                 mimo_min_dist_from_wall=0.6,
                 deco_template_asset_file="asset_deco_template.xml",
                 deco_template_geom_file="geom_deco_template.xml",
                 toys_num_range=[1,10],
                 toy_template_asset_file="asset_toy_template.xml",
                 toy_template_body_file="body_toy_template.xml",
                 toy_area_frustum_deg=30, toy_distance_range=[0.5, 1.5], toy_scale_range=[0.01, 0.05],
                 toy_z_max=1.0):
        assert str(mimo_version) in ["v1", "v2", "1", "2"], "Invalid MIMo version"

        self.mimo_version = "v2" if str(mimo_version) in ["v2", "2"] else "v1"

        assert len(room_size_min) == 3, "Invalid room size min"
        assert len(room_size_max) == 3, "Invalid room size max"

        self.room_size = {"min": room_size_min, "max": room_size_max}

        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.template_dir = template_dir if template_dir[0] == "/" else os.path.join(self.base_dir, template_dir)

        self.template_file = os.path.join(self.template_dir, template_scene_file)
        self.output_scene_file = os.path.join(SCENE_DIRECTORY, output_scene_file)

        self.deco_template_asset_file = os.path.join(self.template_dir, deco_template_asset_file)
        self.deco_template_geom_file = os.path.join(self.template_dir, deco_template_geom_file)

        # MIMo

        self.mimo_minwdist = mimo_min_dist_from_wall

        # Toys

        self.toy_num_range = toys_num_range
        self.toy_template_asset_file = os.path.join(self.template_dir, toy_template_asset_file)
        self.toy_template_body_file = os.path.join(self.template_dir, toy_template_body_file)
        self.toys_dir = os.path.join(SCENE_DIRECTORY, "meshes")
        self.toy_area_f = np.deg2rad(toy_area_frustum_deg) # degrees to radians
        assert len(toy_distance_range) == 2 and toy_distance_range[0] <= toy_distance_range[1], "Invalid toy distance range"
        self.toy_dist_r = toy_distance_range
        self.toy_scale_r = toy_scale_range
        self.toy_z_max = toy_z_max

        self.num_toys = -1

        # --- #

        self.scene = ""

    def make_scene(self):
        replacements = {}
        KWDS = self.KWDS

        replacements["MIMOVERSION"] = "v2" if self.mimo_version == "v2" else ""

        # Load the scene XML template
        template = self.load_scene_template()

        # ASSETS sampling
        
        for dirname in KWDS["room"]:
            params = self.sample_asset_params(dirname)

            for key in KWDS["room"][dirname]["asset"]:
                for subkey in KWDS["room"][dirname]["asset"][key]:
                    # If the value is a dictionary parse it as a string formatted as "key1=value1 key2=value2 ..."
                    if isinstance(params[key][subkey], dict):
                        replacements[KWDS["room"][dirname]["asset"][key][subkey]] = " ".join([f"{k}=\"{v}\"" for k, v in params[key][subkey].items()])
                    else:
                        replacements[KWDS["room"][dirname]["asset"][key][subkey]] = params[key][subkey]
        
        # BODY sampling
        room_size = self.sample_room_size()
        geom = self.sample_body_params(room_size)

        for key in KWDS["room"]:
            for subkey in KWDS["room"][key]["geom"]:
                if isinstance(geom[key][subkey], dict):
                    for side, value in geom[key][subkey].items():
                        replacements[KWDS["room"][key]["geom"][subkey][side]] = value
                else:
                    replacements[KWDS["room"][key]["geom"][subkey]] = geom[key][subkey]

        # LIGHTS
        for key in KWDS["lights"]:
            replacements[KWDS["lights"][key]["pos"]] = self.sample_light_params(room_size)

        # DECORATIONS
        deco_assets, deco_geoms = self.generate_decorations(geom)
        replacements[KWDS["deco"]["asset"]["placeholder"]] = deco_assets
        replacements[KWDS["deco"]["geom"]["placeholder"]] = deco_geoms


        # MIMo

        mimo_pos, mimo_quat, mimo_angle = self.pose_mimo(room_size) # !DEBUG!

        # mimo_pos = [0,0,0]
        # mimo_quat = self.INIT_MIMO_ROT
        # mimo_angle = 0
        replacements["MIMOPOS"] = " ".join([f"{x:.3f}" for x in mimo_pos])
        replacements["MIMOQUAT"] = " ".join([f"{x:.3f}" for x in mimo_quat])

        # TOYS

        toy_assets, toy_bodies = self.spawn_toys(mimo_pos, mimo_angle, room_size)
        replacements[KWDS["toys"]["asset"]["placeholder"]] = toy_assets
        replacements[KWDS["toys"]["body"]["placeholder"]] = toy_bodies

        # Insert the sampled variations into the scene XML

        self.scene = self.replace_placeholders(template, replacements)

        return dict(mimo_pos=mimo_pos, mimo_quat=mimo_quat, mimo_angle=mimo_angle, room_size=room_size)

    def load_scene_template(self):
        with open(self.template_file, 'r') as file:
            template = file.read()
        return template

    def replace_placeholders(self, template, replacements):
        for key, value in replacements.items():
            ph = "@" + key + "@"
            template = template.replace(ph, value)
        return template

    def write_scene_file(self, scene=None):
        if scene is None:
            scene = self.scene
        with open(self.output_scene_file, 'w') as file:
            file.write(scene)

    def sample_asset_params(self, dirname):

        # Get the list of files in the directory
        DIR_NAME = os.path.join(self.base_dir, "scene_chunks", dirname)
        files = os.listdir(DIR_NAME)

        # Sample one of the files
        sampled_file = random.choice(files)

        # Read the JSON params from the sampled file
        with open(os.path.join(DIR_NAME, sampled_file), 'r') as file:
            params = json.load(file)
        return params
    
    def sample_room_size(self):
        # Sample the dimensions of the room
        room_size = [
            round(random.uniform(self.room_size["min"][0], self.room_size["max"][0]),3),
            round(random.uniform(self.room_size["min"][1], self.room_size["max"][1]),3),
            round(random.uniform(self.room_size["min"][2], self.room_size["max"][2]),3)
        ]
        return room_size

    def sample_body_params(self, room_size):

        # Generate the proper dimensions and positions for the walls, floor, and ceiling
        geom = {
            "floor": {"size" :f"{room_size[0]} {room_size[1]} 0.25"},

            "wall": {
                "size":{
                    "left":  f"{room_size[0]} {room_size[2]} 0.25",
                    "right": f"{room_size[0]} {room_size[2]} 0.25",
                    "front": f"{room_size[1]} {room_size[2]} 0.25",
                    "back":  f"{room_size[1]} {room_size[2]} 0.25"
                },
                "pos": {
                    "left":  f"0 {room_size[1]} {room_size[2]}",
                    "right": f"0 {-room_size[1]} {room_size[2]}",
                    "front": f"{room_size[0]} 0 {room_size[2]}",
                    "back":  f"{-room_size[0]} 0 {room_size[2]}",
                },
                "xyaxes": {
                    "left":  self.XYAXES["left"],
                    "right": self.XYAXES["right"],
                    "front": self.XYAXES["front"],
                    "back":  self.XYAXES["back"],
                }
            },
            "ceil": {"pos": f"0 0 {2*room_size[2]}"},
        }

        return geom

    def sample_light_params(self, room_size):
        z = random.uniform(0.8*room_size[2], room_size[2])

        x = random.uniform(-room_size[0]/2, room_size[0]/2)
        y = random.uniform(-room_size[1]/2, room_size[1]/2)

        return f"{x} {y} {z}"

    def generate_decorations(self, geom):
        # - select the number of decorations Nd
        # - for Nd
        #   - select a wall
        #   - split that wall into 4 regions ul ur bl br
        #   - assign the decoration to one region, checking first that it’s empty
        #   - sample the size that goes from min to min(max size, region size)
        #   - set the region as occupied
        #   - iterate
        def load_deco_templates():
            asset_template = ""
            geom_template = ""

            with open(self.deco_template_asset_file, 'r') as file:
                for line in file:
                    asset_template += line+"\t\t"
            with open(self.deco_template_geom_file, 'r') as file:
                for line in file:
                    geom_template += line+"\t\t"
            
            return asset_template, geom_template

        def sample_regions(sides, regions, N):
            l = [[s, r] for r in regions for s in sides]
            random.shuffle(l)
            return l[:N]
        
        def get_region_ext(sides, regions):
            region_corners = {k: [0.0, 0.0] for w in geom["wall"]["pos"].keys() for k in regions} 
            region_centers = {k: [0.0, 0.0] for w in geom["wall"]["pos"].keys() for k in regions} 
            for side in sides:

                wallpos = [float(x) for x in geom["wall"]["pos"][side].split(" ")]
                wallsize = [float(x) for x in geom["wall"]["size"][side].split(" ")[:2]]
                
                d = 1 if side in ["left", "right"] else 0 # non null dimension

                cw, ch = 0, wallpos[2]
                w,  h  = wallsize[0], wallsize[1] # NOTE: halfdims per MuJoCo convention
                wr = 0.8*w # region where the decoration can be placed
                hr = 0.8*h

                # Region corners
                region_corners["tl"] = [cw - wr, ch + hr]
                region_corners["tr"] = [cw + wr, ch + hr]
                region_corners["bl"] = [cw - wr, ch - hr]
                region_corners["br"] = [cw + wr, ch - hr]

                for reg in regions:
                    region_centers[reg] = [(region_corners[reg][0] + cw)/2, (region_corners[reg][1] + ch)/2]

            return region_corners, region_centers
        
        ############################

        KWDS = self.KWDS
        
        deco_files = os.listdir(os.path.join(self.base_dir, "scene_chunks", "decoration"))
        Nd = random.randint(1, len(deco_files))

        # shuffle deco_files
        random.shuffle(deco_files)

        deco_list = deco_files[:Nd]

        regions = ["tl", "tr", "bl", "br"]
        sides   = geom["wall"]["pos"].keys()

        sampled_regions = sample_regions(sides, regions, Nd)

        region_corners, region_centers = get_region_ext(sides, regions)

        i = 0

        asset_template, geom_template = load_deco_templates()

        deco_assets = ""
        deco_geoms = ""


        for (deco, sr) in zip(deco_list, sampled_regions):
            with open(os.path.join(self.base_dir, "scene_chunks", "decoration", deco), 'r') as file:
                params = json.load(file)

            side, reg = sr

            wallpos = [float(x) for x in geom["wall"]["pos"][side].split(" ")]
            
            d = 1 if side in ["left", "right"] else 0 # non null dimension
            wr = abs(region_corners[reg][0] - wallpos[1-d])
            hr = abs(region_corners[reg][1] - wallpos[2])

            maxsizeregion = min(wr, hr)

            mindecosize = params["geom"]["size_x"]["min"]
            maxdecosize = min(params["geom"]["size_x"]["max"], maxsizeregion)
            decosize_x = random.uniform(mindecosize, maxdecosize)/2


            decosize_y = decosize_x / params["geom"]["aspect_ratio"]

            decopos = [0,0,0]
            decopos[d] = wallpos[d] + (-1)**(side in ['left', 'front'])*0.001
            decopos[1-d] = region_centers[reg][0]
            decopos[2] = region_centers[reg][1]

            decoxyaxes = self.XYAXES[side]
            
            decoreplacements = {}

            decoreplacements[KWDS["deco"]["asset"]["texture"]["name"]] = f"texdeco{i}"
            decoreplacements[KWDS["deco"]["asset"]["texture"]["filename"]] = params["texture"]["filename"]
            decoreplacements[KWDS["deco"]["asset"]["material"]["name"]] = f"matdeco{i}"
            decoreplacements[KWDS["deco"]["asset"]["material"]["params"]] = " ".join([f"{k}=\"{v}\"" for k, v in  params["material"]["params"].items()])

            decoreplacements[KWDS["deco"]["geom"]["name"]] = f"deco{i}"
            decoreplacements[KWDS["deco"]["geom"]["size"]] = f"{decosize_x:.3f} {decosize_y:.3f} 0.1"
            decoreplacements[KWDS["deco"]["geom"]["pos"]] = f"{decopos[0]:.3f} {decopos[1]:.3f} {decopos[2]:.3f}"
            decoreplacements[KWDS["deco"]["geom"]["xyaxes"]] = decoxyaxes

            deco_assets += self.replace_placeholders(asset_template, decoreplacements) + "\n\n\t\t"
            deco_geoms += self.replace_placeholders(geom_template, decoreplacements) + "\n\n\t\t"

            i+=1
        
        return deco_assets, deco_geoms

    def pose_mimo(self, room_size):

        room_lim = np.array(room_size) - self.mimo_minwdist
        pose = self.INIT_MIMO_POS.copy()
        pose[:2] = np.random.standard_normal(2)*room_lim[:2]/3
        pose = np.clip(pose, -room_lim, room_lim)

        angle = random.uniform(0, 2*np.pi)

        r1 = R.from_quat(me_utils.quat_wlast(self.INIT_MIMO_ROT))
        r2 = R.from_rotvec([0,0,angle])
        r = me_utils.quat_wfirst((r2*r1).as_quat())

        return pose, r, angle
    
    def spawn_toys(self, mimo_pos, mimo_angle, room_size):

        # sample num toys from 1 to n_toys_max
        # for each toy
        #   sample a position around MIMo
        #       sample from either a left or right von Mises distribution centered at MIMo's 
        #       angle +- gamma degrees
        #       sample the distance with a uniform distribution
        #   sample a rotation
        #   sample a scale
        #   sample a toy from the assets/meshes folder
        #   sample a color material from a known list


        def load_toy_templates():
            asset_template = ""
            body_template = ""

            with open(self.toy_template_asset_file, 'r') as file:
                for line in file:
                    asset_template += line+"\t\t"
            with open(self.toy_template_body_file, 'r') as file:
                for line in file:
                    body_template += line+"\t\t"
            
            return asset_template, body_template
        # --- #

        KWDS = self.KWDS
        Nt = random.randint(*self.toy_num_range)
        self.num_toys = Nt

        toys_list = [file for file in os.listdir(self.toys_dir) if file.endswith('.stl')]
        toys_list = random.choices(toys_list, k=Nt)

        asset_template, body_template = load_toy_templates()

        params = dict(c=self.toy_area_f, a=0, b=np.pi)

        dist_l = DoubleCosine(**params)
        dist_r = InvDist(dist_type=DoubleCosine, **params)

        l_toys = np.random.binomial(Nt, 0.5)
        r_toys = Nt - l_toys

        dl = dist_l.rvs(size=l_toys)
        dr = dist_r.rvs(size=r_toys)

        if l_toys == 0:
            toy_angles = dr + mimo_angle
        elif r_toys == 0:
            toy_angles = dl + mimo_angle
        else:
            toy_angles = np.concatenate([dl, dr], axis=0) + mimo_angle

        toy_dists = np.empty(Nt)

        toy_angles = toy_angles % (2*np.pi)
        
        for i, ang in enumerate(toy_angles):
            d = self._dist_from_wall(mimo_pos, ang, room_size, margin=0.1)
            #print(f"DEBUG: dist from wall {d} [{mimo_pos=}], [{ang=} ({mimo_angle})], [{room_size=}]")
            toy_dists[i] = random.uniform(self.toy_dist_r[0], min(self.toy_dist_r[1], d))

        toy_pos = np.zeros((Nt, 3)) + mimo_pos
        toy_pos[:, 0] += toy_dists*np.cos(toy_angles)
        toy_pos[:, 1] += toy_dists*np.sin(toy_angles)
        toy_pos[:, 2] += np.random.uniform(0.1, self.toy_z_max-mimo_pos[2], Nt)


        toy_rot = np.random.uniform(0, 2*np.pi, Nt)

        toy_scales = np.random.uniform(*self.toy_scale_r, Nt) 

        toy_colors = np.empty((Nt, 3), dtype=np.float32)
        for i in range(3):
            toy_colors[:, i] = np.random.uniform(0, 1, Nt)
        
        toy_specular = np.random.beta(1, 5, Nt)
        toy_shininess = np.random.beta(1, 5, Nt)

        toy_assets = ""
        toy_bodies = ""

        for i in range(Nt):
            toyreplacements = {}

            toyreplacements[KWDS["toys"]["asset"]["material"]["name"]] = f"toy{i}_mat"
            toyreplacements[KWDS["toys"]["asset"]["material"]["rgba"]] = f"{toy_colors[i][0]:.3f} {toy_colors[i][1]:.3f} {toy_colors[i][2]:.3f} 1"
            toyreplacements[KWDS["toys"]["asset"]["material"]["specular"]] = f"{toy_specular[i]:.3f}" 
            toyreplacements[KWDS["toys"]["asset"]["material"]["shininess"]] = f"{toy_shininess[i]:.3f}" 


            toyreplacements[KWDS["toys"]["asset"]["mesh"]["name"]] = f"mesh_toy{i}"
            toyreplacements[KWDS["toys"]["asset"]["mesh"]["file"]] = toys_list[i]
            toyreplacements[KWDS["toys"]["asset"]["mesh"]["scale"]] = f"{toy_scales[i]:.3f} {toy_scales[i]:.3f} {toy_scales[i]:.3f}"

            toyreplacements[KWDS["toys"]["body"]["name"]] = f"toy{i}"
            toyreplacements[KWDS["toys"]["body"]["pos"]] = f"{toy_pos[i][0]:.3f} {toy_pos[i][1]:.3f} {toy_pos[i][2]:.3f}"
            toyreplacements[KWDS["toys"]["body"]["euler"]] = f"0 0 {np.rad2deg(toy_rot[i]):.3f}"

            toy_assets += self.replace_placeholders(asset_template, toyreplacements) + "\n\n\t\t"
            toy_bodies += self.replace_placeholders(body_template, toyreplacements) + "\n\n\t\t"

        return toy_assets, toy_bodies
    
    def _check_position_in_room(self, pos, room_size, margin):
        return np.all(np.abs(pos) < np.array(room_size) - margin)
    
    def _dist_from_wall(self, pos, angle, room_size, margin=0.0):
        """Return the length of the vector from pos to the nearest wall along direction angle"""
        
        # a. find the two closest walls to projection

        if np.isclose(angle, 0, atol=1e-3) or np.isclose(angle, np.pi, atol=1e-3): return abs(room_size[0] - pos[0])
        if np.isclose(angle, np.pi/2, atol=1e-3) or np.isclose(angle, 3*np.pi/2, atol=1e-3): return abs(room_size[1] - pos[1])


        k = np.empty(2)
        if angle > 0 and angle < np.pi/2:           k = np.array((1,1))
        elif angle > np.pi/2 and angle < np.pi:     k = np.array((-1,1))
        elif angle > np.pi and angle < 3*np.pi/2:   k = np.array((-1,-1))
        else:                                       k = np.array((1,-1))

        dists = np.abs(k*room_size[:2] - pos[:2])

        d = min(abs(dists[0]/np.cos(angle)), abs(dists[1]/np.sin(angle)))

        return d - margin
    
    def read_scene_file(self):
        with open(self.output_scene_file, 'r') as file:
            scene = file.read()

        def _read_pos_quat_angle(scene: str):
            pos = None
            quat = None

            match = re.search(r'<body name="mimo_location" pos="(.*?)" quat="(.*?)">', scene)
            if not match:
                raise ValueError("MIMo location and quaterion not found in scene file")
            
            pos = np.array([float(x) for x in match.group(1).split()])
            quat = np.array([float(x) for x in match.group(2).split()])

            q_final = R.from_quat(me_utils.quat_wfirst(quat))
            q_init = R.from_quat(me_utils.quat_wfirst(self.INIT_MIMO_ROT))

            angle = q_init*q_final.inv()

            return pos, quat, angle.as_rotvec()[2]
        
        def _read_room_size(scene: str):
            match = re.search(r'<geom name="wall_left"  type="plane" material="matwall"  size="(.*?)" pos="(.*?)" xyaxes="1 0 0 0 0 1"/>', scene)
            if not match:
                raise ValueError("Room size not found in scene file")
            
            wallsize = [float(x) for x in match.group(1).split()]
            wallpos = [float(x) for x in match.group(2).split()]

            return np.array([wallsize[0], wallpos[1], wallpos[2]])
        
        mimo_pos, mimo_quat, mimo_angle = _read_pos_quat_angle(scene)
        room_size = _read_room_size(scene)

        def read_num_toys(scene: str):            
            num_toys = len(re.findall(r'<body name="toy\d+"', scene))
            return num_toys

        self.num_toys = read_num_toys(scene)

        return dict(mimo_pos=mimo_pos, mimo_quat=mimo_quat, mimo_angle=mimo_angle, room_size=room_size)
            

def main():

    sc = SceneComposer("v2", [1,1,1], [3,3,3])
    sc.make_scene()
    sc.write_scene_file()



if __name__ == "__main__":
    main()