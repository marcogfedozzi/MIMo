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

from mimoEnv.envs.mimo_env import SCENE_DIRECTORY

BASE_DIR = os.path.join(SCENE_DIRECTORY, '..', 'scene_composition')

TEMPLATE_SCENE_FILE = os.path.join(BASE_DIR, "template_scene.xml")
OUTPUT_SCENE_FILE   = os.path.join(SCENE_DIRECTORY, "random_explore_scene.xml")

ROOM_SIZE = { # actuaslly half-sizes
    "min": [1, 1, 1],
    "max": [2, 2, 2]
}

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
}


def load_scene_template():
    with open(TEMPLATE_SCENE_FILE, 'r') as file:
        template = file.read()
    return template

def replace_placeholders(template, replacements):
    for key, value in replacements.items():
        ph = "@" + key + "@"
        template = template.replace(ph, value)
    return template

def write_scene_file(template):
    with open(OUTPUT_SCENE_FILE, 'w') as file:
        file.write(template)

def sample_asset_params(dirname):

    # Get the list of files in the directory
    DIR_NAME = os.path.join(BASE_DIR, "scene_chunks", dirname)
    files = os.listdir(DIR_NAME)

    # Sample one of the files
    sampled_file = random.choice(files)

    # Read the JSON params from the sampled file
    with open(os.path.join(DIR_NAME, sampled_file), 'r') as file:
        params = json.load(file)
    return params

def sample_room_size():
    # Sample the dimensions of the room
    room_size = [
        round(random.uniform(ROOM_SIZE["min"][0], ROOM_SIZE["max"][0]),3),
        round(random.uniform(ROOM_SIZE["min"][1], ROOM_SIZE["max"][1]),3),
        round(random.uniform(ROOM_SIZE["min"][2], ROOM_SIZE["max"][2]),3)
    ]
    return room_size

def sample_body_params(room_size):

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
            }
        },
        "ceil": {"pos": f"0 0 {2*room_size[2]}"},
    }

    return geom

def sample_light_params(room_size):
    z = random.uniform(0.8*room_size[2], room_size[2])

    x = random.uniform(-room_size[0]/2, room_size[0]/2)
    y = random.uniform(-room_size[1]/2, room_size[1]/2)

    return f"{x} {y} {z}"


def main():

    replacements = {}

    # Load the scene XML template
    template = load_scene_template()

    # ASSETS sampling
    
    for dirname in KWDS["room"]:
        params = sample_asset_params(dirname)

        for key in KWDS["room"][dirname]["asset"]:
            for subkey in KWDS["room"][dirname]["asset"][key]:
                # If the value is a dictionary parse it as a string formatted as "key1=value1 key2=value2 ..."
                if isinstance(params[key][subkey], dict):
                    replacements[KWDS["room"][dirname]["asset"][key][subkey]] = " ".join([f"{k}=\"{v}\"" for k, v in params[key][subkey].items()])
                else:
                    replacements[KWDS["room"][dirname]["asset"][key][subkey]] = params[key][subkey]
    
    # BODY sampling
    room_size = sample_room_size()
    geom = sample_body_params(room_size)

    for key in KWDS["room"]:
        for subkey in KWDS["room"][key]["geom"]:
            if isinstance(geom[key][subkey], dict):
                for side, value in geom[key][subkey].items():
                    replacements[KWDS["room"][key]["geom"][subkey][side]] = value
            else:
                replacements[KWDS["room"][key]["geom"][subkey]] = geom[key][subkey]

    # LIGHTS
    for key in KWDS["lights"]:
        replacements[KWDS["lights"][key]["pos"]] = sample_light_params(room_size)

    # Insert the sampled variations into the scene XML
    scene = replace_placeholders(template, replacements)

    # Write the scene XML file
    write_scene_file(scene)

if __name__ == "__main__":
    main()