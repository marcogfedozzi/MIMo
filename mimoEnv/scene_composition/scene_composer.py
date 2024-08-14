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

ROOM_SIZE = { # actually half-sizes
    "min": [1, 1, 0.5],
    "max": [2, 2, 1]
}


XYAXES = {"left": "1 0 0 0 0 1", "right": "-1 0 0 0 0 1", "front": "0 -1 0 0 0 1", "back": "0 1 0 0 0 1"}

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
            },
            "xyaxes": {
                "left":  XYAXES["left"],
                "right": XYAXES["right"],
                "front": XYAXES["front"],
                "back":  XYAXES["back"],
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

def generate_decorations(geom, replacements):
    # - select the number of decorations Nd
    # - for Nd
    #   - select a wall
    #   - split that wall into 4 regions ul ur bl br
    #   - assign the decoration to one region, checking first that it’s empty
    #   - sample the size that goes from min to min(max size, region size)
    #   - set the region as occupied
    #   - iterate
    def load_deco_templates():
        assetfile = os.path.join(BASE_DIR,  "asset_deco_template.xml")
        geomfile = os.path.join(BASE_DIR, "geom_deco_template.xml")

        asset_template = ""
        geom_template = ""

        with open(assetfile, 'r') as file:
            for line in file:
                asset_template += line+"\t\t"
        with open(geomfile, 'r') as file:
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
    
    deco_files = os.listdir(os.path.join(BASE_DIR, "scene_chunks", "decoration"))
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
        with open(os.path.join(BASE_DIR, "scene_chunks", "decoration", deco), 'r') as file:
            params = json.load(file)

        side, reg = sr

        wallpos = [float(x) for x in geom["wall"]["pos"][side].split(" ")]
        wallsize = [float(x) for x in geom["wall"]["size"][side].split(" ")[:2]]
        
        d = 1 if side in ["left", "right"] else 0 # non null dimension
        wr = abs(region_corners[reg][0] - wallpos[d])
        hr = abs(region_corners[reg][1] - wallpos[2])

        maxsizeregion = min(wr, hr)

        mindecosize = params["geom"]["size_x"]["min"]
        maxdecosize = min(params["geom"]["size_x"]["max"], maxsizeregion)
        decosize_x = random.uniform(mindecosize, maxdecosize)
        decosize_y = decosize_x / params["geom"]["aspect_ratio"]

        decopos = [0,0,0]
        decopos[d] = wallpos[d] + (-1)**(side in ['left', 'front'])*0.001
        decopos[1-d] = region_centers[reg][0]
        decopos[2] = region_centers[reg][1]

        decoxyaxes = XYAXES[side]
        
        decoreplacements = {}

        decoreplacements[KWDS["deco"]["asset"]["texture"]["name"]] = f"texdeco{i}"
        decoreplacements[KWDS["deco"]["asset"]["texture"]["filename"]] = params["texture"]["filename"]
        decoreplacements[KWDS["deco"]["asset"]["material"]["name"]] = f"matdeco{i}"
        decoreplacements[KWDS["deco"]["asset"]["material"]["params"]] = " ".join([f"{k}=\"{v}\"" for k, v in  params["material"]["params"].items()])

        decoreplacements[KWDS["deco"]["geom"]["name"]] = f"deco{i}"
        decoreplacements[KWDS["deco"]["geom"]["size"]] = f"{decosize_x:.3f} {decosize_y:.3f} 0.1"
        decoreplacements[KWDS["deco"]["geom"]["pos"]] = f"{decopos[0]:.3f} {decopos[1]:.3f} {decopos[2]:.3f}"
        decoreplacements[KWDS["deco"]["geom"]["xyaxes"]] = decoxyaxes

        deco_assets += replace_placeholders(asset_template, decoreplacements) + "\n\n\t\t"
        deco_geoms += replace_placeholders(geom_template, decoreplacements) + "\n\n\t\t"

        i+=1
    
    return deco_assets, deco_geoms


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

    # DECORATIONS
    deco_assets, deco_geoms = generate_decorations(geom, replacements)
    replacements[KWDS["deco"]["asset"]["placeholder"]] = deco_assets
    replacements[KWDS["deco"]["geom"]["placeholder"]] = deco_geoms

    # Insert the sampled variations into the scene XML
    scene = replace_placeholders(template, replacements)

    # Write the scene XML file
    write_scene_file(scene)

if __name__ == "__main__":
    main()