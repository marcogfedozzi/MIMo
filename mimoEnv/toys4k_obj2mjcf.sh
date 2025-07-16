#!/bin/bash

# This script converts toy objects from the Toys4k dataset to MJCF format.

PATH_TO_TOYS4K="${PATH_TO_TOYS4K}"

N_CONVEX_HULLS=60
CONC_THRESHOLD=0.05

if [ -z "$PATH_TO_TOYS4K" ]; then
    echo "Please set the PATH_TO_TOYS4K environment variable to the path of the Toys4k dataset."
    exit 1
fi


if [ -z "$PATH_TO_MJCF" ]; then
    echo "Please set the PATH_TO_MJCF environment variable to the path of the modified Toys4k dataset."
    exit 1
fi


# Toys4k dataset structure
#
# objtype1
# - objinstance1
# - objinstance2
# objtype2
# - objinstance1
# ...

# for each objinstance run the following command
# obj2mjcf --obj-dir {OBJINSTANCE_DIR} --save-mjcf --add-free-joint --decompose \
# --coacd-args.preprocess-resolution 20 --coacd-args.max-convex-hull ${N_CONVEX_HULLS} \
# --coacd-args.threshold ${CONC_THRESHOLD}

# the follwing command generated a new folder within OBJINSTANCE_DIR with the same name
# move this folder to a new path generating a new dataset structure
# ${PATH_TO_TOYS4K}_mjcf/OBJTYPE/OBJINSTANCE

OUTPUT_DIR="${PATH_TO_MJCF}"
mkdir -p "$OUTPUT_DIR"

for OBJTYPE in "$PATH_TO_TOYS4K"/*; do
    [ -d "$OBJTYPE" ] || continue
    OBJTYPE_NAME=$(basename "$OBJTYPE")
    for OBJINSTANCE in "$OBJTYPE"/*; do
        [ -d "$OBJINSTANCE" ] || continue
        OBJINSTANCE_NAME=$(basename "$OBJINSTANCE")

        # Run obj2mjcf
        obj2mjcf --obj-dir "$OBJINSTANCE" --save-mjcf --add-free-joint --decompose \
            --coacd-args.preprocess-resolution 20 \
            --coacd-args.max-convex-hull ${N_CONVEX_HULLS} \
            --coacd-args.threshold ${CONC_THRESHOLD}
        if [ $? -ne 0 ]; then
            echo "obj2mjcf failed for $OBJINSTANCE, skipping."
            continue
        fi

        # The MJCF folder is created inside OBJINSTANCE with the same name
        MJCF_FOLDER="$OBJINSTANCE/$OBJINSTANCE_NAME"
        DEST_DIR="$OUTPUT_DIR/$OBJTYPE_NAME/$OBJINSTANCE_NAME"
        mkdir -p "$DEST_DIR"

        # Move the generated MJCF folder to the new dataset structure
        if [ -d "$MJCF_FOLDER" ]; then
            mv "$MJCF_FOLDER"/* "$DEST_DIR/"
            rmdir "$MJCF_FOLDER"
        else
            echo "MJCF folder not found for $OBJINSTANCE"
        fi
    done
done

