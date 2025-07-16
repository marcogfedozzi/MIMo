#!/bin/bash

# This script checks whether each OBJINSTANCE folder in the MJCF dataset contains the expected XML file.


if [ -z "$PATH_TO_MJCF" ]; then
    echo "Please set the PATH_TO_MJCF environment variable to the path of the modified Toys4k dataset."
    exit 1
fi

missing_count=0
correct_count=0


for OBJTYPE in "$PATH_TO_MJCF"/*; do
    [ -d "$OBJTYPE" ] || continue
    OBJTYPE_NAME=$(basename "$OBJTYPE")
    for OBJINSTANCE in "$OBJTYPE"/*; do
        [ -d "$OBJINSTANCE" ] || continue
        OBJINSTANCE_NAME=$(basename "$OBJINSTANCE")
        XML_FILE="$OBJINSTANCE/$OBJINSTANCE_NAME.xml"
        if [ ! -f "$XML_FILE" ]; then
            echo "Missing: $XML_FILE"
            rm -rf "$OBJINSTANCE"
            missing_count=$((missing_count+1))
        else
            correct_count=$((correct_count+1))
        fi
    done
done

if [ "$missing_count" -eq 0 ]; then
    echo "All $correct_count OBJINSTANCE folders contain their XML file."
else
    echo "$missing_count OBJINSTANCE folders are missing their XML file."
fi