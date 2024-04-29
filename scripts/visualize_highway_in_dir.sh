#!/bin/bash

TO_PLOT="$1"
STORE_DIR="$2"
DOMAIN="$3"

for DIR in ${TO_PLOT}/*;
do
    # echo "${DIR}"
    echo "${DIR}"
    bash scripts/visualize_highway.sh "$DIR"  "$STORE_DIR" "$DOMAIN" \
        -p /media/project0 &
done