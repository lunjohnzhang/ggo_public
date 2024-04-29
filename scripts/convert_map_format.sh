#!/bin/bash

USAGE="Usage: bash scripts/convert_map_format.sh MAP_FILEPATH DOMAIN [-s STORE_DIR]"

MAP_FILEPATH="$1"
DOMAIN="$2"

shift 2
while getopts "p:b:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      s) STORE_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${MAP_FILEPATH}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${DOMAIN}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi
STORE_DIR_ARG=""
if [ -n "$STORE_DIR" ]; then
  STORE_DIR_ARG="--store_dir $STORE_DIR"
fi
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/analysis/convert_map_format.py \
        --map_filepath "$MAP_FILEPATH" \
        --domain "$DOMAIN" \
        $STORE_DIR_ARG