#!/bin/bash

USAGE="Usage: bash scripts/iter_update_algo.sh CONFIG_FILE N_VALID_EDGE"

CONFIG_FILE="$1"
N_VALID_EDGE="$2"
N_VALID_VERTICES="$3"
DOMAIN="$4"

shift 4
while getopts "p:i:k:n:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      i) INIT_WEIGHT_FILE=$OPTARG;;
      k) KIVA_MAP_FILEPATH=$OPTARG;;
      n) N_AGENTS=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${CONFIG_FILE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${N_VALID_EDGE}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${N_VALID_VERTICES}" ]
then
  echo "${USAGE}"
  exit 1
fi

SINGULARITY_OPTS="--cleanenv --env MALLOC_TRIM_THRESHOLD_=0"
if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi

singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/iterative_update/algo.py \
        --config_file "$CONFIG_FILE" \
        --n_valid_edges "$N_VALID_EDGE" \
        --n_valid_vertices "$N_VALID_VERTICES" \
        --init_weight_file "$INIT_WEIGHT_FILE" \
        --kiva_map_filepath "$KIVA_MAP_FILEPATH" \
        --num_agents "$N_AGENTS" \
        --domain "$DOMAIN"