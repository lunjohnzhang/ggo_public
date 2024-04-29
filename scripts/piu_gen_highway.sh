#!/bin/bash

USAGE="Usage: bash scripts/iter_update_algo.sh MODEL_PATH MAP_PATH PIU_CONFIG N_AGENT_START N_AGENT_STEP N_AGENT_END N_WORKERS DOMAIN"

MODEL_PATH="$1"
MAP_PATH="$2"
PIU_CONFIG="$3"
N_AGENT_START="$4"
N_AGENT_STEP="$5"
N_AGENT_END="$6"
N_WORKERS="$7"
DOMAIN="$8"

shift 8
while getopts "p:r:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      r) RELOAD_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${MODEL_PATH}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${MAP_PATH}" ]
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

RELOAD_OPT=""
if [ -n "$RELOAD_DIR" ]; then
  RELOAD_OPT="--reload_dir $RELOAD_DIR"
fi
singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/iterative_update/gen_highway.py \
        --model_filepath $MODEL_PATH \
        --map_filepath $MAP_PATH \
        --piu_config_file $PIU_CONFIG \
        --domain $DOMAIN \
        --n_agent_start $N_AGENT_START \
        --n_agent_end $N_AGENT_END \
        --n_agent_step $N_AGENT_STEP \
        --n_workers $N_WORKERS \
        $RELOAD_OPT
