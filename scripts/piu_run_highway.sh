#!/bin/bash

USAGE="Usage: bash scripts/iter_update_algo.sh LOGDIR DOMAIN_CONFIG N_EVALS N_WORKERS DOMAIN"

LOGDIR="$1"
DOMAIN_CONFIG="$2"
N_EVALS="$3"
N_WORKERS="$4"
DOMAIN="$5"

shift 5
while getopts "p:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -z "${LOGDIR}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${DOMAIN_CONFIG}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${N_EVALS}" ]
then
  echo "${USAGE}"
  exit 1
fi

if [ -z "${N_WORKERS}" ]
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

singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
    python env_search/iterative_update/run_highway.py \
        --logdir_path $LOGDIR \
        --domain_config $DOMAIN_CONFIG \
        --n_evals $N_EVALS \
        --domain $DOMAIN \
        --n_workers $N_WORKERS
