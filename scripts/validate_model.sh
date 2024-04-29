#!/bin/bash

USAGE="Usage: bash scripts/validate_model.sh LOGDIR DOMAIN [OFFLINE_LOGDIR_DATA]"

LOGDIR="$1"
DOMAIN="$2"
OFFLINE_LOGDIR_DATA="$3"

if [ -z "${LOGDIR}" ]
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

if [ -z "${OFFLINE_LOGDIR_DATA}" ]
then
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/validate_model.py \
            --domain "$DOMAIN" \
            --logdir "$LOGDIR"
    exit 1
else
    singularity exec ${SINGULARITY_OPTS} singularity/ubuntu_warehouse.sif \
        python env_search/analysis/validate_model.py \
            --logdir "$LOGDIR" \
            --domain "$DOMAIN" \
            --offline_logdir_data "${OFFLINE_LOGDIR_DATA}"
fi


