#!/bin/bash

USAGE="Usage: bash scripts/run_single_sim.sh SIM_CONFIG MAP_FILE AGENT_NUM N_EVALS MODE N_SIM N_WORKERS"

SIM_CONFIG="$1"
MAP_FILE="$2"
AGENT_NUM="$3"
AGENT_NUM_STEP_SIZE="$4"
N_EVALS="$5"
MODE="$6"
N_SIM="$7"
N_WORKERS="$8"
DOMAIN="$9"

shift 9
while getopts "p:r:" flag; do
  case "$flag" in
      p) PROJECT_DIR=$OPTARG;;
      r) RELOAD=$OPTARG;;
      *) echo "Invalid option. ${USAGE}"
  esac
done

if [ -n "$PROJECT_DIR" ]; then
  SINGULARITY_OPTS="$SINGULARITY_OPTS --bind ${PROJECT_DIR}:${PROJECT_DIR}"
fi

if [ "${DOMAIN}" = "kiva" ]
then
    if [ "${MODE}" = "inc_agents" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
            python env_search/warehouse/module.py \
                --warehouse-config "$SIM_CONFIG" \
                --map-filepath "$MAP_FILE" \
                --agent-num "$AGENT_NUM" \
                --agent-num-step-size "$AGENT_NUM_STEP_SIZE" \
                --n_evals "$N_EVALS" \
                --mode "$MODE" \
                --n_sim "$N_SIM" \
                --n_workers "$N_WORKERS" \
                --reload "$RELOAD"
        sleep 2
    fi

    if [ "${MODE}" = "constant" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
        python env_search/warehouse/module.py \
            --warehouse-config "$SIM_CONFIG" \
            --map-filepath "$MAP_FILE" \
            --agent-num "$AGENT_NUM" \
            --n_evals "$N_EVALS" \
            --mode "$MODE" \
            --n_workers "$N_WORKERS"
    fi
fi

if [ "${DOMAIN}" = "competition" ]
then
    if [ "${MODE}" = "inc_agents" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
            python env_search/competition/module.py \
                --competition-config "$SIM_CONFIG" \
                --map-filepath "$MAP_FILE" \
                --agent-num "$AGENT_NUM" \
                --agent-num-step-size "$AGENT_NUM_STEP_SIZE" \
                --n_evals "$N_EVALS" \
                --mode "$MODE" \
                --n_sim "$N_SIM" \
                --n_workers "$N_WORKERS" \
                --reload "$RELOAD"
        sleep 2
    fi

    if [ "${MODE}" = "constant" ]
    then
        singularity exec ${SINGULARITY_OPTS} --cleanenv singularity/ubuntu_warehouse.sif \
        python env_search/competition/module.py \
            --competition-config "$SIM_CONFIG" \
            --map-filepath "$MAP_FILE" \
            --agent-num "$AGENT_NUM" \
            --n_evals "$N_EVALS" \
            --mode "$MODE" \
            --n_workers "$N_WORKERS"
    fi
fi