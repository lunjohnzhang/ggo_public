# Guidance Graph Optimization for Lifelong Multi-Agent Path Finding

This repository is the official implementation of **[Guidance Graph Optimization for Lifelong Multi-Agent Path Finding](https://arxiv.org/abs/2402.01446)**, accepted to IJCAI 2024. The repository builds on top of the repository of [Multi-Robot Coordination and Layout Design for Automated Warehousing](https://github.com/lunjohnzhang/warehouse_env_gen_public)

**If you are just looking for the optimized guidance graphs, please refer to the [Optimized Guidance Graphs](#optimized-guidance-graphs) section below or go to the [project website](https://yulunzhang.net/publication/zhang2024ggo/).**


## Installation

This is a hybrid C++/Python project. The simulation environment is written in C++ and the rests are in Python. We use [pybind11](https://pybind11.readthedocs.io/en/stable/) to bind the two languages.

1. **Initialize pybind11:** After cloning the repo, initialize the pybind11 submodule

   ```bash
   git submodule init
   git submodule update
   ```

1. **Install Singularity:** All of our code runs in a Singularity container.
   Singularity is a container platform (similar in many ways to Docker). Please
   see the instructions [here](https://sylabs.io/docs/) for installing Singularity.
   As a reference, we use version 3.10.5 on Ubuntu 20.04.

1. **Download Boost:** From the root directory of the project, run the following to download the Boost 1.71, which is required for compiling C++ simulator. You don't have to install it on your system since it will be passed into the container and installed there.

   ```
   wget https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz --no-check-certificate
   ```

1. **Build Singularity container:** Run the provided script to build the container. Note that this need `sudo` permission on your system.
   ```
   bash build_container.sh
   ```
   The script will first build a container as a sandbox, compile the C++ simulator, then convert that to a regular `.sif` Singularity container.

## Optimizing Guidance Graphs

### Training Logging Directory Manifest

Regardless of where the script is run, the log files and results are placed in a
logging directory in `logs/`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_<dashed-name>_<uuid>`, e.g.
`2020-12-01_15-00-30_experiment-1_ff1dcb2b`. Inside each directory are the
following files:

```text
- config.gin  # All experiment config variables, lumped into one file.
- seed  # Text file containing the seed for the experiment.
- reload.pkl  # Data necessary to reload the experiment if it fails.
- metrics.json  # Data for a MetricLogger with info from the entire run, e.g. QD score.
- hpc_config.sh  # Same as the config in the Slurm dir, if Slurm is used.
- archive/  # Snapshots of the full archive, including solutions and metadata,
            # in pickle format.
- archive_history.pkl  # Stores objective values and behavior values necessary
                       # to reconstruct the archive. Solutions and metadata are
                       # excluded to save memory.
- dashboard_status.txt  # Job status which can be picked up by dashboard scripts.
                        # Only used during execution.
- evaluations # Output logs of LMAPF simulator
```

### Running Locally

#### Single Run

To run one experiment locally, use:

```bash
bash scripts/run_local.sh CONFIG SEED NUM_WORKERS
```

For instance, with 4 workers:

```bash
bash scripts/run_local.sh config/foo.gin 42 4
```

`CONFIG` is the [gin](https://github.com/google/gin-config) experiment config
for `env_search/main.py`.

### Running on Slurm

Use the following command to run an experiment on an HPC with Slurm (and
Singularity) installed:

```bash
bash scripts/run_slurm.sh CONFIG SEED HPC_CONFIG
```

For example:

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh
```

`CONFIG` is the experiment config for `env_search/main.py`, and `HPC_CONFIG` is a shell
file that is sourced by the script to provide configuration for the Slurm
cluster. See `config/hpc` for example files.

Once the script has run, it will output commands like the following:

- `tail -f ...` - You can use this to monitor stdout and stderr of the main
  experiment script. Run it.
- `bash scripts/slurm_cancel.sh ...` - This will cancel the job.


### Troubleshooting

1. If you encounter the following error while running experiments in the Singularity container:

   ```
   container creation failed: mount /proc/self/fd/3->/usr/local/var/singularity/mnt/session/rootfs error: while mounting image`/proc/self/fd/3: failed to find loop device: no loop devices available
   ```

   Please try downgrading/upgrading the Linux kernel version to `5.15.0-67-generic`, as suggested in [this Github issue](https://github.com/sylabs/singularity/issues/1499).

1. On Linux, if you are running anything in the container from external drivers mounted to the home driver (e.g. `/mnt/project`), you need to add `--bind /mnt/project:/mnt/project` to the singularity command in order to bind that external driver also to the container. For example, if you are running an experiment from an external driver, run with:
   ```
   bash scripts/run_local.sh CONFIG SEED NUM_WORKERS -p /mnt/project
   ```
   The `-p` argument helps you add the `--bind` argument to the singularity command in the script.



### Reloading

While the experiment is running, its state is saved to `reload.pkl` in the
logging directory. If the experiment fails, e.g. due to memory limits, time
limits, or network connection issues, `reload.pkl` may be used to continue the
experiment. To do so, execute the same command as before, but append the path to
the logging directory of the failed experiment.

```bash
bash scripts/run_slurm.sh config/foo.gin 42 config/hpc/test.sh -r logs/.../
```

The experiment will then run to completion in the same logging directory. This
works with `scripts/run_local.sh` too.

### Configuration Files of GGO Methods

The `config/` directory contains the config files required to run optimize guidance graphs either by using CMA-ES or PIU.

| Config file                                                                                | MAPF Algorithm | Map             | GGO Method |
| ------------------------------------------------------------------------------------------ | -------------- | --------------- | ---------- |
| config/competition/CMA_ES_PIBT_32x32_random-map_400-agents.gin                             | PIBT           | random-32-32-20 | CMA-ES     |
| config/competition/CMA_ES_PIBT_warehouse-33x36_400-agents.gin                              | PIBT           | warehouse-33-36 | CMA-ES     |
| config/competition/CMA_ES_PIBT_room-64-64-8_1500-agents.gin                                | PIBT           | room-64-64-8    | CMA-ES     |
| config/competition/CMA_ES_PIBT_maze-32-32-4_400-agents.gin                                 | PIBT           | maze-32-32-4    | CMA-ES     |
| config/competition/CMA_ES_PIBT_empty-48-48_1000-agents.gin                                 | PIBT           | empty-48-48     | CMA-ES     |
| config/competition/CMA_ES_PIBT_random-64-64-8_1500-agents.gin                              | PIBT           | random-64-64-20 | CMA-ES     |
| config/competition/CMA_ES_PIBT_den312d_1200-agents.gin                                     | PIBT           | den312d         | CMA-ES     |
| config/competition/cnn_iter_update/CMA_ES_PIBT_32x32_random-map_400-agents_iter_update.gin | PIBT           | random-32-32-20 | PIU        |
| config/competition/cnn_iter_update/CMA_ES_PIBT_warehouse-33x36_400-agents_iter_update.gin  | PIBT           | warehouse-33-36 | PIU        |
| config/competition/cnn_iter_update/CMA_ES_PIBT_room-64x64_1500-agents_iter_update.gin      | PIBT           | room-64-64-8    | PIU        |
| config/competition/cnn_iter_update/CMA_ES_PIBT_maze-32-32-4_400-agents_iter_update.gin     | PIBT           | maze-32-32-4    | PIU        |
| config/competition/cnn_iter_update/CMA_ES_PIBT_empty-48-48_1000-agents_iter_update.gin     | PIBT           | empty-48-48     | PIU        |
| config/competition/cnn_iter_update/CMA_ES_PIBT_random-64-64-20_1500-agents_iter_update.gin | PIBT           | random-64-64-20 | PIU        |
| config/competition/cnn_iter_update/CMA_ES_PIBT_den312d_1200-agents_iter_update.gin         | PIBT           | den312d         | PIU        |
| config/warehouse/CMA_ES_RHCR_33x36_human-map_220_agents_one_wait.gin                       | RHCR           | warehouse-33-36 | CMA-ES     |
| config/warehouse/CMA_ES_DPP_17x20_human-map_88_agents.gin                                  | DPP            | warehouse-20-17 | CMA-ES     |

### Optimized Guidance Graphs

The `maps/` directory contains the map files that are optimized by CMA-ES and generated by update models optimized with PIU.

**Note: the guidance graphs here are offered in json files. For more readable/understandable csv files, please go to the [project website](https://yulunzhang.net/publication/zhang2024ggo/).**


| Guidance Graph file                                                                                 | MAPF Algorithm | Map             | GGO Method |
| --------------------------------------------------------------------------------------------------- | -------------- | --------------- | ---------- |
| maps/competition/ours/pibt_random_cma-es_32x32_400_agents_four-way-move.json                        | PIBT           | random-32-32-20 | CMA-ES     |
| maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es_400_agents_four-way-move.json              | PIBT           | warehouse-33-36 | CMA-ES     |
| maps/competition/ours/pibt_room_cma-es_64x64_1500_agents_four-way-move.json                         | PIBT           | room-64-64-8    | CMA-ES     |
| maps/competition/ours/pibt_maze-32-32-4_cma-es_400_agents_four-way-move.json                        | PIBT           | maze-32-32-4    | CMA-ES     |
| maps/competition/ours/pibt_empty-48-48_cma-es_1000_agents_four-way-move.json                        | PIBT           | empty-48-48     | CMA-ES     |
| maps/competition/ours/pibt_random-64-64-20_cma-es_1500_agents_four-way-move.json                    | PIBT           | random-64-64-20 | CMA-ES     |
| maps/competition/ours/pibt_den312d_cma-es_1200_agents_four-way-move.json                            | PIBT           | den312d         | CMA-ES     |
| maps/competition/ours/pibt_random_cma-es_piu-transfer_32x32_400_agents_four-way-move.json           | PIBT           | random-32-32-20 | PIU        |
| maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es-piu-transfer_400_agents_four-way-move.json | PIBT           | warehouse-33-36 | PIU        |
| maps/competition/ours/pibt_room_cma-es-piu-transfer_64x64_1500_agents_four-way-move.json            | PIBT           | room-64-64-8    | PIU        |
| maps/competition/ours/pibt_maze-32-32-4_cma-es-piu_400_agents_four-way-move.json                    | PIBT           | maze-32-32-4    | PIU        |
| maps/competition/ours/pibt_empty-48-48_cma-es-piu_1000_agents_four-way-move.json                    | PIBT           | empty-48-48     | PIU        |
| maps/competition/ours/pibt_random-64-64-20_cma-es-piu_1500_agents_four-way-move.json                | PIBT           | random-64-64-20 | PIU        |
| maps/competition/ours/pibt_den312d_cma-es-piu_1200_agents_four-way-move.json                        | PIBT           | den312d         | PIU        |
| maps/warehouse/ours/kiva_33x36_human_cma-es_opt_220_agents.json                                     | RHCR           | warehouse-33-36 | CMA-ES     |
| maps/warehouse/ours/kiva_dpp_17x20_human_cma-es_opt_88_agents.json                                  | DPP            | warehouse-20-17 | CMA-ES     |

The `maps/` directory also contains the baseline maps.

| Guidance Graph file                                                                    | MAPF Algorithm | Map             | Baseline     |
| -------------------------------------------------------------------------------------- | -------------- | --------------- | ------------ |
| maps/competition/human/pibt_random_unweight_32x32.json                                 | PIBT           | random-32-32-20 | Unweighted   |
| maps/competition/human/pibt_random_unweight_32x32_alternate_baseline_bi-directed.json  | PIBT           | random-32-32-20 | Crisscross   |
| maps/competition/expert_baseline/pibt_random_unweight_32x32_flow_baseline.json         | PIBT           | random-32-32-20 | Traffic Flow |
| maps/competition/expert_baseline/pibt_random_unweight_32x32_HM_baseline.json           | PIBT           | random-32-32-20 | HM Cost      |
| maps/competition/human/pibt_room-64-64-8.json                                          | PIBT           | room-64-64-8    | Unweighted   |
| maps/competition/human/pibt_room-64-64-8_alternate_baseline_bi-directed.json           | PIBT           | room-64-64-8    | Crisscross   |
| maps/competition/expert_baseline/pibt_room-64-64-8_flow_baseline.json                  | PIBT           | room-64-64-8    | Traffic Flow |
| maps/competition/expert_baseline/pibt_room-64-64-8_HM_baseline.json                    | PIBT           | room-64-64-8    | HM Cost      |
| maps/competition/human/pibt_maze-32-32-4.json                                          | PIBT           | maze-32-32-4    | Unweighted   |
| maps/competition/human/pibt_maze-32-32-4_alternate_baseline_bi-directed.json           | PIBT           | maze-32-32-4    | Crisscross   |
| maps/competition/expert_baseline/pibt_maze-32-32-4_flow_baseline.json                  | PIBT           | maze-32-32-4    | Traffic Flow |
| maps/competition/expert_baseline/pibt_maze-32-32-4_HM_baseline.json                    | PIBT           | maze-32-32-4    | HM Cost      |
| maps/competition/human/pibt_empty-48-48.json                                           | PIBT           | empty-48-48     | Unweighted   |
| maps/competition/human/pibt_empty-48-48_alternate_baseline_bi-directed.json            | PIBT           | empty-48-48     | Crisscross   |
| maps/competition/expert_baseline/pibt_empty-48-48_flow_baseline.json                   | PIBT           | empty-48-48     | Traffic Flow |
| maps/competition/expert_baseline/pibt_empty-48-48_HM_baseline.json                     | PIBT           | empty-48-48     | HM Cost      |
| maps/competition/human/pibt_random-64-64-20.json                                       | PIBT           | random-64-64-20 | Unweighted   |
| maps/competition/human/pibt_random-64-64-20_alternate_baseline_bi-directed.json        | PIBT           | random-64-64-20 | Crisscross   |
| maps/competition/expert_baseline/pibt_random-64-64-20_flow_baseline.json               | PIBT           | random-64-64-20 | Traffic Flow |
| maps/competition/expert_baseline/pibt_random-64-64-20_HM_baseline.json                 | PIBT           | random-64-64-20 | HM Cost      |
| maps/competition/human/pibt_den312d.json                                               | PIBT           | den312d         | Unweighted   |
| maps/competition/human/pibt_den312d_alternate_baseline_bi-directed.json                | PIBT           | den312d         | Crisscross   |
| maps/competition/expert_baseline/pibt_den312d_flow_baseline.json                       | PIBT           | den312d         | Traffic Flow |
| maps/competition/expert_baseline/pibt_den312d_HM_baseline.json                         | PIBT           | den312d         | HM Cost      |
| maps/competition/human/pibt_warehouse_33x36_w_mode.json                                | PIBT           | warehouse-33-36 | Unweighted   |
| maps/competition/human/pibt_warehouse_33x36_w_mode_alternate_baseline_bi-directed.json | PIBT           | warehouse-33-36 | Crisscross   |
| maps/competition/expert_baseline/pibt_warehouse_33x36_w_mode_flow_baseline.json        | PIBT           | warehouse-33-36 | Traffic Flow |
| maps/competition/expert_baseline/pibt_warehouse_33x36_w_mode_HM_baseline.json          | PIBT           | warehouse-33-36 | HM Cost      |
| maps/warehouse/human/kiva_large_w_mode.json                                            | RHCR           | warehouse-33-36 | Unweighted   |
| maps/warehouse/human/kiva_large_w_mode_alternate_baseline.json                         | RHCR           | warehouse-33-36 | Crisscross   |
| maps/warehouse/expert_baseline/kiva_large_w_mode_flow_baseline.json                    | RHCR           | warehouse-33-36 | Traffic Flow |
| maps/warehouse/expert_baseline/kiva_large_w_mode_HM_baseline.json                      | RHCR           | warehouse-33-36 | HM Cost      |
| maps/warehouse/human/kiva_small_r_mode.json                                            | DPP            | warehouse-20-17 | Unweighted   |
| maps/warehouse/human/kiva_small_r_mode_alternate_baseline_bi-directed.json             | DPP            | warehouse-20-17 | Crisscross   |
| maps/warehouse/expert_baseline/kiva_small_r_mode_flow_baseline.json                    | DPP            | warehouse-20-17 | Traffic Flow |
| maps/warehouse/expert_baseline/kiva_small_r_mode_HM_baseline.json                      | DPP            | warehouse-20-17 | HM Cost      |

### Optimized Update Models

The `piu_model` directory contains the optimized PIU models.

| Model file                                                                      | MAPF Algorithm | Map             |
| ------------------------------------------------------------------------------- | -------------- | --------------- |
| piu_model/competition/pibt_random_32x32_400_agents_four-way-move.json           | PIBT           | random-32-32-20 |
| piu_model/competition/pibt_warehouse-33x36_w_mode_400_agents_four-way-move.json | PIBT           | warehouse-33-36 |
| piu_model/competition/pibt_room-64x64_1500_agents_four-way-move.json            | PIBT           | room-64-64-8    |
| piu_model/competition/pibt_maze-32-32-4_400_agents_four-way-move.json           | PIBT           | maze-32-32-4    |
| piu_model/competition/pibt_empty-48-48_1000_agents_four-way-move.json           | PIBT           | empty-48-48     |
| piu_model/competition/pibt_random-64-64-20_1500_four-way-move.json              | PIBT           | random-64-64-20 |
| piu_model/competition/pibt_den312d_1200_agents_four-way-move.json               | PIBT           | den312d         |

## Generate and Evaluate Guidance Graphs

### Generate Guidance Graphs with Optimized Update Model

To generate guidance graph with optimized update models, run the following:

```
bash scripts/iter_update_algo.sh MODEL_PATH MAP_PATH PIU_CONFIG \
    N_AGENT_START N_AGENT_STEP N_AGENT_END N_WORKERS DOMAIN
```

- `MODEL_PATH`: path to the update models, as specified above
- `MAP_PATH`: path to the maps
- `PIU_CONFIG`: PIU config files under `config/competition/pure_iter_update` and `config/warehouse/pure_iter_update` directories
- `N_AGENT_START`: start of the number of agents
- `N_AGENT_STEP`: step size of number of agents
- `N_AGENT_END`: end of the number of agents
- `N_WORKERS`: number of processes to run in parallel
- `DOMAIN`: `kiva` for warehouse maps, or `competition` for random and room maps

For example, the following command:

```
bash scripts/piu_gen_highway.sh \
    piu_model/competition/pibt_warehouse-33x36_w_mode_400_agents_four-way-move.json \
    maps/warehouse/human/additional_eval/kiva_human_45x47_w_mode.json \
    config/competition/pure_iter_update/cnn_warehouse_45x47.gin \
    850 50 1501 16 competition
```

generates guidance graphs for the warehouse 45 $\times$ 47 map using the update model optimized with PIBT and the warehouse-33-36 map, starting and ending with 850 and 1501 agents with step size of 50 agents.

### Evaluate Guidance Graphs

After getting the guidance graphs, we want to evaluate the guidance graphs by running simulations. To do so in the provided MAPF simulators, run the following:

```
bash scripts/run_single_sim.sh SIM_CONFIG MAP_FILE AGENT_NUM AGENT_NUM_STEP_SIZE \
    N_EVALS MODE N_SIM N_WORKERS DOMAIN -r RELOAD
```

- `SIM_CONFIG`: gin simulation configuration file, stored under `pure_simulation` directory under `config/<domain>`
- `MAP_FILE`: path of the guidance graph to run the simulations in
- `AGENT_NUM`: number of agent to start with while running simulations
- `AGENT_NUM_STEP_SIZE`: step size of the number of agents to run simulations
- `N_EVALS`: number of evaluations to run, interpreted differently under different modes, see example below
- `MODE`: `inc_agents` or `constant`, see example below
- `N_SIM`: number of simulations to run
- `N_WORKERS`: number of processes to run in parallel
- `DOMAIN`: `kiva` for warehouse maps, or `competition` for random and room maps
- `RELOAD`: optional, log directory to reload an experiment

There are two modes associated with the `MODE` parameter, namely `inc_agents` and `constant`.

**`inc_agents` mode:** the script will run simulations on the provided guidance graph with an increment number of agents. Specifically, starting with `AGENT_NUM`, it increment the number by step size of `AGENT_NUM_STEP_SIZE`, until the number of agents reaches `AGENT_NUM + N_EVALS`. For each number of agents, it runs the simulations `N_SIM` times with seeds from `0` to `N_SIM - 1`. All simulations run in parallel on `N_WORKERS` processes.

For example, the following command:

```
bash scripts/run_single_sim.sh config/competition/pure_simulation/PIBT_warehouse_33x36.gin \
    maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es_400_agents_four-way-move.json \
    100 100 701 inc_agents 50 16 competition
```

runs 50 simulations with 100 to 800 agents, increment in step size of 100, in the guidance graph `maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es_400_agents_four-way-move.json` with simulation config `config/competition/pure_simulation/PIBT_warehouse_33x36.gin` in the warehouse domain.

**`constant` mode**: the script will run simulations on the provided guidance graph with a fixed number of agents with random seeds. Specifically, it runs `N_EVALS` simulations with `AGENT_NUM` agents, in parallel on `N_WORKERS` processes. Other parameters will be ignored but they must be given some dummy values for the script to pick up the correct parameter.

For example, the following command:

```
bash scripts/run_single_sim.sh config/competition/pure_simulation/PIBT_warehouse_33x36.gin \
    maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es_400_agents_four-way-move.json \
    400 1 50 constant 50 16 competition
```

runs 50 simulations with 400 agents in guidance graph `maps/competition/ours/pibt_warehouse-33x36_w_mode_cma-es_400_agents_four-way-move.json` with simulation config `config/competition/pure_simulation/PIBT_warehouse_33x36.gin`.

#### Evaluation Logging Directory Manifest

Running the above scripts will generate separate logging directories under `logs`. The directory's name is of the form
`%Y-%m-%d_%H-%M-%S_<guidance-graph-name>`, e.g.
`2020-12-01_15-00-30_guidance-graph-1`. Inside each directory are the
following files:

```text
- results # contains the configuration and result of all simulations, stored in json.
- map.json # the guidance graph file
```

### Generate Baseline Guidance Graphs

### Crisscross

To generate the crisscross guidance graphs, run the following:

```
bash scripts/gen_crisscross_baseline.sh MAP_FILEPATH STORE_DIR DOMAIN [-b BI_DIRECTED]
```

- `MAP_FILEPATH`: path to the map on which crisscross guidance graph to be generated
- `STORE_DIR`: directory to store the generated crisscross guidance graph
- `DOMAIN`: `kiva` for warehouse maps, or `competition` for random and room maps
- `BI_DIRECTED`: `True` for bi-directed, `False` for strictly directed (i.e. the reverse direction is blocked)

For example, the following command:

```
bash scripts/gen_crisscross_baseline.sh \
    maps/competition/human/pibt_room-64-64-8.json maps/competition/ \
    competition -b True
```

generates a bi-directed crisscross guidance graph for map `maps/competition/human/pibt_room-64-64-8.json` and stores in `maps/competition`.

### Traffic Flow and HM Cost

To generate Traffic Flow and HM Cost guidance graphs, run the following:

```
bash scripts/gen_expert_baseline.sh MAP_FILEPATH STORE_DIR DOMAIN N_ITER
```

- `MAP_FILEPATH`: path to the map on which crisscross guidance graph to be generated
- `STORE_DIR`: directory to store the generated crisscross guidance graph
- `DOMAIN`: `kiva` for warehouse maps, or `competition` for random and room maps
- `N_ITER`: number of single-agent path planning to run

For example, the following command:

```
bash scripts/gen_expert_baseline.sh \
    maps/competition/human/pibt_room-64-64-8.json \
    maps/ \
    competition \
    10
```

generates Traffic Flow and HM Cost guidance graphs for map `maps/competition/human/pibt_room-64-64-8.json` with 10 iterations of single-agent path planning and stores in `maps/`.

## Plotting the Results

### Results

To generate the results shown in the paper, we will reorganize the evaluation logging directories in the following structure:

```text
<eval_exp_name>
|___ Meta Directory of guidance graph 1
     |__ Evaluation Logging Directory1
     |__ Evaluation Logging Directory2
     |__ meta.yaml
|___ Meta Directory of guidance graph 2
     |__ Evaluation Logging Directory1
     |__ Evaluation Logging Directory2
     |__ meta.yaml
```

The evaluation logging directories are obtained by running the aforementioned script. The parent meta directories musted be created to group different evaluation logging directories. A `meta.yaml` file must be created under each meta directorty. An example `meta.yaml` is as follows:

```
algorithm: "RHCR"
map_size: "large"
mode: "w"
map_from: "CMA-ES"
```

where `algorithm` is the lifelong MAPF algorithm, `map_size` is the size of the guidance graph, `mode` is the mode of simulation (`w` for warehouse-33-36 and `r` for warehouse 20 $\times$ 17), and `map_from` is the algorithm used to optimize the guidance graph.

Then, the following script can be used to plot throughput vs. number agents and generate numerical results shown in the paper:

```
bash scripts/plot_throughput.sh <eval_exp_name> cross_n_agents
```

## License

The code is released under the MIT License, with the following exceptions:
- `RHCR/` are adapted and modified from [Rolling-Horizon Collision Resolution (RHCR) repo](https://github.com/Jiaoyang-Li/RHCR) under [USC – Research License](https://github.com/Jiaoyang-Li/RHCR/blob/master/license.md).
