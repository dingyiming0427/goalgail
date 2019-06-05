# Goal-conditioned Imitation Learning

## Environment setup
`conda env create -f environment.yaml`

## Run Experiment

The following command will run the three experiments as in Fig. 3 in the paper for both four rooms env and Fetch Pick&Place env.

Four rooms env: `python sandbox/young_clgan/experiments/goals/maze/maze_her_gail.py`

Fetch Pick and Place env: `python sandbox/young_clgan/experiments/goals/pick_n_place/pnp.py`

## Plot Learning Curves

The following command will reproduce the learning curves for two environments as in Fig. 3 in the paper.

Four rooms env: `python plotting/gail_plot.py data/s3/fourroom fourroom`

Fetch Pick and Place env: `python plotting/gail_plot.py data/s3/fetchpnp fetchpnp`

The generated figures can be found in folder `figures`.
