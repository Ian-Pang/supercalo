# SuperCalo: Calorimeter shower super-resolution
## by Ian Pang, John Andrew Raine and David Shih

This repository contains the source code for reproducing the results of

_"SuperCalo: Calorimeter shower super-resolution"_ by Ian Pang, John Andrew Raine and David Shih, [arxiv: 2308.11700](https://arxiv.org/abs/2308.11700)

### Running Flow-1, Flow-2 and SuperCalo A

Note that `--which_flow` specifies which subset of the 3 flows we are working with. The subset of flows to be worked with are encoded in a binary sum: Flow-1 is contributes 1, Flow-2 contributes 2, and SuperCalo A contributes 4. Training all 3 flows means `--which_flow` is 1+2+4=7. Only training flow 1 and 3 means `--which_flow` is 1+4=5, etc.

#### Flow-1 (Layer energy flow)
To train Flow-1, run

`python run_supercalo.py --which_ds 2 --which_flow 1 --train --output_dir /path/to/output_directory --data_dir /path/to/data_directory --with_noise`

To generate from the Flow-1, run

`python run_supercalo.py --which_ds 2 --which_flow 1 --generate --output_dir /path/to/output_directory`

#### Flow-2 (Coarse voxel flow)

To train Flow-2, run

`python run_supercalo.py --which_ds 2 --which_flow 2 --train --output_dir /path/to/output_directory --data_dir /path/to/data_directory --with_noise --noise_level 5e-3`

To generate coarse showers from Flow-2 given true conditional inputs, run

`python run_supercalo.py --which_ds 2 --which_flow 2 --generate --output_dir /path/to/output_directory --data_dir /path/to/data_directory --with_noise --noise_level 5e-3`

#### SuperCalo A

To train SuperCalo A, run

`python run_supercalo.py --which_ds 2 --which_flow 4 --train --output_dir /path/to/output_directory --data_dir /path/to/data_directory --with_noise --noise_level 1e-4`

To upsample from true coarse voxels using SuperCalo A, run

`python run_supercalo.py --which_ds 2 --which_flow 4 --upsample --output_dir /path/to/output_directory --data_dir /path/to/data_directory --with_noise --noise_level 1e-4`

#### Full chain (Flow-1 + Flow-2 + SuperCalo A)

To generate using full chain, run

`python run_supercalo.py --which_ds 2 --which_flow 4 --generate --output_dir /path/to/output_directory`