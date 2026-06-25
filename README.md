Fundamental Recovery Bounds for SPAD Signals of Stationary Poissonian Flux

This repository contains the implementation for simulating and reconstructing Single-Photon Avalanche Diode (SPAD) sensor signals using diffusion models.

Installation
Before running the project, set up your Python environment and install the necessary dependencies.

Using pip:
pip install -r requirements.txt

(Alternatively, if you are using Conda and have exported an environment.yml file, you can run: conda env create -f environment.yml and activate it).

Model Checkpoints
A pre-trained diffusion model checkpoint is required to execute the scripts.

Download the checkpoint from this Google Drive link: https://drive.google.com/file/d/10OsRRxHqSyzxXvbmgyt9Kc3CfFQDxZkB/view?usp=sharing

Create a "models" directory in the root of the repository (if it doesn't already exist).

Place the downloaded checkpoint file directly inside the "models" folder.

Running the Script
We have provided a shell script to easily run the model with the correct configuration files (model_config.yaml, diffusion_config.yaml, and spad_config.yaml).

To run the simulation locally, run the following commands in your terminal:

chmod +x run_spad.sh
./run_spad.sh