****Fundamental Recovery Bounds for SPAD Signals under Stationary Flux****

  
This repository contains the implementation for simulating and reconstructing Single-Photon Avalanche Diode (SPAD) sensor signals using diffusion models.

**Installation**  
Before running the project, set up your Python environment and install the necessary dependencies.

Using pip:  
`pip install -r requirements.txt`

Model Checkpoints  
A pre-trained diffusion model checkpoint is required to execute the scripts.

Download the checkpoint from this Google Drive link:  
https://drive.google.com/file/d/10OsRRxHqSyzxXvbmgyt9Kc3CfFQDxZkB/view?usp=sharing

Create a "models" directory in the root of the repository (if it doesn't already exist).

Place the downloaded checkpoint file directly inside the "models" folder.

**Running the Script**  
We have provided a shell script to easily run the model with the correct configuration files (model_config.yaml, diffusion_config.yaml, and spad_config.yaml).

To run the simulation locally, run the following commands in your terminal:

`chmod +x run_spad.sh`  
`./run_spad.sh`

**Acknowledgments and References**   
This code is built on and utilizes the following open-source projects and datasets:
* **Diffusion Posterior Sampling (DPS):** [https://github.com/dps2022/diffusion-posterior-sampling](https://github.com/dps2022/diffusion-posterior-sampling)
* **OpenAI Guided Diffusion** (used to train the neural network): [https://github.com/openai/guided-diffusion](https://github.com/openai/guided-diffusion)
* **FFHQ Dataset:** [https://github.com/nvlabs/ffhq-dataset](https://github.com/nvlabs/ffhq-dataset)

**Citing Our Work**  
You are welcome to use our database and code, and to adapt them for your own work. If you do, please cite and credit our paper:

> Lior Dvir, Nadav Torem, Mohit Gupta, and Yoav Y. Schechner, "Fundamental Recovery Bounds for SPAD Signals under Stationary Flux," *Proceedings IEEE International Conference on Computational Imaging (ICCP)*, 2026.

Please also provide a link to the source page: [https://zenodo.org/records/20858184](https://zenodo.org/records/20858184)
