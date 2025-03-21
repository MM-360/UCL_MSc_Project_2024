# UCL_MSc_Project_2024, final 2024-09-09
Repository for UCL MSc project - *Forecasting Changes in Antarctic Sea Ice using Latent Variable Techniques*

This repository contains all the notebooks and helper files created for my MSc Project.

All code was written using Python version 3.12.3 using miniconda. 

PC used for development:
* Windows 11 Pro
* NVIDIA RTX 4090 FE
* 64GB RAM
* Intel Corei7 14700K


# Install

To install the dependencies please install the environment.yaml file to a new conda environment or if you prefer pip please use the requirements.txt

For conda users:
```bash
conda create --name myenv
conda activate myenv
conda install pip
```
To load the enviornment file
```bash
conda env update -f environment.yml --prune
```
# Directory structure

`/data`: Raw data can be placed here, file location will need to be changed in *get_ice_data* function within the *sic_data_functions.py* file

`/images` Diagrams used in the thesis, citations are provided in the thesis

`/inputs` contains inputs for the models.

`/notebooks` contains research notebooks. 

`/outputs` contains outputs from the models.

`/src` contains `.py` dependencies/helper files. 


