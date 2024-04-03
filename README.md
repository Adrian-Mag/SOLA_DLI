# Sola

Sola is a Python package for solving Backus-Gilbert SOLA inference problems.

As of right now, only 1D problems with no data errors are supported. Functionality for dealing with data errors and 2D/3D models will be implemented soon.

## Installation

The easiest way to install sola is using conda. The following commands will
create a new conda environment called "sola" (name can be changed from
environment.yml before installation) with all the required dependancies and the
sola package. Go the the directory in which you want to install sola and run:

```bash
git clone https://github.com/Adrian-Mag/SOLA_DLI.git
cd SOLA_DLI
conda env create -f environment.yml
conda activate sola
```

Creating the conda environment might take a few minutes. After the installation
is done, you can run the provided tests (while still in SOLA_DLI) using:

```bash
python -m unittest
```

## Usage
In examples/ there are multiple jupyter notebooks presenting the basic
functionalities of the package.

Within examples/Target_Paper_Examples you can find the codes used to produce the results of paper *insert paper link*
