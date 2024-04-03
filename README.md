# Sola

Sola is a Python package for ...

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
## Running the Tests

After installing Sola, you can verify your installation by running the test suite. Here's how you can do it:

1. Navigate to the directory containing the `sola` package:

```bash
cd /path/to/sola
```
## Usage

Here's a simple example of how to use Sola:

```python
from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes import functions

domain = HyperParalelipiped([[0, 1]], fineness=100)
f = functions.Gaussian_Bump_1D(domain=domain, center=0.5, width=0.1)