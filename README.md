# Sola

Sola is a Python package for ...

## Installation

To install Sola, you can ...

## Running the Tests

After installing Sola, you can verify your installation by running the test suite. Here's how you can do it:

1. Navigate to the directory containing the `sola` package:

```bash
cd /path/to/sola

## Usage

Here's a simple example of how to use Sola:

```python
from sola.main_classes.domains import HyperParalelipiped
from sola.main_classes import functions

domain = HyperParalelipiped([[0, 1]], fineness=100)
f = functions.Gaussian_Bump_1D(domain=domain, center=0.5, width=0.1)