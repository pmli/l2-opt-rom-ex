# Code for Numerical Experiments in "$\mathcal{L}_2$-optimal Reduced-order Modeling Using Parameter-separable Forms"

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pmli/l2-opt-rom-ex/HEAD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6618116.svg)](https://doi.org/10.5281/zenodo.6618116)

This repository contains code for numerical experiments reported in

> P. Mlinarić, S. Gugercin,
> **$\mathcal{L}_2$-optimal Reduced-order Modeling Using Parameter-Separable
> Forms**,
> [*arXiv preprint*](https://arxiv.org/abs/2206.02929),
> 2022

## Installation

To run the examples, at least Python 3.6 is needed
(the code was tested using Python 3.7.11).

The necessary packages are listed in [`requirements.txt`](requirements.txt).
They can be installed in a virtual environment by, e.g.,

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## Running the Experiments

The experiments are given as `runme_*.py` scripts.
They can be opened as Jupyter notebooks via
[`jupytext`](https://jupytext.readthedocs.io/en/latest/)
(included when installing via [`requirements.txt`](requirements.txt)).

## Author

Petar Mlinarić:

- affiliation: Virginia Tech
- email: mlinaric@vt.edu
- ORCiD: 0000-0002-9437-7698

## License

The code is published under the MIT license.
See [LICENSE](LICENSE).
