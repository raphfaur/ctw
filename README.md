# CTW

## Overview

`ctw` is a Python library for online Context Tree Weighting models applied to autoregressive time series. The repository delivers production-ready implementations for univariate and bivariate AR tree models, quantization utilities, and notebook-driven demonstrations.

## Features

- Online autoregressive model learning with recursive context tree weighting
- `LiveARTree` for univariate AR series
- `BivariateARTree` and `BivariateARTreeBis` for paired series and bivariate regimes
- Quantization utilities for continuous-to-discrete encoding
- Tree construction and visualization helpers using Graphviz and AnyTree
- Ready-to-run example notebooks for model evaluation and visualization

## Tech Stack

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Graphviz
- AnyTree
- Pandas (notebook support)
- Jupyter Notebook

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-org>/ctw.git
cd ctw
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\\Scripts\\activate
```

3. Install dependencies:

```bash
python -m pip install numpy scipy matplotlib graphviz anytree pandas jupyter
```

## Running locally

The library is designed for interactive experimentation and local model development.

1. Launch Jupyter Notebook from the repository root:

```bash
python -m notebook
```

2. Open one of the example notebooks:

- `live_ar_tree_demo.ipynb`
- `bivariate_synthetic_data.ipynb`
- `bivariate_pollution.ipynb`
- `unemployment_rate.ipynb`
- `meteo.ipynb`
- `sunspot.ipynb`

3. Or import the package in a Python session:

```python
from ctw.models import LiveARTree, Quantizer

quantizer = Quantizer(thresholds=[-1.0, 0.0, 1.0])
model = LiveARTree(order=2, max_depth=3, quantizer=quantizer, beta=0.5)

sequence = [0.1, -0.2, 0.5, 1.0, -0.1, 0.3]
for x in sequence:
    model.observe(x)

prediction, node_id = model.predict([0.5, -0.1])
print('Next value:', prediction)
```

## Usage examples

### Univariate AR Tree

```python
from ctw.models import LiveARTree, Quantizer

quantizer = Quantizer(thresholds=[-0.5, 0.0, 0.5])
model = LiveARTree(max_depth=2, order=2, quantizer=quantizer, beta=0.5)

for x in data:
    model.observe(x)

prediction, node_id = model.predict([0.2, -0.1])
```

### Bivariate AR Tree

```python
from ctw.models import BivariateQuantizer, BivariateARTree
import numpy as np

thresholds = np.array([[0.0, 0.5], [0.0, 0.5]])
quantizer = BivariateQuantizer(thresholds=thresholds)
model = BivariateARTree(order=2, max_depth=3, quantizer=quantizer, beta=0.5)

for x, y in zip(x_series, y_series):
    model.observe(x, y)

prediction, node_id = model.predict(np.array([[x1, x2], [y1, y2]]))
```

### Data visualization and utilities

`utils.py` provides AR fitting, prediction, regime detection, and plotting helpers for time series evaluation.

## Branch comparison

- `main`: Contains the baseline univariate Context Tree Weighting autoregressive implementation. It focuses on single-series online AR modeling with `LiveARTree` and the univariate tree builder.
- `bivariate_bis`: Extends the repository with explicit bivariate support, including `BivariateARTree`, `BivariateARTreeBis`, `BivariateQuantizer`, and new tree-building logic for mixed x/y contexts. This branch is used for paired-series experiments and multivariate regime modeling.

## Repository structure

- `ctw/`
  - `__init__.py` - package marker
  - `models.py` - core AR tree and quantization models
  - `tree.py` - tree structures and visualization utilities
  - `series.py` - autoregressive sequence generator
- `utils.py` - plotting and helper utilities
- `*.ipynb` - example notebooks for demos and experiments
- `PRSA_data_2010.1.1-2014.12.31.csv` - sample dataset used by notebooks
- `.gitignore` - ignores `.env` and Python cache files

## Available scripts

This repository is designed for direct import and notebook-driven use. No additional CLI or packaged scripts are required.

## Deployment

The project is production-ready for local experimentation and model development. Standard Python tooling supports package installation and notebook execution.
