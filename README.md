# Adversarial Turing Patterns
This repository contains the source code for our AAAI'21 paper titled [Adversarial Turing Patterns from Cellular Automata](https://arxiv.org/abs/2011.09393 "Adversarial Turing Patterns from Cellular Automata").

Repository structure:

* `experiments.ipynb` contains the whole pipeline of experiments
* `true_turing_patterns.ipynb` generates true Turing patterns via cellular automata and draws examples
* `drawing.py` contains utils used for drawing patterns in the notebooks
* `turing.py` contains code for cellular automata Turing pattern generation
* `turing_dft.py` contains code for DFT of the patterns
* `turing_224_true` folder contains Turing patterns generated via `true_turing_patterns.ipynb`
* `npy_results` folder contains `*.npy` files of patterns and kernels generated in the experiments
* `csv_transferability` folder contains `*.csv` files with transferability tables of different patterns
