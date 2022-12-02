<h1 align="center">
 MIDS W266 Project
</h1>

<h1 align="center">
NotebookNet - An Explorative Machine Learning Approach in Code Comprehension
</h1>

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Google Drive Models](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1mZILOzPW7pw0siHYzowtgDRLFAMEoKRr)
[![Wandb Badge](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/sotoodaa/w266-project)

![flake8](https://github.com/sotoodaa-ucb/ucb_mids_w266_project/actions/workflows/flake8.yml/badge.svg)
[![PyPi version](https://badgen.net/pypi/v/w266-project/)](https://pypi.org/project/w266-project/)
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)
## Members
- Qian Qiao
- Sophie Yeh
- Andrew Sotoodeh

## Overview
Install requirements:
```
pip install -r requirements.txt
```



## Usage
```
# Start training.
python w266_project/train.py

# Run test inference.
python w266_project/test.py
```


## Examples

![test](./res/example_2.png)

| content                                           |   actual_pct_rank |   predicted_pct_rank |
|:--------------------------------------------------|------------------:|---------------------:|
| import numpy as np                                |          0.333333 |             0.333333 |
| # This adds two numbers and returns the sum.      |          0.166667 |             0.396025  ✅|
| def add(a: int, b: int) -> int: return a + b      |          0.5      |             0.5      |
| def subtract(a: int, b: int) -> int: return a - b |          0.666667 |             0.666667 |
| def multiply(a: int, b: int) -> int: return a * b |          0.833333 |             0.833333 |
| def divide(a: int, b: int) -> int: return a // b  |          1        |             1        |

# Troubleshooting
There are known issues with installing dependencies on M1 Mac.
```
# Problem
error: can't find Rust compiler

# Solution
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
```