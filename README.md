<h1 align="center">
 MIDS W266 Project
</h1>

<h1 align="center">
NotebookNet - An Explorative Machine Learning Approach in Code Comprehension
</h1>

![flake8](https://github.com/sotoodaa-ucb/ucb_mids_w266_project/actions/workflows/flake8.yml/badge.svg)

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

![test](./res/example_1.png)

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