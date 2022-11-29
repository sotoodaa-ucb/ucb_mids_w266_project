# MIDS W266 Project: Natural Language Processing

![flake8](https://github.com/sotoodaa-ucb/ucb_mids_w266_project/actions/workflows/flake8.yml/badge.svg)

## Overview
Install requirements:
```
pip install -r requirements.txt
```

There are known issues with installing dependencies on M1 Mac.
```
# Problem
error: can't find Rust compiler

# Solution
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
```

## Members
- Qian Qiao
- Sophie Yeh
- Andrew Sotoodeh
