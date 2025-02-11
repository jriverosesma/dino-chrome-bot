[![Unit tests](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml/badge.svg)](https://github.com/jriverosesma/python-project/actions/workflows/unit_tests.yaml)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/jriverosesma/python-project?include_prereleases&label=latest%20release&color=blue)](https://github.com/jriverosesma/python-project/releases)
[![PyPI version](https://img.shields.io/pypi/v/python-project.svg?color=blue)](https://pypi.org/project/python-project/)

# Python Project Template

*Python Project Template: Simple and effective*
________________________________________________________

## Table of contents
1. [Overview](README.md#1-overview)  
2. [Installation](README.md#2-installation)  
    2.1 [Users](README.md#2.1-users)  
    2.2 [Developers](README.md#2.2-developers)  
3. [Features](README.md#3-features)

## 1. Overview

This is a simple and effective Python project template.

## 2. Installation

### 2.1 Users

```bash
pip install my-package
```

### 2.2 Developers

```bash
conda create -y -n my-package python=3.10 --no-default-packages
conda activate my-package
python -m pip install --upgrade pip
pip install -e .[all]
pre-commit install
```

## 3. Features

- Template `pyproject.toml`.
- Sample `README.md` with badges and table of contents.
- Lightweight pre-commits: automatic formatting.
- Simple and generic unit tests and release workflows.
- Issue and PR templates.
- Sample `.gitignore` and `.gitattributes`.
- Sample `.editorconfig`.
- Sample `NOTICES` (generated using `pip-licenses`), `LICENSE` and `CHANGELOG.md`.
