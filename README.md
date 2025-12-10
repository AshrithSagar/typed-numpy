# typed-numpy

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Typed NumPy with static shape typing and runtime shape validation.

> [!WARNING]
> Experimental & WIP

## Installation

<details>

<summary>Install uv</summary>

Install [`uv`](https://docs.astral.sh/uv/), if not already.
Check [here](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

It is recommended to use `uv`, as it will automatically install the dependencies in a virtual environment.
If you don't want to use `uv`, skip to the next step.

**TL;DR: Just run**

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

<details>

<summary>Install the package</summary>

The dependencies are listed in the [pyproject.toml](pyproject.toml) file.

Install the package from the PyPI release:

```shell
# Using uv
uv add typed-numpy

# Or with pip
pip3 install typed-numpy
```

To install from the latest commit:

```shell
uv add git+https://github.com/AshrithSagar/typed-numpy.git@main
```

</details>

## License

This project falls under the [MIT License](LICENSE).
