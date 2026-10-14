# Native Mojo distribution

This directory builds `mojolearn.mojoc` for direct use from Mojo on Apple silicon
and Linux. It is an additional distribution alongside the PyPI wheels. Python
is used only by the package builder and packaging tests; native callers import
Mojo modules and execute the compiled code directly.

The default is `identical`, with no compiler flag required.
In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
`fast` is Apple only: opt in with `mojo run -D MOJOLEARN_NUMERIC_FAST=1 app.mojo`
(or the same flag with `mojo build`). Conflicting mode flags are rejected.
Choose the mode when compiling the application; there is one package for both.

## Build and use locally

Use the repository's locked Mojo 1.0.0 / MAX 26.5.0 environment:

```sh
PREFIX="$PWD/native-install" pixi run bash packaging/native/build.sh
pixi run mojo run -I "$PWD/native-install/lib/mojo" packaging/native/tests/cpu.mojo
pixi run mojo run -I "$PWD/native-install/lib/mojo" packaging/native/tests/gpu.mojo
```

To retain installed-package checks outside the checkout, including both Apple
modes and NVIDIA/AMD compile checks, run on an Apple GPU machine:

```sh
pixi run bash packaging/native/validate.sh "$PWD/native-install" /tmp/mojolearn-native-checks --gpu --apple-fast --cross-compile
```

Use a fresh output directory. Without those optional flags the command runs CPU
and mode-selection checks only. Cross-compilation does not run the GPU program.

The GPU example fits k-means and predicts labels using `DeviceContext` and native
buffers. The CPU example exercises matrix multiplication, an optimizer update,
and model configuration. For example:

```mojo
from mojolearn.core.host_gemm_simd import host_gemm_identical
from mojolearn.gemm.contract import OP_NN

def main():
    var a: List[Float32] = [1.0, 2.0]
    var b: List[Float32] = [3.0, 4.0]
    var result = host_gemm_identical(a, b, OP_NN, 1, 1, 2)
    print(result[0])
```

`entrypoints.txt` declares the native module entry points. The builder includes
their complete dependency closure, preserves their directory structure beneath
`mojolearn`, relocates repository-local imports, selects the distribution default,
and renames embedded standalone checks' `main` functions to private helpers.
The tokenizer table is generated using its existing pinned Unicode generator.
These are the existing native function and buffer interfaces, rather
than Python class wrappers. Adding another entry point requires its native
dependencies to resolve; a Python import causes the package build to fail.

`$PREFIX/share/mojolearn/native-sources.json` records every original and packaged
source hash. The package uses Mojo's supported non-elaborated `.mojoc` format;
the application's compilation generates code for its target. The exact compiler
and MAX versions are pinned in the recipe. This does not make unsupported GPU
architectures supported or provide prebuilt kernels for arbitrary hardware.

## Community channel submission

Generate a recipe from a committed, pushed revision containing the packager:

```sh
python3 packaging/native/render_recipe.py /path/to/modular-community/recipes/mojolearn/recipe.yaml --revision HEAD
```

The output pins the full Git commit and takes the version and recipe from that
same commit. Submit this directory to
[modular-community](https://github.com/modular/modular-community). Maintainers
review the recipe and enable their CI; merging there publishes the package.
Merging this repository alone does not publish it to their channel.

After publication, consumers can add the package to a Pixi project with:

```sh
pixi add -c https://repo.prefix.dev/modular-community -c https://conda.modular.com/max -c conda-forge mojolearn
pixi run mojo run app.mojo
```

No `-I` is needed when the package is installed into the active environment's
`lib/mojo`. For a local conda build, render with `--local`, then run
`rattler-build build -r /path/to/recipe.yaml -c https://conda.modular.com/max -c conda-forge`.

## Validation boundaries

The recipe runs an installed-package CPU test on each build platform. GPU tests
are separate because the community builders do not provide NVIDIA/AMD devices.
Native packaging does not replace the existing release identity evidence, and
a successful precompile or cross-compile does not establish device execution.
Retain full build/test logs and report each platform and mode separately.
The manual `Native Mojo package` GitHub workflow builds and tests the conda
package on macOS arm64, Linux x86-64, and Linux arm64 and retains its artifacts
and full logs. It does not publish a channel package.
