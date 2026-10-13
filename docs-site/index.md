# mojolearn documentation

mojolearn is a GPU machine-learning library written in Mojo whose results are reproducible down to
the bits across GPU vendors. One codebase runs on Apple, NVIDIA and AMD GPUs and on the CPU, and covers
over 100 algorithms and model components, from gradient boosting and forests to transformers and
state space models. `identical` is the default mode. In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Install
:link: install
:link-type: doc
Pick your platform and GPU, install with pip, and check your machine.
:::

:::{grid-item-card} Quickstart
:link: user_guide/quickstart
:link-type: doc
Train a model, predict, and see which device and numeric mode ran.
:::

:::{grid-item-card} The reproducibility contract
:link: notes/reproducibility
:link-type: doc
What `identical` promises, what it does not, and what it costs.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Every public class and function, by family.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

install
user_guide/index
api/index
notes/index
community/index
```
