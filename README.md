# CounterNet

**This repository is only avaiable during the ICML 2021 reviewing phrase.**

## Install the Package

```
pip install nbdev jupyter
pip install -e .
```

## Explanations for different directory

- `counterfactual`: main python source code in the paper 
- `data`: datasets used in the paper
- `docs`: ignore this folder
- `nbs`: source code in jupyter notebooks; empowered by `nbdev`
- `results`: raw results in the paper
- `saved_weights`: store trained models
- `scripts`: some additional scripts 

## Some useful commands

### Start tensorboard

```
tensorboard --logdir log --bind_all
```

### Build nbs to module

```
nbdev_build_lib
```

### Update nbs from module
```
nbdev_update_lib
```

### clean notebooks
```
nbdev_clean_nbs
```
