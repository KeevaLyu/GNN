Hierarchical Graph Representation Learning with Differentiable Pooling
============


Paper link: [https://arxiv.org/abs/1806.08804](https://arxiv.org/abs/1806.08804)

Author's code repo: [https://github.com/RexYing/diffpool](https://github.com/RexYing/diffpool)


Requirements
------------
* PyTorch 1.0+

How to run
----------
```bash
# ENZYMES - Diffpool
python -m simple_train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6

# DD - Diffpool
python -m simple_train --bmname=DD --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --num-classes=2
```

