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
# ENZYMES
python -m train --bmname=ENZYMES --max-nodes=100 --num-classes=6

# ENZYMES - Diffpool
python -m train --bmname=ENZYMES --assign-ratio=0.1 --hidden-dim=30 --output-dim=30 --num-classes=6 --method=soft-assign

# DD
python -m train --bmname=DD --max-nodes=500 --epochs=1000 --num-classes=2

# DD - Diffpool
python -m train --bmname=DD --assign-ratio=0.1 --hidden-dim=64 --output-dim=64 --num-classes=2 --method=soft-assign
```

Performance
-----------
DD 78.33% (with early stopping)

ENZYMES 68.33% (with early stopping)

**Accuracy**

Each experiment with improved model is only conducted once, thus the result may has noise.

|         |   Original |   Improved |
| ------- | ---------: | ---------: |
| DD      | **79.31%** |     78.33% |
| ENZYMES |     63.33% | **68.33%** |
