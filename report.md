# Case Study Report – Self-Pruning Neural Network

## Why L1 Regularization Encourages Sparsity

L1 regularization penalizes the absolute values of parameters. When applied to gate values (bounded between 0 and 1 via sigmoid), it encourages many of them to shrink towards zero. This effectively disables the corresponding weights, leading to a sparse network.

---

## Results Summary

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.0001 | 44.93%   | 1.56%    |
| 0.001  | 41.67%   | 1.72%    |
| 0.01   | 38.14%   | 1.72%    |

---

## Observations

* Increasing λ reduces accuracy, as expected due to stronger regularization
* However, sparsity does not significantly increase
* This indicates insufficient pruning pressure on gate values

---

## Analysis

The low sparsity suggests:

* λ is too small compared to the magnitude of loss
* Gates are not being driven close enough to zero
* More aggressive regularization or longer training is required

---

## Conclusion

The implementation correctly integrates pruning into training.
However, achieving effective sparsity requires careful tuning of λ and training duration.
