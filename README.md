# Regularization for Deep State Space models

This repository contains code and an illustrative example for performing Hankel singular value based regularization for deep state space models (SSMs), as explored in our NeurIPS paper
```bibtex
@article{SchwerdtnerBP2025Hankel,
    authors={Paul Schwerdtner and Jules Berman and Benjamin Peherstorfer},
    title={Hankel Singular Value Regularization for Highly Compressible State Space Models}
}
```

To get a quick comparison between the regularized and unregularized cases, you can train two SSMs via
```bash
python hankelreg/driver.py data.epochs=250 opt.hsv_regmag=1e-5 outfile='regularized'
python hankelreg/driver.py data.epochs=250 opt.hsv_regmag=0.0 outfile='unregularized'
```
and then compare the accuracies after state truncation via
```bash
python hankelreg/driver_eval.py
```

This will display the accuracies after applying truncation to the ratios `[0.5, ..., 0.9]`. Note that for the regularized case, the high accuracy is regained even for large truncation ratios.

| truncation ratio | regularized  | unregularized |
| ---------------- | -----------  | ------------- |
|       0.5        |   99.50%     |   **99.57%**  |
|       0.6        | **99.50%**   |     98.42%    |
|       0.7        | **99.50%**   |     66.68%    |
|       0.8        | **99.47%**   |     44.57%    |
|       0.9        | **92.73%**   |     20.75%    |


Details of the model architecture can be found in `hankelreg/model.py`, Hankel singular value is implemented in `hankelreg/system_theory.py` and the balancing-based model reduction accross several layers is in `hankelreg/ssm_reduction.py`.
