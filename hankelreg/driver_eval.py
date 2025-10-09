import logging
import pickle
from pathlib import Path
from time import time

import jax.numpy as jnp
import optax
import pandas as pd
from flax import nnx

import hankelreg.optimizer as hopt
from hankelreg.dataset import get_dataset
from hankelreg.model import SSM
from hankelreg.ssm_reduction import reduce_ssm

if __name__ == "__main__":
  datafiles = ['data/regularized.pkl', 'data/unregularized.pkl']
  trunc_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
  df = pd.DataFrame({
      "truncation ratio": trunc_ratios,
      Path(datafiles[0]).stem: jnp.nan,
      Path(datafiles[1]).stem: jnp.nan,
  })

  for datafile in datafiles:
    with open(datafile, "rb") as f:
      results = pickle.load(f)

    cfg = results['cfg']
    model = SSM.from_config(cfg.ssm, nnx.Rngs(0))
    train, test = get_dataset(cfg.data, num_workers=1)
    metrics = hopt.get_metrics()
    opt = hopt.get_optimizer(model, len(train) * cfg.data.epochs, cfg.opt)
    graphdef, _ = nnx.split((model, opt, metrics))
    state = nnx.from_tree(results['state'])
    accuracies = []
    for i, ratio in enumerate(trunc_ratios):
      perc = 1.0 - ratio
      model, optimizer, metrics = nnx.merge(graphdef, state)
      model.eval()
      model, ranks = reduce_ssm(model, model.hankel_singvals(), perc)

      @nnx.jit
      def call(model, x):
        return model(x)

      for batch in test:
        x = jnp.reshape(jnp.array(batch[0]), (len(batch[0]), -1, 1))
        y = jnp.array(batch[1])
        y_pred = call(model, x)
        l = optax.softmax_cross_entropy_with_integer_labels(y_pred, y)
        metrics.update(loss=l, logits=y_pred, labels=y)

      t0 = time()
      for batch in test:
        y_pred = call(model, x)
      jax.block_until_ready(y_pred)
      t1 = time()
      print('Runtime at ', ratio, 'is ', t1 - t0)
      m = metrics.compute()
      df.loc[i, Path(datafile).stem] = m['accuracy']

  print(df.to_string(index=False))
