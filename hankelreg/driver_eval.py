import pickle

import jax.numpy as jnp
import optax
from flax import nnx

import hankelreg.optimizer as hopt
from hankelreg.dataset import get_dataset
from hankelreg.model import SSM
from hankelreg.ssm_reduction import reduce_ssm

if __name__ == "__main__":
  print('Modules loaded')

  with open("data/regularized.pkl", "rb") as f:
    results = pickle.load(f)

  print('Pickle read')
  cfg = results['cfg']

  model = SSM.from_config(cfg.ssm, nnx.Rngs(0))
  train, test = get_dataset(cfg.data, num_workers=1)
  metrics = hopt.get_metrics()
  opt = hopt.get_optimizer(model, len(train) * cfg.data.epochs, cfg.opt)
  graphdef, _ = nnx.split((model, opt, metrics))
  state = nnx.from_tree(results['state'])
  for perc in [0.99, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]:
    model, optimizer, metrics = nnx.merge(graphdef, state)
    print('Model defined')
    model.eval()
    print(model.hankel_singvals().shape)
    model, ranks = reduce_ssm(model, model.hankel_singvals(), perc)
    print(ranks)

    @nnx.jit
    def call(model, x):
      return model(x)

    for batch in test:
      x = jnp.reshape(jnp.array(batch[0]), (len(batch[0]), -1, 1))
      y = jnp.array(batch[1])
      y_pred = call(model, x)
      l = optax.softmax_cross_entropy_with_integer_labels(y_pred, y)
      metrics.update(loss=l, logits=y_pred, labels=y)

    print('Truncation Ratio: ', 1 - perc, 'Metrics: ', metrics.compute())
