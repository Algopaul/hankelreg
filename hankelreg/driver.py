import logging
import pickle
from pathlib import Path

import hydra
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

import hankelreg.optimizer as hopt
from hankelreg.config import Config
from hankelreg.dataset import get_dataset
from hankelreg.model import SSM
from hankelreg.util import log_duration

results = {}


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration()
def main(cfg: Config) -> None:
  logging.info("\n%s", OmegaConf.to_yaml(cfg))
  model = SSM.from_config(cfg.ssm, nnx.Rngs(0))
  train, test = get_dataset(cfg.data)
  metrics = hopt.get_metrics()
  opt = hopt.get_optimizer(model, len(train) * cfg.data.epochs, cfg.opt)
  train_step, graphdef, state = hopt.get_train_step(model, metrics, opt, cfg)
  eval_step = hopt.get_eval_step(cfg)

  for i_epoch in range(cfg.data.epochs):
    pbar = tqdm(train, desc=f"Epoch {i_epoch + 1} / {cfg.data.epochs}")
    for batch in pbar:
      x = jnp.reshape(jnp.array(batch[0]), (len(batch[0]), -1, 1))
      y = jnp.array(batch[1])
      loss_val, state, _ = train_step(graphdef, state, x, y)
      pbar.set_postfix({"loss": f'{loss_val:.4f}'})

    model, opt, metrics = nnx.merge(graphdef, state)
    tm_train = metrics.compute()
    metrics.reset()

    for batch in test:
      x = jnp.reshape(jnp.array(batch[0]), (len(batch[0]), -1, 1))
      y = jnp.array(batch[1])
      state = eval_step(graphdef, state, x, y)
    _, _, metrics = nnx.merge(graphdef, state)
    tm_test = metrics.compute()
    logging.info(
        'After epoch %i [train] loss %.3e acc %.2f%% [test] loss %.3e acc %.2f%% [hsvsum] %.2e',
        i_epoch + 1,
        tm_train['loss'],
        100 * tm_train['accuracy'],
        tm_test['loss'],
        100 * tm_test['accuracy'],
        jnp.sum(model.hankel_singvals()),
    )
    metrics.reset()

  results['state'] = nnx.to_pure_dict(state)
  results['cfg'] = cfg
  filename = f"data/{cfg.outfile}.pkl"
  Path(filename).parent.mkdir(exist_ok=True, parents=True)
  with open(filename, "wb") as f:
    pickle.dump(results, f)

  pass


if __name__ == "__main__":
  main()
