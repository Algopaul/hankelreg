import logging

import hydra
import jax.numpy as jnp
from flax import nnx
from omegaconf import OmegaConf

from hankelreg.config import Config
from hankelreg.dataset import get_dataset
from hankelreg.model import SSM
from hankelreg.optimizer import get_optimizer
from hankelreg.util import log_duration


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration()
def main(cfg: Config) -> None:
  logging.info("\n%s", OmegaConf.to_yaml(cfg))
  model = SSM.from_config(cfg.ssm, nnx.Rngs(0))
  train, test = get_dataset(cfg.data)
  opt = get_optimizer(model, len(train) * cfg.data.epochs, cfg.opt)
  print(model.layers[0].sequence_processor.hankel_singvals())

  @nnx.jit
  def call(model, x):
    return model(x)

  for idx, batch in enumerate(train):
    x = jnp.reshape(jnp.array(batch[0]), (len(batch[0]), -1, 1))
    y = jnp.array(batch[1])
    print(idx, call(model, x).shape)

  pass


if __name__ == "__main__":
  main()
