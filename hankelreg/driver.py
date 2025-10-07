import logging

import hydra
from omegaconf import OmegaConf
from hankelreg.config import Config
from hankelreg.util import log_duration


@hydra.main(version_base=None, config_name='config', config_path='../conf')
@log_duration()
def main(cfg: Config) -> None:
  logging.info("\n%s", OmegaConf.to_yaml(cfg))
  pass


if __name__ == "__main__":
  main()
