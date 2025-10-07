from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class Config:
  tag: str = 'default'


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
