from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class SSMConfig:
  channels_in: int = 1
  channels_out: int = 10
  state_dim: int = 128
  io_dim: int = 128
  layers: int = 4
  use_skip: bool = True
  dropout_rate: float = 0.1
  angle_shift: float = 0.0
  angle_scale: float = 1.0
  retention_shift: float = 1.0
  retention_scale: float = 0.25
  reduce_mean: bool = True


@dataclass
class DatasetConfig:
  name: str = 'mnist'
  epochs: int = 50
  batch_dim: int = 64


@dataclass
class OptimizerConfig:
  eval_interval: int = 1000
  optimizer_name: str = 'adamw'
  sequence_optimizer_name: str = 'adam'
  learning_rate: float = 1e-3
  final_learning_rate: float = 1e-5
  weight_decay: float = 5e-2
  peak_learning_rate: float | None = None
  peak_sequence_learning_rate: float | None = None
  final_sequence_learning_rate: float | None = None
  warmup_steps: int = 500
  warmup_start_learning_rate: float = 0.0
  hsv_regmag: float = 1e-5
  loss_fn: str = 'crossentr_integer'


@dataclass
class Config:
  tag: str = 'default'
  ssm: SSMConfig = field(default_factory=SSMConfig)
  data: DatasetConfig = field(default_factory=DatasetConfig)
  opt: OptimizerConfig = field(default_factory=OptimizerConfig)
  outfile: str = 'regularized'


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
