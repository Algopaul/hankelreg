import logging

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from hankelreg.config import Config, OptimizerConfig


def get_optimizer(model, train_steps, cfg: OptimizerConfig):
  model_state = nnx.state(model, nnx.Param)
  is_param = lambda x: isinstance(x, nnx.Param)
  paths_and_leaves, treedef = jax.tree_util.tree_flatten_with_path(
      model_state, is_leaf=is_param)

  def label_map(path):
    lkeys = dict_keys(path)
    if any(x in lkeys for x in ['sequence_processor']):
      return 'seq'
    else:
      return 'default'

  def dict_keys(path):
    return [k.key for k in path if isinstance(k, jax.tree_util.DictKey)]

  labels = [label_map(path) for path, _ in paths_and_leaves]
  label_tree = jax.tree_util.tree_unflatten(treedef, labels)
  optimizer_dict = {
      'adam': optax.adam,
      'adamw': optax.adamw,
  }

  learning_rate_normal = optax.warmup_cosine_decay_schedule(
      cfg.warmup_start_learning_rate,
      if_none(cfg.peak_learning_rate, cfg.learning_rate),
      min(cfg.warmup_steps, train_steps) - 1,
      train_steps,
      cfg.final_learning_rate,
  )
  learning_rate_sequence = optax.warmup_cosine_decay_schedule(
      cfg.warmup_start_learning_rate,
      if_none(cfg.peak_sequence_learning_rate, cfg.learning_rate),
      min(cfg.warmup_steps, train_steps) - 1,
      train_steps,
      if_none(cfg.final_sequence_learning_rate, cfg.final_learning_rate),
  )
  tx_seq = optimizer_dict[cfg.sequence_optimizer_name](learning_rate_sequence)
  tx_def = optimizer_dict[cfg.optimizer_name](
      learning_rate_normal,
      weight_decay=cfg.weight_decay,
  )
  transforms = {'seq': tx_seq, 'default': tx_def}
  tx = optax.multi_transform(transforms, label_tree)  # pyright: ignore
  return nnx.Optimizer(model, tx, wrt=nnx.Param)


def get_train_step(model, metrics, optimizer, cfg: Config):
  graphdef, state = nnx.split((model, optimizer, metrics))
  rm = cfg.opt.hsv_regmag

  @jax.jit
  def jax_train_step(graphdef, state, input, target):
    model, optimizer, metrics = nnx.merge(graphdef, state)

    def loss_fn(model):
      y_pred = model(input)
      reg = jnp.sum(model.hankel_singvals())
      l = optax.softmax_cross_entropy_with_integer_labels(y_pred, target)
      return rm * reg + jnp.mean(l), y_pred

    val_grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, model_output), grads = val_grad_fn(model)
    optimizer.update(model, grads)
    metrics.update(loss=loss, logits=model_output, labels=target)

    state = nnx.state((model, optimizer, metrics))
    return loss, state, grads

  return jax_train_step, graphdef, state


def get_metrics():
  loss = nnx.metrics.Average('loss')
  acc = nnx.metrics.Accuracy()
  return nnx.MultiMetric(loss=loss, accuracy=acc)


def get_eval_step(cfg: Config):
  rm = cfg.opt.hsv_regmag

  @jax.jit
  def jax_eval_step(graphdef, state, input, target):
    model, _, metrics = nnx.merge(graphdef, state)
    model.eval()

    def loss_fn(model):
      y_pred = model(input)
      reg = jnp.sum(model.hankel_singvals())
      l = optax.softmax_cross_entropy_with_integer_labels(y_pred, target)
      return rm * reg + jnp.mean(l), y_pred

    loss, model_output = loss_fn(model)
    metrics.update(loss=loss, logits=model_output, labels=target)

    return state

  return jax_eval_step


def if_none(a, b):
  return b if a is None else a
