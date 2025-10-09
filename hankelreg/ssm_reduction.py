from copy import deepcopy

import jax
import jax.numpy as jnp

from hankelreg.model import DiagonalConjugate
from hankelreg.system_theory import balanced_realization


def diagonalize(A, B, C):
  V, T = jnp.linalg.eig(A)
  B = jnp.linalg.inv(T) @ B
  C = C @ T
  return V, B, C


def retained_energy(x):
  return jnp.cumsum(jnp.sort(x, descending=True)) / jnp.sum(x)


def reduction_index(hsvs, min_energy=0.98):
  if min_energy >= 1.0 - 1e-6:
    return len(hsvs)
  else:
    return jnp.argwhere(retained_energy(hsvs) > min_energy)[0][0]


def mean_rank_to_energy(rank_percentage, hsvs):
  f1 = lambda x, y: reduction_index(x, y)

  def val(energy):
    ranks = 0
    for hsv in hsvs:
      ranks += f1(hsv, energy)
    mean_ranks = ranks / len(hsvs)
    return mean_ranks / len(hsvs[0]) - rank_percentage

  min_energy = 0.0
  max_energy = 1.0
  n_iters = 0
  energy = (min_energy + max_energy) / 2
  while abs(val(energy)) > 1e-8 and n_iters < 100:
    if val(energy) > 0:
      max_energy = energy
    else:
      min_energy = energy
    energy = (min_energy + max_energy) / 2
    n_iters += 1

  return energy


def reduce_by_rank(model, max_ranks, hsvs):
  jax.config.update('jax_enable_x64', True)
  mm = deepcopy(model)
  energies = []
  ranks = []
  for i in range(len(mm.layers)):
    A, B, C, D = model.layers[i].sequence_processor.get_ABCD()
    # A = jnp.asarray(A, dtype=jnp.complex64)
    # B = jnp.asarray(B, dtype=jnp.complex64)
    # C = jnp.asarray(C, dtype=jnp.complex64)
    r = max_ranks[i]
    if r < 1.0:
      r = int(jnp.floor(A.shape[0] * r))
    ranks.append(r)
    energies.append(retained_energy(hsvs[i])[r])
    Ab, Bb, Cb = diagonalize(*balanced_realization(A, B, C, rank=int(r)))
    Ab = jnp.asarray(Ab, dtype=jnp.complex64)
    Bb = jnp.asarray(Bb, dtype=jnp.complex64)
    Cb = jnp.asarray(Cb, dtype=jnp.complex64)
    new_sp = DiagonalConjugate(Ab, Bb, Cb, D)
    mm.layers[i].sequence_processor = new_sp
  jax.config.update('jax_enable_x64', False)
  return mm, jnp.stack(energies), jnp.stack(ranks)


def reduce_ssm(model, hsvs, max_rank):
  retained = mean_rank_to_energy(max_rank, hsvs)
  ranks = []
  for i in range(len(model.layers)):
    ranks.append(reduction_index(hsvs[i], retained))
  ranks = jnp.stack(ranks)
  return reduce_by_rank(model, ranks, hsvs)[0], ranks
