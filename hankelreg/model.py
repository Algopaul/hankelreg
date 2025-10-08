import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import nnx
from flax.nnx.nn import initializers
from jax._src import dtypes
from jax.lax import associative_scan

from hankelreg.config import SSMConfig
from hankelreg.system_theory import hankel_sv


def interval_trafo_tanh(x, a=0.0, b=1.0):
  x = (nnx.tanh(x) + 1.0) / 2
  return (b - a) * x + a


def shifted_normal(std_dev=1.0, shift=0.0):

  def init(key, shape, dtype=dtypes.float_):
    x = std_dev * jrd.normal(key, shape, dtype)
    return x + shift

  return init


default_angle_init = shifted_normal()
default_retention_init = shifted_normal(std_dev=0.25, shift=1.0)


class Rotation(nnx.Module):

  def __init__(
      self,
      n_blocks,
      *,
      dtype=jnp.float32,
      angle_init: nnx.Initializer = default_angle_init,
      retention_init: nnx.Initializer = default_retention_init,
      rngs: nnx.Rngs,
  ):
    a_key = rngs.params()
    self.angles = nnx.Param(angle_init(a_key, (n_blocks,), dtype))
    r_key = rngs.params()
    self.retentions = nnx.Param(retention_init(r_key, (n_blocks,), dtype))

  def get_op(self, input):
    del input
    angles = interval_trafo_tanh(self.angles, 0, jnp.pi)
    retentions = interval_trafo_tanh(self.retentions, 0, 1.0)
    return angles, retentions

  def system_update(self, Ai, Aj):
    angles_i, retentions_i = Ai
    angles_j, retentions_j = Aj
    return angles_i + angles_j, retentions_i * retentions_j

  def state_update(self, Aj, xi, xj) -> jnp.ndarray:
    angles_j, retentions_j = Aj
    caj = retentions_j * jnp.cos(angles_j)
    saj = retentions_j * jnp.sin(angles_j)
    A = jnp.stack((caj, saj))
    B = jnp.stack((-saj, caj))
    AB = jnp.stack((A, B))
    return jnp.einsum('klji,jil->jik', AB, xi) + xj

  def binary_op(self, sys_state_i, sys_state_j):
    system_i, state_i = sys_state_i
    system_j, state_j = sys_state_j
    system_out = self.system_update(system_i, system_j)
    state_out = self.state_update(system_j, state_i, state_j)
    return system_out, state_out

  def __call__(self, input_sequence):
    A = jax.vmap(self.get_op)(input_sequence)
    return A

  def block_matrix(self):
    angles, retentions = self.get_op(None)

    def construct_block(angle, retention):
      ca = jnp.cos(angle)
      sa = jnp.sin(angle)
      return retention * jnp.array([[ca, sa], [-sa, ca]])

    return jax.vmap(construct_block)(angles, retentions)


class LTI(nnx.Module):

  def __init__(
      self,
      n_blocks: int,
      io_dim: int,
      *,
      angle_init: nnx.Initializer = default_angle_init,
      retention_init: nnx.Initializer = default_retention_init,
      b_init: nnx.Initializer = initializers.lecun_normal(),
      c_init: nnx.Initializer = initializers.lecun_normal(),
      d_init: nnx.Initializer = initializers.truncated_normal(),
      dtype=jnp.float32,
      rngs: nnx.Rngs,
  ):
    self.io_dim = io_dim
    self.rotation = Rotation(
        n_blocks,
        angle_init=angle_init,
        retention_init=retention_init,
        dtype=dtype,
        rngs=rngs,
    )
    self.sys_in = nnx.Param(b_init(rngs.params(), (n_blocks, 2, io_dim), dtype))
    self.sys_out = nnx.Param(
        c_init(rngs.params(), (io_dim, n_blocks, 2), dtype))
    self.sys_feed = nnx.Param(d_init(rngs.params(), (io_dim,), dtype))
    pass

  def __call__(self, u_in):
    inputs = jax.vmap(lambda u: jnp.einsum('ijm,m->ij', self.sys_in, u))(u_in)
    systems = self.rotation(u_in)
    _, states = associative_scan(self.rotation.binary_op, (systems, inputs))
    out = jax.vmap(lambda x: jnp.einsum('kij,ij->k', self.sys_out, x))(states)
    return out + jax.vmap(lambda u: self.sys_feed * u)(u_in)

  def batched_process(self, input_sequence):
    p = lambda input_sequence: self(input_sequence)
    return jax.vmap(p)(input_sequence)

  def get_ABCD(self):
    A = self.rotation.block_matrix()
    B = jnp.array(self.sys_in)
    C = jnp.array(self.sys_out)
    D = jnp.array(self.sys_feed)
    return A, B, C, D

  def hankel_singvals(self):
    A, B, C, _ = self.get_ABCD()
    return hankel_sv(A, B, C)


class DiagonalConjugate(nnx.Module):

  def __init__(self, A, B, C, D):
    self.A = A
    self.B = B
    self.C = C
    self.D = D

  def solve(self, input_sequence):
    L, B, C = (self.A, self.B, self.C)
    inputs = jax.vmap(lambda x: B @ x)(input_sequence)
    systems = jax.vmap(lambda _: L)(input_sequence)
    _, states = jax.lax.associative_scan(self.binary_op, (systems, inputs))
    out = jax.vmap(lambda x: C @ x)(states)
    if len(self.D.shape) == 2:
      Du = jax.vmap(lambda x: self.D @ x)(input_sequence)
    else:
      Du = jax.vmap(lambda x: self.D * x)(input_sequence)
    return jnp.real(out) + Du

  def system_update(self, ti, tj):
    return ti * tj

  def state_update(self, Aj, xi, xj):
    return jnp.einsum('ik,ik->ik', Aj, xi) + xj

  def binary_op(self, sys_state_i, sys_state_j):
    system_i, state_i = sys_state_i
    system_j, state_j = sys_state_j
    system_out = self.system_update(system_i, system_j)
    state_out = self.state_update(system_j, state_i, state_j)
    return system_out, state_out


class SequenceLayer(nnx.Module):

  def __init__(
      self,
      sequence_processor: LTI,
      *,
      dropout_rate: float = 0.1,
      apply_skip: bool = True,
      rngs,
  ):
    io_dim = sequence_processor.io_dim
    self.sequence_processor = sequence_processor
    self.apply_skip = apply_skip
    self.dropout = nnx.Dropout(dropout_rate, broadcast_dims=[1], rngs=rngs)
    self.out2 = nnx.Linear(io_dim, io_dim, rngs=rngs)
    self.activation = 'halfglu'
    self.prenorm = nnx.BatchNorm(io_dim, momentum=0.9, rngs=rngs)

  def hankel_singvals(self):
    return self.sequence_processor.hankel_singvals()

  def __call__(self, input_sequence):
    x = input_sequence
    if self.apply_skip:
      skip = x
    x = self.prenorm(x)
    x = self.sequence_processor.batched_process(x)
    if self.activation == 'halfglu':
      x = self.dropout(nnx.gelu(x))
      x = x * jax.nn.sigmoid(self.out2(x))
      x = self.dropout(x)
    if self.apply_skip:
      x += skip  # pyright: ignore
    return x


class SSM(nnx.Module):

  @classmethod
  def from_config(cls, cfg: SSMConfig, rngs: nnx.Rngs):
    return cls(
        cfg.channels_in,
        cfg.channels_out,
        cfg.state_dim // 2,
        cfg.io_dim,
        cfg.layers,
        angle_shift=cfg.angle_shift,
        angle_scale=cfg.angle_scale,
        retention_shift=cfg.retention_shift,
        retention_scale=cfg.retention_scale,
        dropout_rate=cfg.dropout_rate,
        apply_skip=cfg.use_skip,
        reduce_mean=cfg.reduce_mean,
        rngs=rngs,
    )

  def __init__(
      self,
      channels_in,
      channels_out,
      n_blocks,
      io_dim,
      n_layers,
      *,
      angle_shift=0.0,
      angle_scale=1.0,
      retention_shift=0.0,
      retention_scale=1.0,
      dropout_rate=0.01,
      apply_skip=True,
      reduce_mean=False,
      rngs: nnx.Rngs,
  ):
    self.enc = nnx.Linear(channels_in, io_dim, rngs=rngs)
    self.dec = nnx.Linear(io_dim, channels_out, rngs=rngs)
    angle_init = shifted_normal(std_dev=angle_scale, shift=angle_shift)
    retention_init = shifted_normal(
        std_dev=retention_scale, shift=retention_shift)
    self.reduce_mean = reduce_mean
    self.layers = nnx.List([
        SequenceLayer(
            sequence_processor=LTI(
                n_blocks,
                io_dim,
                rngs=rngs,
                angle_init=angle_init,
                retention_init=retention_init,
            ),
            dropout_rate=dropout_rate,
            apply_skip=apply_skip,
            rngs=rngs,
        ) for _ in range(n_layers)
    ])

  def hankel_singvals(self):

    hsvs = []
    for i in range(len(self.layers)):
      hsvs.append(self.layers[i].hankel_singvals())
    return jnp.stack(hsvs)

  def __call__(self, x):
    x = self.enc(x)
    for sl in self.layers:
      x = sl(x)
    out = self.dec(x)
    if self.reduce_mean:
      return jnp.mean(out, axis=-2)
    else:
      return out
