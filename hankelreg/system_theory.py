import jax
import jax.numpy as jnp
from einops import rearrange
from jax import jit
from jax.numpy.linalg import cholesky, svd


@jit
def solve_discrete_lyapunov(A, Q, safety_factor=1e-7):
  fac = 1 - safety_factor
  M = jnp.kron(fac * A, fac * A) - jnp.eye(A.shape[0]**2)
  P = jnp.linalg.solve(M, -jnp.reshape(Q, -1))
  return jnp.reshape(P, (A.shape[0], A.shape[0]))


@jit
def solve_discrete_sylvester(A, B, Q, safety_factor=1e-7):
  fac = 1 - safety_factor
  M = jnp.kron(fac * B.T, fac * A) - jnp.eye(4)
  P = jnp.linalg.solve(M, -jnp.reshape(Q.T, -1))
  return jnp.reshape(P, (2, 2)).T


def control_lyap(A, B):

  def solve_for_row(ai, bi):
    return jax.vmap(
        lambda aj, bj: solve_discrete_sylvester(ai, aj.T, bi @ bj.T))(A, B)

  return jax.vmap(solve_for_row)(A, B)


def obs_lyap(A, C):

  def solve_for_row(ai, bi):
    return jax.vmap(
        lambda aj, bj: solve_discrete_sylvester(ai.T, aj, bi.T @ bj),
        in_axes=[0, 1])(A, C)

  return jax.vmap(solve_for_row, in_axes=[0, 1])(A, C)


def hankel_sv(A, B, C, cutoff=1e-10):
  P = control_lyap(A, B)
  Q = obs_lyap(A, C)
  Pd = rearrange(P, 'i j a b -> (i a) (j b)')
  Qd = rearrange(Q, 'i j a b -> (i a) (j b)')
  hsv_sq = jnp.real(jnp.linalg.eigvals(Pd @ Qd))
  return jnp.sqrt(jnp.maximum(cutoff, hsv_sq))


def balanced_realization(A, B, C, rank=10, eps=1e-7):
  P, Q = control_lyap(A, B), obs_lyap(A, C)
  Pd, Qd = dense_sym(P), dense_sym(Q)
  reg = eps * jnp.eye(Pd.shape[0])
  S, R = cholesky(Pd + reg), cholesky(Qd + reg)
  U, Sigma, VT = svd(S.T @ R, full_matrices=False)
  T = jnp.diag(jnp.sqrt(1 / Sigma)[:rank]) @ VT[:rank, :] @ R.T
  Tinv = S @ U[:, :rank] @ jnp.diag(jnp.sqrt(1 / Sigma)[:rank])
  Ad = blkdiag_to_dense(A)
  Bd = rearrange(B, 'i j m -> (i j) m')
  Cd = rearrange(C, 'm i j -> m (i j)')
  return T @ Ad @ Tinv, T @ Bd, Cd @ Tinv


def dense_sym(A):
  Ad = rearrange(A, 'i j a b -> (i a) (j b)')
  return 0.5 * (Ad + Ad.T)


def blkdiag_to_dense(block_bd: jnp.ndarray) -> jnp.ndarray:
  n_blocks, block_dim1, block_dim2 = block_bd.shape
  n = n_blocks * block_dim1
  m = n_blocks * block_dim2
  out = jnp.zeros((n, m), dtype=block_bd.dtype)

  def body(i, val):
    nstart = i * block_dim1
    mstart = i * block_dim2
    return jax.lax.dynamic_update_slice(val, block_bd[i], (nstart, mstart))

  out = jax.lax.fori_loop(0, n_blocks, body, out)
  return out


def diagonalize(A, B, C):
  V, T = jnp.linalg.eig(A)
  B = jnp.linalg.inv(T) @ B
  C = C @ T
  return V, B, C
