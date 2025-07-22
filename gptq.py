import keras
import keras.ops as ops
import time
import math
import copy

from quant import quantize as k3_quantize

DEBUG = False

def _set_diag(matrix, diagonal):
    """Backend-agnostic implementation to set the diagonal of a matrix."""
    off_diagonal_mask = ops.ones_like(matrix) - ops.eye(matrix.shape[0], dtype=matrix.dtype)
    off_diagonals = matrix * off_diagonal_mask
    new_diagonals = ops.diag(diagonal)
    return off_diagonals + new_diagonals

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        W = ops.cast(layer.weights[0], 'float32')
        self.rows = W.shape[1]
        self.columns = W.shape[0]
        self.H = ops.zeros((self.columns, self.columns), dtype='float32')
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        if len(inp.shape) == 1:
            inp = ops.expand_dims(inp, 0)
        if isinstance(self.layer, (keras.layers.Dense, keras.layers.Conv1D)):
            if len(inp.shape) == 3:
                inp = ops.reshape(inp, (-1, inp.shape[-1]))
        inp = ops.cast(inp, 'float32')
        tmp = inp.shape[0]
        self.H = self.H * (self.nsamples / (self.nsamples + tmp))
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp
        self.H = self.H + ops.matmul(ops.transpose(inp), inp)

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = ops.transpose(ops.cast(self.layer.weights[0], 'float32'))
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = ops.cast(self.H, 'float32')

        # Robust Hessian Preparation
        H = ops.where(ops.isfinite(H), H, 0.0)
        diag_h = ops.diagonal(H)
        diag_h = ops.where(ops.less_equal(diag_h, 0), 1.0, diag_h)
        H = _set_diag(H, diag_h)
        damp = percdamp * ops.mean(ops.diagonal(H))
        H = _set_diag(H, ops.diagonal(H) + damp)
        H = ops.where(ops.isfinite(H), H, ops.ones_like(H))

        if actorder:
            perm = ops.argsort(-ops.diagonal(H))
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        try:
            L = ops.linalg.cholesky(H)
            identity = ops.eye(self.columns, dtype='float32')
            L_inv = ops.linalg.solve_triangular(L, identity, lower=True)
            H_inv = ops.matmul(ops.transpose(L_inv), L_inv)
            Hinv = ops.transpose(ops.linalg.cholesky(H_inv))
        except Exception as e:
            print(f"CRITICAL WARNING: Hessian inversion failed despite sanitization: {e}. Using identity matrix as a failsafe.")
            Hinv = ops.eye(self.columns, dtype='float32')

        Q = ops.zeros_like(W, dtype='float32')

        # === START: COLUMN-BY-COLUMN UPDATE (Original Logic) ===
        for i in range(self.columns):
            w = W[:, i]
            d = Hinv[i, i]

            if groupsize != -1 and i % groupsize == 0:
                self.quantizer.find_params(W[:, i:(i + groupsize)], weight=True)

            if self.quantizer.perchannel:
                idx = i % groupsize if groupsize != -1 else i
                scale = self.quantizer.scale[idx]
                zero = self.quantizer.zero[idx]
            else:
                scale = self.quantizer.scale
                zero = self.quantizer.zero
            
            q = k3_quantize(ops.expand_dims(w, 1), scale, zero, self.quantizer.maxq)
            q = ops.squeeze(q, axis=1)

            Q = ops.concatenate([Q[:, :i], ops.expand_dims(q, 1), Q[:, i+1:]], axis=1)

            err = (w - q) / d
            
            # Update all subsequent columns immediately
            if i < self.columns - 1:
                W_remaining = W[:, i+1:]
                Hinv_slice = ops.expand_dims(Hinv[i, i+1:], 0)
                update = ops.matmul(ops.expand_dims(err, 1), Hinv_slice)
                W_updated_remaining = W_remaining - update
                W = ops.concatenate([W[:, :i+1], W_updated_remaining], axis=1)
        # === END: COLUMN-BY-COLUMN UPDATE ===

        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        self.layer.weights[0].assign(ops.cast(ops.transpose(Q), self.layer.weights[0].dtype))

    def free(self):
        self.H = None