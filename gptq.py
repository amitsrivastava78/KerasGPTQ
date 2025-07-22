import keras
import keras.ops as ops
import time
import math
import copy

from quant import quantize

def _set_diag(matrix, diagonal):
    diagonal = ops.cast(diagonal, matrix.dtype)
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
        inp = ops.cast(math.sqrt(2 / self.nsamples), dtype='float32') * inp
        self.H = self.H + ops.matmul(ops.transpose(inp), inp)

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = ops.transpose(ops.cast(self.layer.weights[0], 'float32'))
        
        # If not using groups, calculate params once for the whole weight matrix.
        if groupsize == -1:
            self.quantizer.find_params(W, weight=True)

        H = self.H
        if actorder:
            perm = ops.argsort(ops.diagonal(H), direction='DESCENDING')
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)
            
        dead = ops.equal(ops.diagonal(H), 0.0)
        H = _set_diag(H, ops.where(dead, 1.0, ops.diagonal(H)))
        damp = percdamp * ops.mean(ops.diagonal(H))
        H = _set_diag(H, ops.diagonal(H) + damp)

        try:
            Hinv = ops.linalg.inv(H)
        except Exception:
            Hinv = ops.linalg.pinv(H)

        Q = ops.zeros_like(W)

        # Main quantization loop
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            
            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Err1 = ops.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                
                # If using groups, find params for the current group.
                if groupsize != -1 and (i1 + i) % groupsize == 0:
                    self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                # CRITICAL FIX: Pass the entire scale/zero tensors.
                # They are now correctly shaped for broadcasting and no slicing is needed.
                q = quantize(
                    ops.expand_dims(w, 1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq
                )[:, 0]

                Q1 = ops.concatenate([Q1[:, :i], ops.expand_dims(q, 1), Q1[:, i+1:]], axis=1)
                err = (w - q) / d
                Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err, 1), Err1[:, i+1:]], axis=1)

                W1_remaining = W1[:, i+1:]
                update = ops.matmul(ops.expand_dims(err, 1), ops.expand_dims(Hinv1[i, i+1:], 0))
                W1_updated_remaining = W1_remaining - update
                W1 = ops.concatenate([W1[:, :i+1], W1_updated_remaining], axis=1)

            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)
            
            W_remaining_total = W[:, i2:]
            update_total = ops.matmul(Err1, Hinv[i1:i2, i2:])
            W_updated_total = W_remaining_total - update_total
            W = ops.concatenate([W[:, :i2], W_updated_total], axis=1)

        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        self.layer.weights[0].assign(ops.transpose(Q))

    def free(self):
        """Releases memory after quantization."""
        self.H = None