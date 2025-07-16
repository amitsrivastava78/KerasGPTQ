import keras
import keras.ops as ops
import time
import math

# This is the Keras 3.0 implementation, fully ported from the reference PyTorch version
# to correctly handle all cases, including groupsize=-1 and groupsize=128.

DEBUG = False 

def _set_diag(matrix, diagonal):
    """Backend-agnostic implementation to set the diagonal of a matrix."""
    off_diagonal_mask = ops.ones_like(matrix) - ops.eye(matrix.shape[0], dtype=matrix.dtype)
    off_diagonals = matrix * off_diagonal_mask
    new_diagonals = ops.diag(diagonal)
    return off_diagonals + new_diagonals

def quantize(x, scale, zero, maxq):
    """The core quantization function."""
    if maxq < 0:
        return (ops.cast(x > scale / 2, 'float32') * scale) + (ops.cast(x < zero / 2, 'float32') * zero)
    scale_safe = ops.where(ops.equal(scale, 0), 1e-8, scale)
    q = ops.clip(ops.round(x / scale_safe) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        # For Keras layers, weight is usually the first element in .weights
        self.dev = keras.backend.backend() # To know which backend is running
        W = layer.weights[0] 
        self.rows = W.shape[1]
        self.columns = W.shape[0]
        self.H = ops.zeros((self.columns, self.columns), dtype='float32')
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        if len(ops.shape(inp)) == 2:
            inp = ops.expand_dims(inp, 0)
        tmp = ops.shape(inp)[0]

        # Correctly handle different input shapes for dense layers
        if len(ops.shape(inp)) == 3:
            inp = ops.reshape(inp, (-1, ops.shape(inp)[-1]))
        inp = ops.transpose(inp)

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        
        inp = math.sqrt(2 / self.nsamples) * ops.cast(inp, 'float32')
        self.H += ops.matmul(inp, ops.transpose(inp))

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        W = ops.transpose(ops.cast(self.layer.weights[0], 'float32'))

        # --- FIX based on PyTorch version: Handle groupsize=-1 case ---
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        # --- END FIX ---

        H = self.H
        
        dead = ops.equal(ops.diag(H), 0.0)
        H = _set_diag(H, ops.where(dead, 1.0, ops.diag(H)))
        W_update_mask = ops.expand_dims(ops.cast(dead, W.dtype), 0)
        W = W * (1.0 - W_update_mask)

        if actorder:
            perm = ops.argsort(-ops.diag(H))
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        Q = ops.zeros_like(W)

        damp = percdamp * ops.mean(ops.diag(H))
        H = _set_diag(H, ops.diag(H) + damp)

        # Replicating the PyTorch three-step H_inv calculation
        try:
            L = ops.linalg.cholesky(H)
            # torch.cholesky_inverse(L) is equivalent to inv(L L^T) = inv(H)
            H_inv = ops.linalg.inv(H) 
            # Get upper-triangular Cholesky factor of the inverse
            H_inv_chol_L = ops.linalg.cholesky(H_inv)
            Hinv = ops.transpose(H_inv_chol_L)
        except Exception:
             # Fallback for numerical stability
            print("Cholesky decomposition failed. Using pseudo-inverse.")
            H_inv = ops.linalg.pinv(H)
            H_inv_chol_L = ops.linalg.cholesky(H_inv)
            Hinv = ops.transpose(H_inv_chol_L)


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

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                
                q = quantize(
                    ops.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()

                Q1 = ops.concatenate([Q1[:, :i], ops.expand_dims(q, 1), Q1[:, i + 1:]], axis=1)
                
                err1 = (w - q) / d
                Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err1, 1), Err1[:, i + 1:]], axis=1)

                # Update subsequent columns in the local block W1
                W1_remaining = W1[:, i+1:]
                Hinv_slice = ops.expand_dims(Hinv1[i, i+1:], 0)
                update = ops.matmul(ops.expand_dims(err1, 1), Hinv_slice)
                W1 = ops.concatenate([W1[:, :i+1], W1_remaining - update], axis=1)

            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)
            
            # Update columns after the block in the main W matrix
            W_remaining = W[:, i2:]
            Hinv_slice_main = Hinv[i1:i2, i2:]
            W_remaining -= ops.matmul(Err1, Hinv_slice_main)
            W = ops.concatenate([W[:, :i2], W_remaining], axis=1)

        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        # Transpose back before assigning to the layer
        final_Q = ops.transpose(Q)
        self.layer.weights[0].assign(ops.cast(final_Q, self.layer.weights[0].dtype))
        
        # We return the transposed version for testing equivalence with the old script
        return Q

    def free(self):
        self.H = None
        # In a real scenario, you might want to clear GPU cache if using PyTorch backend
        # if keras.backend.backend() == 'torch':
        #     import torch
        #     torch.cuda.empty_cache()

