import keras
import keras.ops as ops
import time
import math
import copy

from quant import quantize

# The _set_diag helper function has been removed.

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

        H = self.H
        if actorder:
            perm = ops.argsort(ops.diagonal(H), direction='DESCENDING')
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        # --- START: THE DEFINITIVE FIX ---
        # Direct diagonal update, exactly like the original TensorFlow code.
        diag_H = ops.diagonal(H)
        dead = ops.equal(diag_H, 0.0)
        diag_H = ops.where(dead, 1.0, diag_H)
        H = H + ops.diag(ops.where(dead, 1.0, ops.zeros_like(diag_H))) # Add 1 to diagonal where it was 0
        
        damp = percdamp * ops.mean(diag_H)
        diag_H = diag_H + damp
        
        # Reconstruct H with the new diagonal
        H = (H - ops.diag(ops.diagonal(H))) + ops.diag(diag_H)
        # --- END: THE DEFINITIVE FIX ---

        try:
            Hinv = ops.linalg.inv(H)
        except Exception:
            Hinv = ops.linalg.pinv(H)

        Q = ops.zeros_like(W)

        # --- START: THE DEFINITIVE FIX ---
        # This nested loop structure is now a direct port of the original gptqkeras_fixed.py
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Err1 = ops.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # Inner loop: quantize columns and update weights *within the block*
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # Find quantization parameters for the current column or group
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                else:
                     self.quantizer.find_params(ops.expand_dims(w, 1), weight=True)

                q = quantize(
                    ops.expand_dims(w, 1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq
                )[:, 0]

                Q1 = ops.concatenate([Q1[:, :i], ops.expand_dims(q, 1), Q1[:, i+1:]], axis=1)
                err = (w - q) / d
                Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err, 1), Err1[:, i+1:]], axis=1)

                # Apply error to subsequent columns *within the block*
                if i < count - 1:
                    W1_remaining = W1[:, i+1:]
                    update = ops.matmul(ops.expand_dims(err, 1), ops.expand_dims(Hinv1[i, i+1:], 0))
                    W1_updated_remaining = W1_remaining - update
                    W1 = ops.concatenate([W1[:, :i+1], W1_updated_remaining], axis=1)

            # Update the full Q matrix with the quantized block
            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)

            # Outer loop step: update the rest of the *entire weight matrix* with the block's error
            if i2 < self.columns:
                W_remaining_total = W[:, i2:]
                update_total = ops.matmul(Err1, Hinv[i1:i2, i2:])
                W_updated_total = W_remaining_total - update_total
                W = ops.concatenate([W[:, :i2], W_updated_total], axis=1)
        # --- END: THE DEFINITIVE FIX ---

        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        self.layer.weights[0].assign(ops.transpose(Q))

    def free(self):
        self.H = None