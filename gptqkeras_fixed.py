import numpy as np
import tensorflow as tf
import keras
import time
import math

# This is the original TF implementation, now with the fix for groupsize=-1

DEBUG = False # Set to True for verbose logging if needed

def quantize(x, scale, zero, maxq):
    """The core quantization function."""
    if maxq < 0: # Trits
        return (tf.cast(x > scale / 2, tf.float32) * scale) + (tf.cast(x < zero / 2, tf.float32) * zero)

    scale_safe = tf.where(tf.equal(scale, 0), 1e-8, scale)
    q = tf.clip_by_value(tf.round(x / scale_safe) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        W = layer.weights[0]
        self.rows = W.shape[1]
        self.columns = W.shape[0]
        self.H = tf.zeros((self.columns, self.columns), dtype=tf.float32)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        """Accumulates the Hessian matrix from a batch of activations."""
        if DEBUG and self.nsamples == 0:
            print(f"  [TF-DEBUG] Inside add_batch for layer {self.layer.name}. Input tensor sum: {tf.reduce_sum(inp):.4f}")
        
        inp_transposed = tf.transpose(inp)
        num_samples_in_batch = tf.shape(inp_transposed)[1]

        if self.nsamples == 0:
            self.nsamples = num_samples_in_batch
        else:
            current_nsamples = tf.cast(self.nsamples, tf.float32)
            new_nsamples = current_nsamples + tf.cast(num_samples_in_batch, tf.float32)
            self.H *= current_nsamples / new_nsamples
            self.nsamples = new_nsamples

        inp_float = tf.cast(inp_transposed, dtype=tf.float32)
        inp_float *= math.sqrt(2.0 / float(self.nsamples))

        self.H += tf.matmul(inp_float, tf.transpose(inp_float))
        
        if DEBUG and self.nsamples == num_samples_in_batch:
            print(f"  [TF-DEBUG] H sum after batch update: {tf.reduce_sum(self.H):.4f}")

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        W_tensor = tf.transpose(tf.cast(self.layer.weights[0], tf.float32))

        # --- FIX: Added this block to handle the groupsize=-1 case ---
        # This aligns with the reference PyTorch implementation.
        if not self.quantizer.ready():
            self.quantizer.find_params(W_tensor, weight=True)
        # --- END FIX ---

        H = self.H
        W_var = tf.Variable(W_tensor)

        if actorder:
            perm = tf.argsort(tf.linalg.diag_part(H), direction='DESCENDING')
            W_var.assign(tf.gather(W_var, perm, axis=1))
            H = tf.gather(tf.gather(H, perm, axis=0), perm, axis=1)
            invperm = tf.argsort(perm)
            
        dead = tf.equal(tf.linalg.diag_part(H), 0.0)
        H = tf.linalg.set_diag(H, tf.where(dead, 1.0, tf.linalg.diag_part(H)))
        W_update_mask = tf.expand_dims(tf.cast(dead, W_var.dtype), 0)
        W_var.assign(W_var * (1.0 - W_update_mask))

        Q = tf.Variable(tf.zeros_like(W_var))
        damp = percdamp * tf.reduce_mean(tf.linalg.diag_part(H))
        H = tf.linalg.set_diag(H, tf.linalg.diag_part(H) + damp)

        try:
            H_chol = tf.linalg.cholesky(H)
            H_inv = tf.linalg.cholesky_solve(H_chol, tf.eye(self.columns, dtype=tf.float32))
            H_inv = tf.transpose(tf.linalg.cholesky(H_inv)) 
        except tf.errors.InvalidArgumentError:
            print("[TF-DEBUG] Cholesky decomposition failed. Using pseudo-inverse.")
            H_inv = tf.linalg.pinv(H)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            Err1 = tf.Variable(tf.zeros_like(W_var[:, i1:i2]))

            for i in range(i2 - i1):
                col_idx = i1 + i
                w = W_var[:, col_idx]

                if groupsize != -1 and col_idx % groupsize == 0:
                    if DEBUG: print(f"  [TF-DEBUG] Col {col_idx}: Calling find_params for group.")
                    self.quantizer.find_params(W_var[:, col_idx:(col_idx + groupsize)], weight=True)
                    if DEBUG: print(f"  [TF-DEBUG] W slice sum for find_params: {tf.reduce_sum(W_var[:, col_idx:(col_idx + groupsize)]):.6f}")
                    if DEBUG: print(f"  [TF-DEBUG] Scale[0]: {self.quantizer.scale.numpy().flatten()[0]:.6f}, Zero[0]: {self.quantizer.zero.numpy().flatten()[0]:.6f}")

                q = quantize(
                    tf.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                )[:, 0]

                Q[:, col_idx].assign(q)
                err = (w - q) / H_inv[col_idx, col_idx]
                
                if DEBUG: print(f"  [TF-DEBUG] Col {col_idx}: w_col_sum={tf.reduce_sum(w):.4f}, q_col_sum={tf.reduce_sum(q):.4f}, err_sum={tf.reduce_sum(err):.4f}")

                Err1[:, i].assign(err)
                W_var[:, col_idx + 1:i2].assign( W_var[:, col_idx + 1:i2] - tf.matmul(tf.expand_dims(err, 1), tf.expand_dims(H_inv[col_idx, col_idx + 1:i2], 0)) )

            # After block is processed, update all columns AFTER the current block
            if i2 < self.columns:
                W_var[:, i2:].assign(W_var[:, i2:] - tf.matmul(Err1, H_inv[i1:i2, i2:]))

        if actorder:
            Q.assign(tf.gather(Q, invperm, axis=1))

        return Q.value()

    def free(self):
        """Releases memory after quantization."""
        self.H = None
