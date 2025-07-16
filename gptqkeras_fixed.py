import numpy as np
import tensorflow as tf
import keras
import time
import math

DEBUG = True

def quantize(x, scale, zero, maxq):
    """The core quantization function."""
    if maxq < 0: # Trits
        return (x > scale / 2) * scale + (x < zero / 2) * zero

    # Add a small epsilon for numerical stability
    scale_safe = tf.where(tf.equal(scale, 0), 1e-8, scale)
    q = tf.clip_by_value(tf.round(x / scale_safe) + zero, 0, maxq)
    return scale * (q - zero)

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        W = layer.weights[0]
        self.rows = W.shape[1]      # output_features
        self.columns = W.shape[0]   # input_features
        self.H = tf.zeros((self.columns, self.columns), dtype=tf.float32)
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        """Accumulates the Hessian matrix from a batch of activations."""
        if DEBUG:
            print(f"  > Inside add_batch for layer {self.layer.name}. Input tensor sum: {tf.reduce_sum(inp).numpy():.4f}")

        # This function receives a 2D activation tensor: (num_tokens, features)
        # We need to transpose it to (features, num_tokens) for the Hessian calculation
        inp_transposed = tf.transpose(inp)

        num_samples_in_batch = tf.shape(inp_transposed)[1]

        # Running average for the Hessian
        if self.nsamples == 0:
            self.nsamples = num_samples_in_batch
        else:
            current_nsamples = tf.cast(self.nsamples, tf.float32)
            new_nsamples = current_nsamples + tf.cast(num_samples_in_batch, tf.float32)
            self.H *= current_nsamples / new_nsamples
            self.nsamples = new_nsamples

        inp_float = tf.cast(inp_transposed, dtype=tf.float32)
        inp_float *= math.sqrt(2.0 / float(self.nsamples))

        # The actual update: (features, tokens) @ (tokens, features) -> (features, features)
        self.H += tf.matmul(inp_float, tf.transpose(inp_float))

        if DEBUG:
            print(f"  > H sum after batch update: {tf.reduce_sum(self.H).numpy():.4f}")


    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        """
        The main quantization method, now a true 1:1 port of the high-accuracy PyTorch version.
        """
        W = tf.transpose(tf.cast(self.layer.weights[0], tf.float32))
        H = self.H

        if DEBUG:
            print(f"--- TF DEBUG ---")
            print(f"Initial W sum: {tf.reduce_sum(W).numpy():.6f}")
            print(f"Initial H sum: {tf.reduce_sum(H).numpy():.6f}")

        dead = tf.equal(tf.linalg.diag_part(H), 0.0)
        H = tf.linalg.set_diag(H, tf.where(dead, 1.0, tf.linalg.diag_part(H)))
        W_update_mask = tf.expand_dims(tf.cast(dead, W.dtype), 0)
        W = W * (1.0 - W_update_mask)

        if actorder:
            perm = tf.argsort(tf.linalg.diag_part(H), direction='DESCENDING')
            W = tf.gather(W, perm, axis=1)
            H = tf.gather(tf.gather(H, perm, axis=0), perm, axis=1)
            invperm = tf.argsort(perm)

        Losses = tf.Variable(tf.zeros_like(W))
        Q = tf.Variable(tf.zeros_like(W))
        # Use tf.Variable for W to allow for efficient in-place updates.
        W_var = tf.Variable(W) 

        damp = percdamp * tf.reduce_mean(tf.linalg.diag_part(H))
        H = tf.linalg.set_diag(H, tf.linalg.diag_part(H) + damp)

        # FIX 2: Cholesky Reformulation for Numerical Stability
        try:
            H_chol = tf.linalg.cholesky(H)
            H_inv = tf.linalg.cholesky_solve(H_chol, tf.eye(self.columns, dtype=tf.float32))
            # The final transpose to get the upper-triangular factor is crucial
            H_inv = tf.transpose(tf.linalg.cholesky(H_inv)) 
        except tf.errors.InvalidArgumentError:
            print("Cholesky decomposition failed. Using pseudo-inverse.")
            H_inv = tf.linalg.pinv(H)

        # Main Quantization Loop
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)

            W1 = W_var[:, i1:i2]
            H_inv1 = H_inv[i1:i2, i1:i2]
            Err1 = tf.Variable(tf.zeros_like(W1))

            for i in range(i2 - i1):
                w = W1[:, i]
                d = H_inv1[i, i]

                # FIX 1: Dynamic Group-Wise Parameter Finding
                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W_var[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quantize(
                    tf.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                )[:, 0]

                Q[:, i1 + i].assign(q)
                Losses[:, i1 + i].assign((w - q) ** 2 / d ** 2)

                err = (w - q) / d

                # Perform the weight update on the tf.Variable
                current_slice = W_var[:, i1+i:i2]
                update_values = tf.matmul(tf.expand_dims(err, axis=1), tf.expand_dims(H_inv1[i, i:], axis=0))
                W_var[:, i1+i:i2].assign(current_slice - update_values)
                Err1[:, i].assign(err)

            # Update the rest of the weights after the block is done
            current_remaining_slice = W_var[:, i2:]
            update_values = tf.matmul(Err1.value(), H_inv[i1:i2, i2:])
            W_var[:, i2:].assign(current_remaining_slice - update_values)

        Losses.assign(Losses / 2)

        if DEBUG:
            print('error', tf.reduce_sum(Losses).numpy())

        if actorder:
            Q.assign(tf.gather(Q.value(), invperm, axis=1))

        if DEBUG:
            print(f"Final Q sum: {tf.reduce_sum(Q).numpy():.6f}")
            print(f"--- END TF DEBUG ---")

        return Q.value()

    def free(self):
        """Releases memory after quantization."""
        self.H = None
