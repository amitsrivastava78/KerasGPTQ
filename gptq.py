import keras
import keras.ops as ops
import time
import math

# Use an explicit alias to prevent any import confusion
from quant import quantize as k3_quantize

# This is the Keras 3.0 implementation, fully ported from the reference PyTorch version
# to correctly handle all cases, including groupsize=-1 and groupsize=128.

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
        self.dev = keras.backend.backend()
        W = layer.weights[0]
        self.rows = W.shape[1]
        self.columns = W.shape[0]
        self.H = ops.zeros((self.columns, self.columns), dtype='float32')
        self.nsamples = 0
        self.quantizer = None

    # def add_batch(self, inp, out):
    #     """Accumulates the Hessian matrix from a batch of activations."""
    #     if DEBUG and self.nsamples == 0:
    #         print(f"  [TF-DEBUG] Inside add_batch for layer {self.layer.name}. Input tensor sum: {tf.reduce_sum(inp):.4f}")
        
    #     inp_transposed = ops.transpose(inp)
    #     num_samples_in_batch = ops.shape(inp_transposed)[1]


    #     if self.nsamples == 0:
    #         self.nsamples = num_samples_in_batch
    #     else:
    #         current_nsamples = ops.cast(self.nsamples, dtype='float32')
    #         new_nsamples = current_nsamples + ops.cast(num_samples_in_batch, dtype='float32')
    #         self.H *= current_nsamples / new_nsamples
    #         self.nsamples = new_nsamples

    #     inp_float = ops.cast(inp_transposed, dtype='float32')
    #     inp_float *= math.sqrt(2.0 / float(self.nsamples))

    #     self.H += ops.matmul(inp_float, ops.transpose(inp_float))
        
    #     if DEBUG and self.nsamples == num_samples_in_batch:
    #         print(f"  [TF-DEBUG] H sum after batch update: {tf.reduce_sum(self.H):.4f}")

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = ops.expand_dims(inp, axis=0)
        tmp = inp.shape[0]
        if isinstance(self.layer, keras.layers.Dense) or isinstance(self.layer, keras.layers.Conv1D):
            if len(ops.shape(inp)) == 3:
                inp = ops.reshape(inp, (-1, ops.shape(inp)[-1]))
            inp = ops.transpose(inp)
        self.H = self.H * (self.nsamples / (self.nsamples + tmp))
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * ops.cast(inp, 'float32')
        self.H = self.H + ops.matmul(inp, ops.transpose(inp))



    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = ops.transpose(ops.cast(self.layer.weights[0], 'float32'))

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        print(f"--- Keras DEBUG ---")
        print(f"Initial W sum: {ops.sum(W).numpy():.6f}")
        print(f"Initial H sum: {ops.sum(H).numpy():.6f}")

        dead = ops.equal(ops.diag(H), 0.0)
        H = _set_diag(H, ops.where(dead, 1.0, ops.diag(H)))
        W_update_mask = ops.expand_dims(ops.cast(dead, W.dtype), 0)
        W = W * (1.0 - W_update_mask)

        if static_groups:
            groups = []
            for i in range(0, self.columns, groupsize):
                # Deep copy the current quantizer instance
                k3_quantizer_copy = copy.deepcopy(self.quantizer)
                
                # Find parameters for the specific weight group
                # 'W' should be in its final state here (e.g., after actorder permutation if applicable)
                k3_quantizer_copy.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(k3_quantizer_copy)

        if actorder:
            perm = ops.argsort(-ops.diag(H))
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        Q = ops.zeros_like(W)

        damp = percdamp * ops.mean(ops.diag(H))
        H = _set_diag(H, ops.diag(H) + damp)

        # --- FINAL FIX: Use a numerically stable method to compute Hinv ---
        # This mirrors the logic of torch.cholesky_inverse(L)
        try:
            L = ops.linalg.cholesky(H)
            identity = ops.eye(self.columns, dtype=H.dtype)
            L_inv = ops.linalg.solve_triangular(L, identity, lower=True)
            H_inv = ops.matmul(ops.transpose(L_inv), L_inv)
            # Get the upper-triangular factor of the inverse for the main loop
            Hinv = ops.transpose(ops.linalg.cholesky(H_inv))
        except Exception as e:
            # This fallback should now be extremely rare
            print(f"Cholesky decomposition failed with {e}. The Hessian may be ill-conditioned.")
            # As a last resort, use a direct but less stable inverse.
            Hinv = ops.transpose(ops.linalg.cholesky(ops.linalg.inv(H)))


        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2]
            Q1 = ops.zeros_like(W1)
            Losses1 = ops.zeros_like(W)
            Err1 = ops.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                

                q = k3_quantize(
                    ops.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                )
                q = ops.squeeze(q, axis=1)

                Q1 = ops.concatenate([Q1[:, :i], ops.expand_dims(q, 1), Q1[:, i + 1:]], axis=1)

                # Calculation of the squared error term
                squared_error = ops.square(w - q)
                denom_squared = ops.square(d)

                # Handle potential division by zero for denom_squared
                denom_squared_safe = ops.where(ops.equal(denom_squared, 0), ops.ones_like(denom_squared) * 1e-8, denom_squared)

                loss_term = squared_error / denom_squared_safe

                # Assign to Losses1 using concatenate, similar to Q1
                Losses1 = ops.concatenate([Losses1[:, :i], ops.expand_dims(loss_term, 1), Losses1[:, i + 1:]], axis=1)

                err1 = (w - q) / d

                # This part extracts the portion of W1 that needs to be updated
                W1_remaining = W1[:, i+1:]

                # This calculates the update matrix, equivalent to err1.unsqueeze(1).matmul(Hinv1[i, i+1:].unsqueeze(0))
                # Note: In gptq.py, Hinv_slice is Hinv1[i, i+1:], aligning with updating subsequent columns
                Hinv_slice = ops.expand_dims(Hinv1[i, i+1:], 0) # Equivalent to Hinv1[i, i:].unsqueeze(0) if i+1 instead of i
                update = ops.matmul(ops.expand_dims(err1, 1), Hinv_slice)

                # This reconstructs W1 by concatenating the unchanged part with the updated part
                W1 = ops.concatenate([W1[:, :i+1], W1_remaining - update], axis=1)
                Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err1, 1), Err1[:, i + 1:]], axis=1)
                if 1:
                     print(f"Hinv sum: {ops.sum(Hinv).numpy():.6f}")
                     print(f"First column scale: {self.quantizer.scale[0].numpy().item():.6f}")
                     print(f"First column q sum: {ops.sum(q).numpy():.6f}")
                     print(f"First column err sum: {ops.sum(err1).numpy():.6f}")


                # err1 = (w - q) / d
                # Err1 = ops.concatenate([Err1[:, :i], ops.expand_dims(err1, 1), Err1[:, i + 1:]], axis=1)
                # if 1:
                #     print(f"Hinv sum: {ops.sum(Hinv).numpy():.6f}")
                #     print(f"First column scale: {self.quantizer.scale[0].numpy().item():.6f}")
                #     print(f"First column q sum: {ops.sum(q).numpy():.6f}")
                #     print(f"First column err sum: {ops.sum(err1).numpy():.6f}")

                # W1_remaining = W1[:, i+1:]
                # Hinv_slice = ops.expand_dims(Hinv1[i, i+1:], 0)
                # update = ops.matmul(ops.expand_dims(err1, 1), Hinv_slice)
                # W1 = ops.concatenate([W1[:, :i+1], W1_remaining - update], axis=1)

            Q = ops.concatenate([Q[:, :i1], Q1, Q[:, i2:]], axis=1)

            if i2 < self.columns:
                W_remaining = W[:, i2:]
                Hinv_slice_main = Hinv[i1:i2, i2:]
                W_remaining -= ops.matmul(Err1, Hinv_slice_main)
                W = ops.concatenate([W[:, :i1], W1, W_remaining], axis=1)
            else:
                W = ops.concatenate([W[:, :i1], W1], axis=1)

        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        return Q

    def free(self):
        self.H = None
