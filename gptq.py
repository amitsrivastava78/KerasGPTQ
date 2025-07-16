import keras
import keras.ops as ops
import time
import math

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
        W = layer.weights[0]
        self.rows = W.shape[1]
        self.columns = W.shape[0]
        self.H = ops.zeros((self.columns, self.columns), dtype='float32')
        self.nsamples = 0
        self.quantizer = None

    def add_batch(self, inp, out):
        if DEBUG: print(f"  > Inside add_batch for layer {self.layer.name}. Input tensor sum: {ops.sum(inp):.4f}")
        inp_transposed = ops.transpose(inp)
        num_samples_in_batch = ops.shape(inp_transposed)[1]
        if self.nsamples == 0:
            self.nsamples = num_samples_in_batch
        else:
            current_nsamples = float(self.nsamples)
            new_nsamples = current_nsamples + float(num_samples_in_batch)
            self.H *= current_nsamples / new_nsamples
            self.nsamples = new_nsamples
        inp_float = ops.cast(inp_transposed, dtype='float32')
        inp_float *= math.sqrt(2.0 / float(self.nsamples))
        self.H += ops.matmul(inp_float, ops.transpose(inp_float))
        if DEBUG: print(f"  > H sum after batch update: {ops.sum(self.H):.4f}")

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False):
        W = ops.transpose(ops.cast(self.layer.weights[0], 'float32'))
        H = self.H

        dead = ops.equal(ops.diag(H), 0.0)
        H = _set_diag(H, ops.where(dead, 1.0, ops.diag(H)))
        
        if actorder:
            perm = ops.argsort(-ops.diag(H))
            W = ops.take(W, perm, axis=1)
            H = ops.take(ops.take(H, perm, axis=0), perm, axis=1)
            invperm = ops.argsort(perm)

        Q = ops.zeros_like(W)
        damp = percdamp * ops.mean(ops.diag(H))
        H = _set_diag(H, ops.diag(H) + damp)

        try:
            H_chol = ops.linalg.cholesky(H)
            H_inv_chol = ops.linalg.solve_triangular(H_chol, ops.eye(self.columns, dtype='float32'), lower=True)
            H_inv = ops.transpose(ops.linalg.solve_triangular(ops.transpose(H_chol), H_inv_chol, lower=False))
        except Exception:
            print("Cholesky decomposition failed. Using pseudo-inverse.")
            H_inv = ops.linalg.pinv(H)

        # Main quantization loop
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            
            # Get the current block of weights we will be processing
            W_block = W[:, i1:i2]
            
            # Initialize storage for errors and quantized weights for this block
            Err_block = ops.zeros_like(W_block)
            Q_block = ops.zeros_like(W_block)

            # Process each column within the current block
            for i in range(i2 - i1):
                col_idx_global = i1 + i  # The global index of the column
                w_col = W_block[:, i]    # The current column from our *local, updating* block

                # Find quantization parameters if it's the start of a new group
                # CRITICAL: Pass the slice from the *main, updated W matrix*
                if groupsize != -1 and col_idx_global % groupsize == 0:
                    self.quantizer.find_params(W[:, col_idx_global:(col_idx_global + groupsize)], weight=True)
                
                # Quantize the current column
                q_col = quantize(ops.expand_dims(w_col, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq)[:, 0]
                
                # Store the quantized column in our local Q_block
                Q_block = ops.concatenate([Q_block[:, :i], ops.expand_dims(q_col, 1), Q_block[:, i + 1:]], axis=1)

                # Calculate the quantization error for the current column
                err = (w_col - q_col) / H_inv[col_idx_global, col_idx_global]
                
                # Store the error
                Err_block = ops.concatenate([Err_block[:, :i], ops.expand_dims(err, 1), Err_block[:, i + 1:]], axis=1)

                # ** Replicate the in-place update for the REST OF THE BLOCK **
                H_inv_part = H_inv[col_idx_global, (col_idx_global + 1):i2]
                update_values = ops.matmul(ops.expand_dims(err, 1), ops.expand_dims(H_inv_part, 0))
                W_block = ops.concatenate([
                    W_block[:, :i + 1],
                    W_block[:, i + 1:] - update_values
                ], axis=1)

            # ** After processing the block, update the main Q and W matrices **
            # Update master Q with the now-complete quantized block
            Q = ops.concatenate([Q[:, :i1], Q_block, Q[:, i2:]], axis=1)
            
            # Propagate the accumulated error from this block to the rest of the main W matrix
            if i2 < self.columns:
                remaining_update = ops.matmul(Err_block, H_inv[i1:i2, i2:])
                W = ops.concatenate([
                    W_block, # The updated block from the inner loop
                    W[:, i2:] - remaining_update 
                ], axis=1)
                # Re-prefix W with the already processed part
                W = ops.concatenate([
                    Q[:,:i1], W
                ], axis=1)


        if actorder:
            Q = ops.take(Q, invperm, axis=1)

        return Q

    def free(self):
        """Releases memory after quantization."""
        self.H = None