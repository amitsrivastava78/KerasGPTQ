import keras.ops as ops


def quantize(x, scale, zero, maxq):
    """Applies quantization to a tensor.

    This function simulates the effect of quantization by mapping a float
    tensor to a discrete set of values and then back to float.

    Args:
        x: The input float tensor to quantize.
        scale: The quantization scale factor(s).
        zero: The quantization zero point(s).
        maxq: The maximum integer value of the quantization range (e.g., 255).

    Returns:
        The quantized-dequantized float tensor, with the same shape as x.
    """
    if maxq < 0:
        # Handles a special binary/ternary quantization case.
        # ops.cast is used to convert boolean tensors to float for arithmetic.
        term1 = ops.cast(x > (scale / 2), dtype=x.dtype) * scale
        term2 = ops.cast(x < (zero / 2), dtype=x.dtype) * zero
        return term1 + term2

    # Standard affine quantization formula:
    # 1. Scale input, shift by zero-point, and round to nearest integer.
    q = ops.round(x / scale) + zero

    # 2. Clamp the result to the valid quantization range [0, maxq].
    # ops.clip is the Keras equivalent of torch.clamp.
    q = ops.clip(q, 0, maxq)

    # 3. De-quantize: Map the integer value back to the float domain.
    return scale * (q - zero)



class Quantizer:
    def __init__(self):
        self.scale = None
        self.zero = None
        self.maxq = None

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        """
        Configures the quantizer parameters.
        Aligned with pquant.py: No 'groupsize' parameter here; grouping is handled upstream.
        """
        self.bits = bits
        # Ensure maxq is a Keras tensor with a default float32 dtype, consistent with pquant.py's tensor creation.
        self.maxq = ops.convert_to_tensor(2 ** bits - 1, dtype='float32') 
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            # For trits, maxq is -1. Ensure it's a tensor.
            self.maxq = ops.convert_to_tensor(-1, dtype='float32') 
    
    def find_params(self, x, weight=False):
        shape = ops.shape(x)
        if self.perchannel:
            if weight:
                # Flatten all dimensions except the batch dimension
                x = ops.reshape(x, (shape[0], -1))
            else:
                # Note: In Keras, the channel dimension is typically last (e.g., N, H, W, C)
                # The logic below assumes the PyTorch channel-first convention (N, C, H, W).
                # If using Keras-native data formats, this logic might need adjustment.
                if len(shape) == 4: # e.g., (N, C, H, W)
                    x = ops.transpose(x, axes=[1, 0, 2, 3]) # -> (C, N, H, W)
                    # Flatten all dimensions after the first (the channel dim)
                    x = ops.reshape(x, (ops.shape(x)[0], -1)) # -> (C, N*H*W)
                elif len(shape) == 3: # e.g., (N, C, L)
                    x = ops.reshape(x, (-1, shape[-1])) # -> (N*C, L)
                    x = ops.transpose(x) # -> (L, N*C)
                elif len(shape) == 2: # e.g., (N, C)
                    x = ops.transpose(x) # -> (C, N)
        else:
            # Flatten the entire tensor and add a leading dimension of 1
            x = ops.reshape(x, (1, -1))
        
        tmp = ops.zeros((ops.shape(x)[0],), dtype=x.dtype)
        xmin = ops.minimum(ops.min(x, axis=1), tmp)
        xmax = ops.maximum(ops.max(x, axis=1), tmp)
        if self.sym:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            tmp = xmin < 0
            xmin = ops.where(tmp, -xmax, xmin)
        tmp = ops.logical_and(xmin == 0, xmax == 0)
        xmin = ops.where(tmp, -1.0, xmin)
        xmax = ops.where(tmp, 1.0, xmax)

        if self.maxq < 0:
            # Direct assignment for dynamic or float quantization
            self.scale = xmax
            self.zero = xmin
        else:
            # Standard affine quantization scale
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                # For symmetric quantization, the zero point is the middle of the range.
                # ops.full_like creates a tensor with the same shape as self.scale.
                fill_value = (self.maxq + 1) / 2
                self.zero = ops.full_like(self.scale, fill_value)
            else:
                # For asymmetric quantization, calculate the zero point based on the min value.
                self.zero = ops.round(-xmin / self.scale)
        
        if self.mse:
            # Initialize a tensor to track the best (lowest) error for each sample.
            best = ops.full((ops.shape(x)[0],), float('inf'), dtype=x.dtype)

            # Iteratively search for the best quantization range by shrinking it.
            for i in range(int(self.maxshrink * self.grid)):
                # Calculate the shrinkage factor
                p = 1.0 - i / self.grid

                # Temporarily shrink the min/max range
                xmin1 = p * xmin
                xmax1 = p * xmax

                # Recalculate scale and zero-point for the new shrunken range
                scale1 = (xmax1 - xmin1) / self.maxq
                if self.sym:
                    zero1 = self.zero
                else:
                    zero1 = ops.round(-xmin1 / scale1)
                
                # Quantize the input tensor with the temporary parameters.
                # ops.expand_dims is needed to make the 1D scale/zero tensors
                # broadcastable with the 2D input tensor 'x'.
                q = quantize(
                    x,
                    ops.expand_dims(scale1, axis=1),
                    ops.expand_dims(zero1, axis=1),
                    self.maxq
                )

                # Calculate the quantization error (e.g., L_p norm).
                # Note the use of functional, out-of-place operations.
                err_q = q - x
                err_q = ops.abs(err_q)
                err_q = ops.power(err_q, self.norm)
                err = ops.sum(err_q, axis=1)

                # Create a boolean mask for samples where the new error is an improvement.
                is_better_mask = err < best

                # If any sample's error improved, update the best-known parameters for those samples.
                if ops.any(is_better_mask):
                    # Use ops.where for immutable, masked updates.
                    best = ops.where(is_better_mask, err, best)
                    self.scale = ops.where(is_better_mask, scale1, self.scale)
                    self.zero = ops.where(is_better_mask, zero1, self.zero)
            
        if not self.perchannel:
            # This logic determines how many times to repeat the scalar scale/zero.
            # It aims to match the number of channels.
            if weight:
                # For a weight tensor (e.g., [out_channels, in_channels, ...]),
                # we use the first dimension.
                reps = shape[0]
            else:
                # For an activation tensor (e.g., [batch, channels, ...]),
                # we typically use the second dimension.
                if len(shape) == 3:
                    # Handling the specific 3D case from the original code.
                    reps = shape[2]
                else:
                    reps = shape[1]

            # ops.tile is the Keras equivalent of torch's .repeat().
            # It repeats the tensor 'reps' times along the first axis.
            self.scale = ops.tile(self.scale, [reps])
            self.zero = ops.tile(self.zero, [reps])

        if weight:
            # This reshapes a 1D scale/zero tensor to broadcast with a weight tensor.
            # e.g., for a 4D weight, a scale of shape [C] becomes [C, 1, 1, 1].
            new_shape = [-1] + [1] * (len(shape) - 1)
            self.scale = ops.reshape(self.scale, new_shape)
            self.zero = ops.reshape(self.zero, new_shape)
            return  # Keep the return statement from the original code

        # The following reshapes are for broadcasting with an activation tensor.
        # See the note below about data formats.

        if len(shape) == 4:
            # Reshapes for a 4D activation, e.g., to [1, C, 1, 1]
            self.scale = ops.reshape(self.scale, (1, -1, 1, 1))
            self.zero = ops.reshape(self.zero, (1, -1, 1, 1))
            
        if len(shape) == 3:
            # Reshapes for a 3D activation, e.g., to [1, 1, C]
            self.scale = ops.reshape(self.scale, (1, 1, -1))
            self.zero = ops.reshape(self.zero, (1, 1, -1))
            
        if len(shape) == 2:
            # Reshapes for a 2D activation, e.g., to [1, C]
            # ops.expand_dims is the equivalent of PyTorch's unsqueeze.
            self.scale = ops.expand_dims(self.scale, axis=0)
            self.zero = ops.expand_dims(self.zero, axis=0)
    
    def quantize(self, x):
        """
        Quantizes the input tensor 'x' if the model's quantization 
        parameters are ready.
        """
        if self.ready():
            # Call the main quantize function with the layer's scale and zero-point
            return quantize(x, self.scale, self.zero, self.maxq)
        # If not ready, return the input tensor unmodified
        return x

    def enabled(self):
        """Checks if quantization is enabled for this layer."""
        return self.maxq > 0

    def ready(self):
        """
        Checks if the quantization scale has been properly initialized 
        (i.e., is not zero).
        """
        # ops.all is the Keras equivalent of torch.all
        return ops.all(self.scale != 0)