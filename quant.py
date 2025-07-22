import keras
import keras.ops as ops

def quantize(x, scale, zero, maxq):
    """The core quantization function, ported from quantkeras.py."""
    # Ensure scale is safe to prevent division by zero.
    scale = ops.where(ops.equal(scale, 0), 1e-8, scale)
    q = ops.round(x / scale) + zero
    q = ops.clip(q, 0, maxq)
    return scale * (q - zero)

class Quantizer:
    """
    A direct Keras 3.0 port of the Quantizer from quantkeras.py.
    This version computes broadcastable scale/zero vectors for the current
    weight group, which is the correct behavior of the original algorithm.
    """
    def __init__(self, shape=1):
        self.scale = None
        self.zero = None
        self.maxq = None
        self.wbits = None
        self.perchannel = False
        self.sym = False
        self.groupsize = -1

    def configure(self, wbits, perchannel=True, sym=False, groupsize=-1):
        """Configures the quantizer settings."""
        self.wbits = wbits
        self.maxq = ops.cast((2 ** wbits) - 1, 'float32')
        self.perchannel = perchannel
        self.sym = sym
        self.groupsize = groupsize

    def find_params(self, x, weight=False):
        """
        Finds quantization parameters for a given tensor `x` (e.g., a weight group).
        """
        if self.perchannel:
            # For per-channel, we want one scale/zero per row.
            x_reshaped = x
        else:
            # For per-tensor, we treat the whole input as one group.
            x_reshaped = ops.reshape(x, [1, -1])

        xmin = ops.min(x_reshaped, axis=1)
        xmax = ops.max(x_reshaped, axis=1)

        if self.sym:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            xmin = -xmax

        # Handle cases where a row/group is all zeros to prevent NaN
        tmp = ops.equal(xmin, xmax)
        xmin = ops.where(tmp, xmin - 1, xmin)
        xmax = ops.where(tmp, xmax + 1, xmax)

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = ops.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = ops.round(-xmin / self.scale)

        # CRITICAL FIX: Reshape scale/zero to be broadcastable with weight columns.
        if self.perchannel:
            self.scale = ops.reshape(self.scale, [-1, 1])
            self.zero = ops.reshape(self.zero, [-1, 1])
        
        # Ensure scale is not zero
        self.scale = ops.where(ops.less_equal(self.scale, 0), 1e-8, self.scale)

    def ready(self):
        """Checks if the quantization parameters have been computed."""
        return self.scale is not None and self.zero is not None