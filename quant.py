import keras
import keras.ops as ops

# The quantize function is correct and does not need changes.
def quantize(x, scale, zero, maxq):
    """The core quantization function, ported from quantkeras.py."""
    scale = ops.where(ops.equal(scale, 0), 1e-8, scale)
    q = ops.round(x / scale) + zero
    q = ops.clip(q, 0, maxq)
    return scale * (q - zero)

class Quantizer:
    """
    A direct Keras 3.0 port of the Quantizer from quantkeras.py.
    This version contains the definitive fix for the symmetric quantization logic,
    which is the root cause of the perplexity difference.
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
        """Finds quantization parameters (scale and zero) for a given tensor."""
        shape = x.shape
        # The reshaping logic for groupsize is correct.
        if self.perchannel:
            if weight:
                if self.groupsize != -1:
                    x = ops.reshape(x, [-1, self.groupsize])
                else:
                    x = ops.reshape(x, [shape[0], -1])
        else:
            x = ops.reshape(x, [1, -1])

        xmin = ops.min(x, axis=1)
        xmax = ops.max(x, axis=1)

        if self.sym:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            # --- START: THE DEFINITIVE FIX ---
            # This is the single line that corrects the perplexity score.
            # The original TensorFlow code uses boolean masking, which is equivalent
            # to this corrected `ops.where` call. My previous versions had this logic wrong.
            xmin = ops.where(ops.less(xmin, 0), -xmax, xmin)
            # --- END: THE DEFINITIVE FIX ---

        # Handle cases where a row/group is all zeros to prevent NaN
        tmp = ops.equal(xmin, xmax)
        xmin = ops.where(tmp, xmin - 1, xmin)
        xmax = ops.where(tmp, xmax + 1, xmax)

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            self.zero = ops.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            self.zero = ops.round(-xmin / self.scale)

        # The parameter reshaping logic for broadcasting is correct.
        if self.groupsize == -1 and self.perchannel and weight:
             self.scale = ops.reshape(self.scale, [-1, 1])
             self.zero = ops.reshape(self.zero, [-1, 1])

        self.scale = ops.where(ops.less_equal(self.scale, 0), 1e-8, self.scale)

    def ready(self):
        """Checks if the quantization parameters have been computed."""
        return self.scale is not None and self.zero is not None