import keras
import keras.ops as ops

def _round_half_up(x):
    """
    Implements the "round half up" behavior of older TensorFlow versions.
    """
    return ops.sign(x) * ops.floor(ops.abs(x) + 0.5)

def quantize(x, scale, zero, maxq):
    """
    Quantizes a tensor using the corrected rounding method.
    """
    maxq = ops.cast(maxq, x.dtype)
    
    # Use the custom rounding function instead of ops.round
    q = _round_half_up(x / scale) + zero
    
    q = ops.clip(q, 0, maxq)
    return scale * (q - zero)


class Quantizer:
    """
    A class to manage the state of quantization parameters.
    It finds the scale and zero-point for a given tensor.
    """

    def __init__(self, shape=1):
        """
        Initializes the Quantizer.
        """
        self.shape = shape
        self.scale = None
        self.zero = None
        self.wbits = None
        self.perchannel = None
        self.sym = None
        self.maxq = None

    def configure(self, wbits, perchannel=True, sym=True):
        """
        Configures the quantizer settings.
        """
        self.wbits = wbits
        self.perchannel = perchannel
        self.sym = sym
        self.maxq = (2 ** self.wbits) - 1

    def find_params(self, x, weight=False):
        """
        Finds the quantization parameters (scale and zero-point) for a tensor.
        """
        if self.wbits is None:
            raise ValueError("Quantizer must be configured before finding parameters. Call `configure()` first.")

        dim = 0 if self.perchannel and weight else None

        if self.sym:
            abs_max = ops.max(ops.abs(x), axis=dim)
            self.scale = abs_max / ((self.maxq + 1) / 2 - 1)
            self.zero = ops.full_like(self.scale, (self.maxq + 1) / 2)
        else:
            if dim is None:
                xmin, xmax = ops.min(x), ops.max(x)
            else:
                xmin, xmax = ops.min(x, axis=dim), ops.max(x, axis=dim)
            
            self.scale = (xmax - xmin) / self.maxq
            
            # Also use the custom rounding for the zero-point calculation
            self.zero = _round_half_up(-xmin / self.scale)

        epsilon = 1e-8
        self.scale = ops.where(ops.abs(self.scale) < epsilon, epsilon, self.scale)

    def ready(self):
        """
        Checks if the quantization parameters have been computed.
        """
        return self.scale is not None and self.zero is not None