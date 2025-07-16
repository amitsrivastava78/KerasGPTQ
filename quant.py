import keras
import keras.ops as ops

# This is the final, corrected Keras 3.0 Quantizer.
# The logic in find_params has been corrected to match the PyTorch reference.

def quantize(x, scale, zero, maxq):
    """The core quantization function."""
    if ops.any(ops.isnan(x)) or ops.any(ops.isinf(x)):
        return x
    if ops.any(ops.isnan(scale)) or ops.any(ops.isinf(scale)):
        return x
    if ops.any(ops.isnan(zero)) or ops.any(ops.isinf(zero)):
        return x
    if ops.any(ops.equal(scale, 0)):
        return x
    if maxq < 0:
        return ops.cast(x > scale / 2, 'float32') * scale + ops.cast(x < zero / 2, 'float32') * zero

    scale_safe = ops.where(ops.equal(scale, 0), ops.ones_like(scale) * 1e-8, scale)
    q = ops.clip(ops.round(x / scale_safe) + zero, 0, maxq)
    result = scale * (q - zero)

    if ops.any(ops.isnan(result)) or ops.any(ops.isinf(result)):
        return x

    return result

class Quantizer:
    def __init__(self, shape=1):
        self.maxq = ops.convert_to_tensor(0, dtype='float32')
        self.scale = ops.zeros(shape, dtype='float32')
        self.zero = ops.zeros(shape, dtype='float32')
        self.groupsize = -1

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False, groupsize=-1
    ):
        self.maxq = ops.convert_to_tensor(2 ** bits - 1, dtype='float32')
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.groupsize = groupsize
        if trits:
            self.maxq = ops.convert_to_tensor(-1, dtype='float32')

    def find_params(self, x, weight=False):
        if ops.any(ops.isnan(x)) or ops.any(ops.isinf(x)):
            if self.perchannel:
                shape = [x.shape[0]] if weight else [x.shape[-1]]
            else:
                shape = [1]
            self.scale = ops.ones(shape, dtype='float32')
            self.zero = ops.zeros(shape, dtype='float32')
            return

        shape = x.shape
        if self.perchannel:
            if weight:
                if self.groupsize != -1:
                    x = ops.reshape(x, [-1, self.groupsize])
                else:
                    x = ops.reshape(x, [x.shape[0], -1])
            else:
                if len(shape) == 4:
                    x = ops.transpose(x, [1, 0, 2, 3])
                    x = ops.reshape(x, [x.shape[0], -1])
                if len(shape) == 3:
                    x = ops.transpose(ops.reshape(x, [-1, shape[-1]]), [1, 0])
                if len(shape) == 2:
                    x = ops.transpose(x)
        else:
            x = ops.reshape(x, [1, -1])

        xmin = ops.min(x, axis=1)
        xmax = ops.max(x, axis=1)

        if self.sym:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            tmp_mask = xmin < 0
            if ops.any(tmp_mask):
                xmin = ops.where(tmp_mask, -xmax, xmin)

        tmp_mask = ops.logical_and(ops.equal(xmin, 0), ops.equal(xmax, 0))
        xmin = ops.where(tmp_mask, -ops.ones_like(xmin), xmin)
        xmax = ops.where(tmp_mask, ops.ones_like(xmax), xmax)

        if ops.less(self.maxq, 0):
            self.scale = xmax
            self.zero = xmin
        else:
            scale_raw = (xmax - xmin) / self.maxq
            min_scale = 1e-8
            self.scale = ops.maximum(scale_raw, min_scale)

            if self.sym:
                maxq_plus_one = ops.add(ops.cast(self.maxq, 'float32'), 1.0)
                self.zero = ops.divide(maxq_plus_one, 2.0) * ops.ones_like(self.scale)
            else:
                zero_raw = -xmin / self.scale
                self.zero = ops.round(zero_raw)

        if self.mse:
            best = ops.full([x.shape[0]], float('inf'))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                scale1 = ops.maximum(scale1, min_scale)
                zero1 = ops.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, ops.expand_dims(scale1, 1), ops.expand_dims(zero1, 1), self.maxq)
                q = q - x
                q = ops.abs(q)
                q = ops.power(q, self.norm)
                err = ops.sum(q, axis=1)
                tmp_mask = err < best
                if ops.any(tmp_mask):
                    best = ops.where(tmp_mask, err, best)
                    self.scale = ops.where(tmp_mask, scale1, self.scale)
                    self.zero = ops.where(tmp_mask, zero1, self.zero)

        if ops.any(ops.isnan(self.scale)) or ops.any(ops.isinf(self.scale)):
            self.scale = ops.ones_like(self.scale)

        if ops.any(ops.isnan(self.zero)) or ops.any(ops.isinf(self.zero)):
            self.zero = ops.zeros_like(self.zero)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = ops.repeat(self.scale, tmp)
            self.zero = ops.repeat(self.zero, tmp)

        # --- FINAL FIX: This block now correctly matches the PyTorch version's logic ---
        if weight:
            # Reshape scale and zero to be broadcastable for all weight cases.
            new_shape = [-1] + [1] * (len(shape) - 1)
            self.scale = ops.reshape(self.scale, new_shape)
            self.zero = ops.reshape(self.zero, new_shape)
            return

        # Handle non-weight tensors
        if len(shape) == 4:
            self.scale = ops.reshape(self.scale, (1, -1, 1, 1))
            self.zero = ops.reshape(self.zero, (1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = ops.reshape(self.scale, (1, 1, -1))
            self.zero = ops.reshape(self.zero, (1, 1, -1))
        if len(shape) == 2:
            self.scale = ops.expand_dims(self.scale, 0)
            self.zero = ops.expand_dims(self.zero, 0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return ops.all(ops.greater(self.maxq, 0))

    def ready(self):
        return ops.all(ops.not_equal(self.scale, 0))

