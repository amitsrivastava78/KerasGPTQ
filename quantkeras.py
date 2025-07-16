import numpy as np
import tensorflow as tf
import keras

ops = tf  # Keras 3.0 ops API

# Quantize function for Keras ops (equivalent to PyTorch version)
def quantize(x, scale, zero, maxq):
    if tf.reduce_any(tf.math.is_nan(x)) or tf.reduce_any(tf.math.is_inf(x)):
        return x
    if tf.reduce_any(tf.math.is_nan(scale)) or tf.reduce_any(tf.math.is_inf(scale)):
        return x
    if tf.reduce_any(tf.math.is_nan(zero)) or tf.reduce_any(tf.math.is_inf(zero)):
        return x
    if tf.reduce_any(tf.equal(scale, 0)):
        return x
    if maxq < 0:
        return tf.cast(x > scale / 2, tf.float32) * scale + tf.cast(x < zero / 2, tf.float32) * zero

    scale_safe = tf.where(tf.equal(scale, 0), tf.ones_like(scale) * 1e-8, scale)
    q = tf.clip_by_value(tf.round(x / scale_safe) + zero, 0, maxq)
    result = scale * (q - zero)

    if tf.reduce_any(tf.math.is_nan(result)) or tf.reduce_any(tf.math.is_inf(result)):
        return x

    return result

class Quantizer:
    def __init__(self, shape=1):
        self.maxq = tf.convert_to_tensor(0, dtype=tf.float32)
        self.scale = tf.zeros(shape, dtype=tf.float32)
        self.zero = tf.zeros(shape, dtype=tf.float32)
        self.groupsize = -1

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False, groupsize=-1
    ):
        self.maxq = tf.convert_to_tensor(2 ** bits - 1, dtype=tf.float32)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.groupsize = groupsize
        if trits:
            self.maxq = tf.convert_to_tensor(-1, dtype=tf.float32)

    def find_params(self, x, weight=False):
        if tf.reduce_any(tf.math.is_nan(x)) or tf.reduce_any(tf.math.is_inf(x)):
            if self.perchannel:
                shape = [x.shape[0]] if weight else [x.shape[-1]]
            else:
                shape = [1]
            self.scale = tf.ones(shape, dtype=tf.float32)
            self.zero = tf.zeros(shape, dtype=tf.float32)
            return

        shape = x.shape
        if self.perchannel:
            if weight:
                # --- START OF GROUP-WISE FIX ---
                if self.groupsize != -1:
                    # Reshape for group-wise quantization
                    x = tf.reshape(x, [-1, self.groupsize])
                else:
                    x = tf.reshape(x, [x.shape[0], -1])
                # --- END OF GROUP-WISE FIX ---
            else:
                if len(shape) == 4:
                    x = tf.transpose(x, [1, 0, 2, 3])
                    x = tf.reshape(x, [x.shape[0], -1])
                if len(shape) == 3:
                    x = tf.transpose(tf.reshape(x, [-1, shape[-1]]), [1, 0])
                if len(shape) == 2:
                    x = tf.transpose(x)
        else:
            x = tf.reshape(x, [1, -1])

        tmp = tf.zeros([x.shape[1]], dtype=x.dtype)
        xmin = tf.reduce_min(x, axis=1)
        xmax = tf.reduce_max(x, axis=1)

        if self.sym:
            xmax = tf.maximum(tf.abs(xmin), xmax)
            tmp_mask = xmin < 0
            if tf.reduce_any(tmp_mask):
                xmin = tf.where(tmp_mask, -xmax, xmin)

        tmp_mask = tf.logical_and(tf.equal(xmin, 0), tf.equal(xmax, 0))
        xmin = tf.where(tmp_mask, -tf.ones_like(xmin), xmin)
        xmax = tf.where(tmp_mask, tf.ones_like(xmax), xmax)

        if tf.less(self.maxq, 0):
            self.scale = xmax
            self.zero = xmin
        else:
            scale_raw = (xmax - xmin) / self.maxq
            min_scale = 1e-8
            self.scale = tf.maximum(scale_raw, min_scale)

            if self.sym:
                maxq_plus_one = tf.add(tf.cast(self.maxq, tf.float32), 1.0)
                self.zero = tf.fill(tf.shape(self.scale), tf.divide(maxq_plus_one, 2.0))
            else:
                zero_raw = -xmin / self.scale
                self.zero = tf.round(zero_raw)

        if self.mse:
            best = tf.fill([x.shape[0]], float('inf'))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                scale1 = tf.maximum(scale1, min_scale)
                zero1 = tf.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, tf.expand_dims(scale1, 1), tf.expand_dims(zero1, 1), self.maxq)
                q = q - x
                q = tf.abs(q)
                q = tf.pow(q, self.norm)
                err = tf.reduce_sum(q, axis=1)
                tmp_mask = err < best
                if tf.reduce_any(tmp_mask):
                    best = tf.where(tmp_mask, err, best)
                    self.scale = tf.where(tmp_mask, scale1, self.scale)
                    self.zero = tf.where(tmp_mask, zero1, self.zero)

        if tf.reduce_any(tf.math.is_nan(self.scale)) or tf.reduce_any(tf.math.is_inf(self.scale)):
            self.scale = tf.ones_like(self.scale)

        if tf.reduce_any(tf.math.is_nan(self.zero)) or tf.reduce_any(tf.math.is_inf(self.zero)):
            self.zero = tf.zeros_like(self.zero)

        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = tf.repeat(self.scale, tmp)
            self.zero = tf.repeat(self.zero, tmp)

        # --- START OF FINAL SHAPE CORRECTION ---
        # The key fix is to NOT reshape the scale and zero tensors
        # when using group-wise quantization, as they should remain
        # as 1D tensors with one entry per group.
        if self.groupsize == -1:
            if weight:
                shape = [-1] + [1] * (len(shape) - 1)
                self.scale = tf.reshape(self.scale, shape)
                self.zero = tf.reshape(self.zero, shape)
                return
            if len(shape) == 4:
                self.scale = tf.reshape(self.scale, (1, -1, 1, 1))
                self.zero = tf.reshape(self.zero, (1, -1, 1, 1))
            if len(shape) == 3:
                self.scale = tf.reshape(self.scale, (1, 1, -1))
                self.zero = tf.reshape(self.zero, (1, 1, -1)) 
            if len(shape) == 2:
                self.scale = tf.expand_dims(self.scale, 0)
                self.zero = tf.expand_dims(self.zero, 0)
        # --- END OF FINAL SHAPE CORRECTION ---


    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return tf.reduce_all(tf.greater(self.maxq, 0))

    def ready(self):
        return tf.reduce_all(tf.not_equal(self.scale, 0))
