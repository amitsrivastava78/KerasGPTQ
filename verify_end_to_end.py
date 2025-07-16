import numpy as np
import keras
import tensorflow as tf # Keep for tf.convert_to_tensor in the TF-specific function
import argparse
import time
import copy

# Import both original TF and migrated Keras 3 versions using aliases
# This assumes your file names are:
# - gptq.py (migrated Keras 3)
# - quant.py (migrated Keras 3)
# - gptqkeras_fixed.py (original TF)
# - quantkeras.py (original TF)
import gptq as k3_gptq_module
import quant as k3_quant_module
import gptqkeras_fixed as tf_gptq_module
import quantkeras as tf_quant_module

# --- Mock Model and Data Setup ---

# Use OPT-125m dimensions for a realistic test
EMBED_DIM = 768
FFN_DIM = 3072
NUM_HEADS = 12
NUM_BLOCKS = 2 # Use 2 blocks for a faster but still representative test

# --- FIX: Add get_config to all custom layers for cloning ---

class MockSelfAttention(keras.layers.Layer):
    """A mock self-attention block to hold the dense layers."""
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = keras.layers.Dense(embed_dim, name="q_proj")
        self.k_proj = keras.layers.Dense(embed_dim, name="k_proj")
        self.v_proj = keras.layers.Dense(embed_dim, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, name="out_proj")

    def call(self, inputs):
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        return self.out_proj(q)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
        })
        return config

class MockMLP(keras.layers.Layer):
    """A mock MLP block."""
    def __init__(self, embed_dim, ffn_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.fc1 = keras.layers.Dense(ffn_dim, name="fc1")
        self.fc2 = keras.layers.Dense(embed_dim, name="fc2")
        self.activation = keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.activation(x)
        return self.fc2(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ffn_dim": self.ffn_dim,
        })
        return config

class MockTransformerBlock(keras.layers.Layer):
    """A mock transformer block combining Attention and MLP."""
    def __init__(self, embed_dim, ffn_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.self_attn = MockSelfAttention(embed_dim, num_heads, name="self_attn")
        self.mlp = MockMLP(embed_dim, ffn_dim, name="mlp")

    def call(self, inputs):
        attn_output = self.self_attn(inputs)
        return self.mlp(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
        })
        return config

class MockModel(keras.Model):
    """A mock OPT-style model."""
    def __init__(self, embed_dim, ffn_dim, num_heads, num_blocks, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.blocks = [
            MockTransformerBlock(embed_dim, ffn_dim, num_heads, name=f"block_{i}")
            for i in range(num_blocks)
        ]

    def call(self, inputs):
        x = inputs
        for block in self.blocks:
            x = block(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "ffn_dim": self.ffn_dim,
            "num_heads": self.num_heads,
            "num_blocks": self.num_blocks,
        })
        return config

def build_test_model_and_data(num_samples=4, seq_len=128):
    """Builds two identical models and random activation data."""
    print("Building mock models and test data...")
    
    keras.utils.set_random_seed(42)

    model1 = MockModel(EMBED_DIM, FFN_DIM, NUM_HEADS, NUM_BLOCKS)
    dummy_input = np.random.rand(num_samples, seq_len, EMBED_DIM).astype(np.float32)
    _ = model1(dummy_input)
    print("Model 1 built.")

    model2 = keras.models.clone_model(model1)
    model2.set_weights(model1.get_weights())
    print("Model 2 cloned and weights set.")

    for i in range(len(model1.weights)):
        assert np.allclose(model1.weights[i].numpy(), model2.weights[i].numpy()), \
            f"Models did not have identical weights after cloning! Mismatch at weight {i} ({model1.weights[i].name})."

    print("Models built successfully with identical initial weights.")
    
    calibration_data = [
        np.random.rand(1, seq_len, EMBED_DIM).astype(np.float32)
        for _ in range(num_samples)
    ]
    
    return model1, model2, calibration_data


# --- Wrapper functions for quantization processes ---

def run_tf_quantization(model, calibration_data):
    """Runs the entire quantization process using the original TF files."""
    print("\n--- Running Original TensorFlow Quantization Process ---")
    
    for i in range(NUM_BLOCKS):
        block = model.blocks[i]
        sub_layers = {
            "q_proj": block.self_attn.q_proj, "k_proj": block.self_attn.k_proj,
            "v_proj": block.self_attn.v_proj, "out_proj": block.self_attn.out_proj,
            "fc1": block.mlp.fc1, "fc2": block.mlp.fc2,
        }
        
        np.random.seed(i)
        activations_attn = [np.random.rand(calibration_data[0].shape[1], EMBED_DIM).astype(np.float32) for _ in calibration_data]
        np.random.seed(i + 100)
        activations_fc2 = [np.random.rand(calibration_data[0].shape[1], FFN_DIM).astype(np.float32) for _ in calibration_data]

        for name, layer in sub_layers.items():
            print(f"Quantizing {block.name}/{layer.name} with TF implementation...")
            current_activations = activations_fc2 if name == 'fc2' else activations_attn

            tf_gptq = tf_gptq_module.GPTQ(layer)
            tf_quantizer = tf_quant_module.Quantizer()
            tf_quantizer.configure(bits=4, perchannel=True, sym=False, groupsize=-1)
            tf_gptq.quantizer = tf_quantizer
            
            for act in current_activations:
                tf_gptq.add_batch(tf.convert_to_tensor(act), None)

            q_weights = tf_gptq.fasterquant(actorder=False)
            layer.weights[0].assign(tf.transpose(q_weights))
            tf_gptq.free()
    
    print("TensorFlow quantization complete.")
    return model

def run_k3_quantization(model, calibration_data):
    """Runs the entire quantization process using the migrated Keras 3 files."""
    print("\n--- Running Migrated Keras 3 Quantization Process ---")

    for i in range(NUM_BLOCKS):
        block = model.blocks[i]
        sub_layers = {
            "q_proj": block.self_attn.q_proj, "k_proj": block.self_attn.k_proj,
            "v_proj": block.self_attn.v_proj, "out_proj": block.self_attn.out_proj,
            "fc1": block.mlp.fc1, "fc2": block.mlp.fc2,
        }
        
        np.random.seed(i)
        activations_attn = [np.random.rand(calibration_data[0].shape[1], EMBED_DIM).astype(np.float32) for _ in calibration_data]
        np.random.seed(i + 100)
        activations_fc2 = [np.random.rand(calibration_data[0].shape[1], FFN_DIM).astype(np.float32) for _ in calibration_data]

        for name, layer in sub_layers.items():
            print(f"Quantizing {block.name}/{layer.name} with Keras 3 implementation...")
            current_activations = activations_fc2 if name == 'fc2' else activations_attn

            k3_gptq = k3_gptq_module.GPTQ(layer)
            k3_quantizer = k3_quant_module.Quantizer()
            k3_quantizer.configure(bits=4, perchannel=True, sym=False, groupsize=-1)
            k3_gptq.quantizer = k3_quantizer
            
            for act in current_activations:
                k3_gptq.add_batch(keras.ops.convert_to_tensor(act), None)

            q_weights = k3_gptq.fasterquant(actorder=False)
            layer.weights[0].assign(keras.ops.transpose(q_weights))
            k3_gptq.free()

    print("Keras 3 quantization complete.")
    return model


if __name__ == "__main__":
    # 1. Build two identical models and the data
    tf_model, k3_model, calib_data = build_test_model_and_data()

    # 2. Run both quantization processes
    tf_model_quantized = run_tf_quantization(tf_model, calib_data)
    k3_model_quantized = run_k3_quantization(k3_model, calib_data)
    
    # 3. Compare the final weights of both models
    print("\n--- Comparing Final Model Weights ---")
    
    all_passed = True
    # Compare trainable weights (kernels)
    for i, (w_tf, w_k3) in enumerate(zip(tf_model_quantized.trainable_weights, k3_model_quantized.trainable_weights)):
        print(f"Comparing kernel: {w_tf.name}")
        if not np.allclose(w_tf.numpy(), w_k3.numpy(), atol=1e-5):
            print(f"❌ MISMATCH FOUND in kernel {w_tf.name}!")
            all_passed = False

    # Compare non-trainable weights (biases) - they should not be touched
    if all_passed:
        for i, (w_tf, w_k3) in enumerate(zip(tf_model_quantized.non_trainable_weights, k3_model_quantized.non_trainable_weights)):
            print(f"Comparing bias: {w_tf.name}")
            if not np.allclose(w_tf.numpy(), w_k3.numpy(), atol=1e-5):
                print(f"❌ MISMATCH FOUND in bias {w_tf.name}!")
                all_passed = False

    print("\n" + "="*40)
    if all_passed:
        print("🎉 SUCCESS: All quantized model weights are identical!")
    else:
        print("🔥 FAILURE: Quantized model weights do NOT match.")
    print("="*40)
