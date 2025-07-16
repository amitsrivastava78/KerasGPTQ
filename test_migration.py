import numpy as np
import keras
import tensorflow as tf
import itertools

# --- Import both original TF and migrated Keras 3 versions ---
# Use aliases to handle the new file names while keeping test code readable
import quant as k3_quantkeras
import gptq as k3_gptqkeras_fixed
import quantkeras as tf_quantkeras
import gptqkeras_fixed as tf_gptqkeras_fixed

# --- Model Dimensions (Matching opt-125m) ---
EMBED_DIM = 768
FFN_DIM = 3072 # 4 * EMBED_DIM

# --- Test Data Parameters ---
NUM_SAMPLES = 2 # Use a small number of calibration samples for speed
SEQ_LEN = 32   # Sequence length

class MockSelfAttention(keras.Layer):
    """A mock self-attention block to hold the layers for testing."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.q_proj = keras.layers.Dense(embed_dim, name="q_proj")
        self.k_proj = keras.layers.Dense(embed_dim, name="k_proj")
        self.v_proj = keras.layers.Dense(embed_dim, name="v_proj")
        self.out_proj = keras.layers.Dense(embed_dim, name="out_proj")

class MockMLP(keras.Layer):
    """A mock MLP block to hold the layers for testing."""
    def __init__(self, embed_dim, ffn_dim, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = keras.layers.Dense(ffn_dim, name="fc1")
        self.fc2 = keras.layers.Dense(embed_dim, name="fc2")

def build_mock_transformer_block(embed_dim, ffn_dim):
    """Builds mock layers that mimic a transformer block."""
    print("Building mock layers...")
    self_attn = MockSelfAttention(embed_dim, name="mock_self_attention")
    mlp = MockMLP(embed_dim, ffn_dim, name="mock_mlp")
    
    # Create a dummy input to build all the layers and initialize their weights
    print("Building sub-layers with a dummy call...")
    dummy_input = np.random.rand(1, embed_dim).astype(np.float32)
    _ = self_attn.q_proj(dummy_input)
    _ = self_attn.k_proj(dummy_input)
    _ = self_attn.v_proj(dummy_input)
    _ = self_attn.out_proj(dummy_input)
    
    dummy_mlp_input = np.random.rand(1, embed_dim).astype(np.float32)
    mlp_hidden = mlp.fc1(dummy_mlp_input)
    _ = mlp.fc2(mlp_hidden)
    print("Sub-layers built successfully.")
    
    return {
        "self_attn.q_proj": self_attn.q_proj,
        "self_attn.k_proj": self_attn.k_proj,
        "self_attn.v_proj": self_attn.v_proj,
        "self_attn.out_proj": self_attn.out_proj,
        "mlp.fc1": mlp.fc1,
        "mlp.fc2": mlp.fc2,
    }

def build_test_data(input_features):
    """Creates random activation data for a given layer's input dimension."""
    return np.random.rand(NUM_SAMPLES * SEQ_LEN, input_features).astype(np.float32)

def run_single_test(name, layer, activations, config):
    """
    Runs a single equivalence test for a given layer and configuration.
    """
    actorder = config["actorder"]
    groupsize = config["groupsize"]
    sym = config["sym"]
    
    print(f"\n--- Testing Layer: {name} with config: actorder={actorder}, groupsize={groupsize}, sym={sym} ---")
    
    # --- TensorFlow Version ---
    tf_gptq = tf_gptqkeras_fixed.GPTQ(layer)
    tf_quantizer = tf_quantkeras.Quantizer()
    tf_quantizer.configure(bits=4, perchannel=True, sym=sym, groupsize=groupsize)
    tf_gptq.quantizer = tf_quantizer
    tf_gptq.add_batch(tf.convert_to_tensor(activations, dtype=tf.float32), None)
    tf_q = tf_gptq.fasterquant(actorder=actorder, groupsize=groupsize)

    # --- Keras 3 Version ---
    k3_gptq = k3_gptqkeras_fixed.GPTQ(layer)
    k3_quantizer = k3_quantkeras.Quantizer()
    k3_quantizer.configure(bits=4, perchannel=True, sym=sym, groupsize=groupsize)
    k3_gptq.quantizer = k3_quantizer
    k3_gptq.add_batch(keras.ops.convert_to_tensor(activations, dtype='float32'), None)
    k3_q = k3_gptq.fasterquant(actorder=actorder, groupsize=groupsize)

    # --- Compare final quantized weights ---
    try:
        np.testing.assert_allclose(tf_q, keras.ops.convert_to_numpy(k3_q), atol=1e-5, rtol=1e-5)
        print(f"✅ Test PASSED for {name} with actorder={actorder}, groupsize={groupsize}, sym={sym}!")
        return True
    except AssertionError as e:
        print(f"❌ Test FAILED for {name} with actorder={actorder}, groupsize={groupsize}, sym={sym}: Final 'Q' matrices are not equivalent!")
        # print(e) # Uncomment for detailed numpy error
        return False

if __name__ == "__main__":
    # 1. Build the mock transformer block
    mock_layers = build_mock_transformer_block(EMBED_DIM, FFN_DIM)
    
    # 2. Build common test data
    print("Building test data...")
    attn_activations = build_test_data(EMBED_DIM)
    mlp_activations = build_test_data(FFN_DIM)
    
    # 3. Define test configurations
    test_configs = list(itertools.product(
        [True, False],  # actorder
        [-1, 128],      # groupsize
        [True, False]   # sym
    ))

    all_tests_passed = True
    
    # 4. Run tests for all layers and configurations
    for name, layer in mock_layers.items():
        # Select the correct activations based on the layer
        activations = mlp_activations if name == 'mlp.fc2' else attn_activations
        
        for config_dict in test_configs:
            config = {"actorder": config_dict[0], "groupsize": config_dict[1], "sym": config_dict[2]}
            
            # The 'sym' flag is only relevant when groupsize is active. For per-channel, both are the same.
            # We can skip redundant tests to save time.
            if config["groupsize"] == -1 and config["sym"] is False:
                continue

            passed = run_single_test(name, layer, activations, config)
            if not passed:
                all_tests_passed = False

    print("\n" + "="*40)
    if all_tests_passed:
        print("🎉 SUCCESS: All comprehensive tests passed!")
    else:
        print("🔥 FAILURE: Some tests did not pass. Please review the logs.")
    print("="*40)

