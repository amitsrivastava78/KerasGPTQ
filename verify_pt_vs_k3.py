import numpy as np
import itertools
import torch
import torch.nn as nn
import keras
import keras.ops as ops

# --- Import the modules to be tested ---
# PyTorch versions
import pquant
import pgptq
# Keras 3.0 versions
import quant as k3_quant
import gptq as k3_gptq_module # Use a distinct alias to avoid name clashes

# --- Test Configuration ---
# Use OPT-125m dimensions for a realistic test
EMBED_DIM = 768
FFN_DIM = 3072

# Test data parameters
NUM_SAMPLES = 4
SEQ_LEN = 32

def build_and_run_test(initial_weights, initial_activations, config):
    """
    Builds both PyTorch and Keras layers with identical state,
    runs both quantization implementations, and compares the results.
    """
    actorder = config["actorder"]
    groupsize = config["groupsize"]
    sym = config["sym"]

    config_str = f"actorder={actorder}, groupsize={groupsize}, sym={sym}"
    print(f"\n--- Testing with config: {config_str} ---")

    # --- PyTorch Setup & Run ---
    pt_layer = nn.Linear(EMBED_DIM, FFN_DIM, bias=False)
    pt_layer.weight.data = torch.from_numpy(initial_weights)

    pt_gptq = pgptq.GPTQ(pt_layer)
    pt_quantizer = pquant.Quantizer()
    pt_quantizer.configure(bits=4, perchannel=True, sym=sym)
    pt_gptq.quantizer = pt_quantizer
    
    pt_activations_torch = torch.from_numpy(initial_activations.reshape(NUM_SAMPLES, SEQ_LEN, -1))
    pt_gptq.add_batch(pt_activations_torch, None)
    
    pt_gptq.fasterquant(actorder=actorder, groupsize=groupsize)
    pt_final_q_numpy = pt_layer.weight.data.numpy()


    # --- Keras 3.0 Setup & Run ---
    k3_layer = keras.layers.Dense(FFN_DIM, use_bias=False)
    k3_layer.build((None, EMBED_DIM)) 
    k3_layer.set_weights([initial_weights.T])

    # --- FIX: Use the correct module alias ---
    k3_gptq = k3_gptq_module.GPTQ(k3_layer)
    # --- END FIX ---
    
    k3_quantizer = k3_quant.Quantizer()
    k3_quantizer.configure(bits=4, perchannel=True, sym=sym, groupsize=groupsize)
    k3_gptq.quantizer = k3_quantizer
    
    k3_activations_ops = ops.convert_to_tensor(initial_activations, dtype='float32')
    k3_gptq.add_batch(k3_activations_ops, None)
    
    k3_q = k3_gptq.fasterquant(actorder=actorder, groupsize=groupsize)
    k3_final_q_numpy = ops.convert_to_numpy(k3_q)

    # --- Final Comparison ---
    try:
        np.testing.assert_allclose(pt_final_q_numpy, k3_final_q_numpy, atol=1e-5, rtol=1e-5)
        print(f"✅ Test PASSED for config: {config_str}!")
        return True
    except AssertionError as e:
        print(f"❌ Test FAILED for config: {config_str}!")
        # print(e) 
        return False


if __name__ == "__main__":
    print("Building initial weights and activation data...")
    np.random.seed(0)
    
    initial_weights_np = np.random.randn(FFN_DIM, EMBED_DIM).astype(np.float32)
    initial_activations_np = np.random.rand(NUM_SAMPLES * SEQ_LEN, EMBED_DIM).astype(np.float32)

    test_configs = list(itertools.product(
        [True, False],
        [-1, 128],
        [True, False]
    ))

    all_tests_passed = True
    
    for config_params in test_configs:
        config = {
            "actorder": config_params[0], 
            "groupsize": config_params[1],
            "sym": config_params[2]
        }
        
        passed = build_and_run_test(initial_weights_np.copy(), initial_activations_np.copy(), config)
        if not passed:
            all_tests_passed = False
            
    print("\n" + "="*50)
    if all_tests_passed:
        print("🎉 SUCCESS: All PyTorch vs. Keras 3.0 tests passed!")
    else:
        print("🔥 FAILURE: The Keras 3.0 port does not match the PyTorch reference.")
    print("="*50)
