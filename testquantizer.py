import numpy as np
import keras
import tensorflow as tf
import itertools

# Import both original TF and migrated Keras 3 versions
# Use aliases to handle the file names
import quant as k3_quantkeras
import quantkeras as tf_quantkeras

# --- Model Dimensions (Matching opt-125m) ---
EMBED_DIM = 768
FFN_DIM = 3072

def run_single_quantizer_test(weights, config):
    """
    Tests that the Quantizer class produces the same scale and zero values
    for a given configuration.
    """
    groupsize = config["groupsize"]
    sym = config["sym"]
    mse = config["mse"]
    
    # Create a detailed config string for logging
    config_str = f"groupsize={groupsize}, sym={sym}, mse={mse}"
    print(f"\n--- Testing Quantizer with config: {config_str} ---")
    
    # --- TensorFlow Version ---
    tf_quantizer = tf_quantkeras.Quantizer()
    tf_quantizer.configure(bits=4, perchannel=True, sym=sym, groupsize=groupsize, mse=mse)
    tf_quantizer.find_params(tf.convert_to_tensor(weights, dtype=tf.float32), weight=True)
    
    tf_scale = tf_quantizer.scale.numpy()
    tf_zero = tf_quantizer.zero.numpy()

    # --- Keras 3 Version ---
    k3_quantizer = k3_quantkeras.Quantizer()
    k3_quantizer.configure(bits=4, perchannel=True, sym=sym, groupsize=groupsize, mse=mse)
    k3_quantizer.find_params(keras.ops.convert_to_tensor(weights, dtype='float32'), weight=True)
    
    k3_scale = keras.ops.convert_to_numpy(k3_quantizer.scale)
    k3_zero = keras.ops.convert_to_numpy(k3_quantizer.zero)

    # --- Compare final quantized weights ---
    try:
        np.testing.assert_allclose(tf_scale, k3_scale, atol=1e-6, rtol=1e-6, err_msg="Scale values are not equivalent")
        np.testing.assert_allclose(tf_zero, k3_zero, atol=1e-6, rtol=1e-6, err_msg="Zero values are not equivalent")
        print(f"✅ Test PASSED for config: {config_str}!")
        return True
    except AssertionError as e:
        print(f"❌ Test FAILED for config: {config_str}!")
        print(e)
        return False

if __name__ == "__main__":
    # 1. Build a mock weight tensor
    print("Building mock weight tensor...")
    # Use a fixed seed for reproducible test runs
    np.random.seed(42)
    mock_weights = np.random.randn(EMBED_DIM, FFN_DIM).astype(np.float32)

    # 2. Define a comprehensive set of test configurations
    test_configs = list(itertools.product(
        [-1, 128],      # groupsize
        [True, False],  # sym
        [False, True]   # mse (adding the untested path)
    ))

    all_tests_passed = True
    
    # 3. Run tests for all configurations
    for config_params in test_configs:
        config = {
            "groupsize": config_params[0], 
            "sym": config_params[1],
            "mse": config_params[2]
        }
        passed = run_single_quantizer_test(mock_weights, config)
        if not passed:
            all_tests_passed = False

    print("\n" + "="*40)
    if all_tests_passed:
        print("🎉 SUCCESS: All Quantizer class tests passed!")
    else:
        print("🔥 FAILURE: A bug was found in the Quantizer class. Please review the failed test.")
    print("="*40)
