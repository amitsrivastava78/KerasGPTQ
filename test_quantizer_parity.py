import numpy as np
import tensorflow as tf
import keras
import keras.ops as ops

# Import the two versions of the quantizer logic with aliases to avoid name conflicts
from quant import Quantizer as QuantizerK3
from quant import quantize as quantize_k3
from quantkeras import Quantizer as QuantizerTF
from quantkeras import quantize as quantize_tf

def run_test_case(test_name, input_data, wbits, perchannel, sym, groupsize):
    """
    Runs a single test case comparing the Keras 3 and TensorFlow quantizers.

    Args:
        test_name (str): A descriptive name for the test scenario.
        input_data (np.ndarray): The numpy array to be quantized.
        wbits (int): Number of bits for quantization.
        perchannel (bool): Whether to use per-channel quantization.
        sym (bool): Whether to use symmetric quantization.
        groupsize (int): The size of quantization groups.
    """
    print(f"\n--- Running Test Case: {test_name} ---")
    print(f"Config: wbits={wbits}, perchannel={perchannel}, sym={sym}, groupsize={groupsize}")

    # --- Keras 3 Implementation ---
    quantizer_k3 = QuantizerK3()
    quantizer_k3.configure(wbits=wbits, perchannel=perchannel, sym=sym, groupsize=groupsize)
    
    # Keras backend expects Keras tensors
    input_k3 = ops.convert_to_tensor(input_data, dtype='float32')
    quantizer_k3.find_params(input_k3, weight=True)
    
    # Get scale and zero parameters
    scale_k3 = quantizer_k3.scale
    zero_k3 = quantizer_k3.zero
    
    # Perform quantization
    output_k3 = quantize_k3(input_k3, scale_k3, zero_k3, quantizer_k3.maxq)

    # --- TensorFlow Implementation ---
    quantizer_tf = QuantizerTF()
    quantizer_tf.configure(wbits=wbits, perchannel=perchannel, sym=sym, groupsize=groupsize)

    # TensorFlow backend expects TF tensors
    input_tf = tf.convert_to_tensor(input_data, dtype=tf.float32)
    quantizer_tf.find_params(input_tf, weight=True)
    
    # Get scale and zero parameters
    scale_tf = quantizer_tf.scale
    zero_tf = quantizer_tf.zero
    
    # Perform quantization
    output_tf = quantize_tf(input_tf, scale_tf, zero_tf, quantizer_tf.maxq)

    # --- Comparison ---
    # Convert all results to numpy for a consistent comparison
    scale_k3_np, zero_k3_np, output_k3_np = ops.convert_to_numpy(scale_k3), ops.convert_to_numpy(zero_k3), ops.convert_to_numpy(output_k3)
    scale_tf_np, zero_tf_np, output_tf_np = scale_tf.numpy(), zero_tf.numpy(), output_tf.numpy()
    
    # Check for near-equality to handle minor floating point differences
    try:
        np.testing.assert_allclose(scale_k3_np, scale_tf_np, rtol=1e-5, atol=1e-5, err_msg="Scale parameters do not match!")
        print("✅ Scale: PASSED")
    except AssertionError as e:
        print("❌ Scale: FAILED")
        print(e)

    try:
        np.testing.assert_allclose(zero_k3_np, zero_tf_np, rtol=1e-5, atol=1e-5, err_msg="Zero parameters do not match!")
        print("✅ Zero: PASSED")
    except AssertionError as e:
        print("❌ Zero: FAILED")
        print(e)
        
    try:
        np.testing.assert_allclose(output_k3_np, output_tf_np, rtol=1e-5, atol=1e-5, err_msg="Final quantized outputs do not match!")
        print("✅ Quantized Output: PASSED")
    except AssertionError as e:
        print("❌ Quantized Output: FAILED")
        print(e)

if __name__ == '__main__':
    # Generate a consistent random input tensor for testing
    np.random.seed(0)
    test_data = np.random.randn(128, 512).astype(np.float32)
    test_data[0, :64] = 0 # Include a row of zeros to test edge cases

    # --- Test Suite ---
    
    # Scenario 1: Symmetric, Per-channel, No groups
    run_test_case(
        "Symmetric, Per-channel",
        input_data=test_data, wbits=4, perchannel=True, sym=True, groupsize=-1
    )
    
    # Scenario 2: Asymmetric, Per-channel, No groups
    run_test_case(
        "Asymmetric, Per-channel",
        input_data=test_data, wbits=4, perchannel=True, sym=False, groupsize=-1
    )

    # Scenario 3: Symmetric, Per-channel, With groups
    run_test_case(
        "Symmetric, Per-channel, Grouped",
        input_data=test_data, wbits=4, perchannel=True, sym=True, groupsize=128
    )

    # Scenario 4: Asymmetric, Per-channel, With groups
    run_test_case(
        "Asymmetric, Per-channel, Grouped",
        input_data=test_data, wbits=4, perchannel=True, sym=False, groupsize=128
    )

    # Scenario 5: Symmetric, Per-tensor
    run_test_case(
        "Symmetric, Per-tensor",
        input_data=test_data, wbits=4, perchannel=False, sym=True, groupsize=-1
    )

    # Scenario 6: Asymmetric, Per-tensor
    run_test_case(
        "Asymmetric, Per-tensor",
        input_data=test_data, wbits=4, perchannel=False, sym=False, groupsize=-1
    )

