import numpy as np
import torch
import keras
import keras.ops as ops
import pquant
import quant as k3_quant # Alias for Keras version

def run_find_params_test(input_data_np, bits, groupsize, test_name="Default Test"):
    """
    Compares scale and zero point outputs of find_params for PyTorch and Keras quantizers.
    Conditions: weight=True, sym=False, mse=False, perchannel=True
    """
    print(f"\n--- Running Test: {test_name} (bits={bits}, groupsize={groupsize}) ---")

    # Ensure input data is float32 for consistency with typical model weights
    input_data_np = input_data_np.astype(np.float32)

    # --- PyTorch Quantizer Setup ---
    pt_quantizer = pquant.Quantizer()
    pt_quantizer.configure(
        bits=bits, 
        perchannel=True, 
        sym=False, 
        mse=False, 
    )
    
    # Input data for PyTorch (must be torch.Tensor)
    # PyTorch's find_params expects weight input for perchannel to be 2D, e.g., (output_features, input_features) or (output_features, groupsize)
    # The pquant.py's find_params internally flattens `x` with `x.flatten(1)` which operates on dim 1 and onwards.
    # So if input_data_np is (OUT_DIM, IN_DIM_OR_GROUPSIZE), it stays that way.
    pt_input_tensor = torch.from_numpy(input_data_np)
    
    # Run PyTorch find_params
    pt_quantizer.find_params(pt_input_tensor, weight=True)
    pt_scale = pt_quantizer.scale.numpy()
    pt_zero = pt_quantizer.zero.numpy()
    print(f"PyTorch Scale Shape: {pt_scale.shape}, Zero Shape: {pt_zero.shape}")
    print(f"PyTorch Scale Sum: {np.sum(pt_scale):.6f}, Zero Sum: {np.sum(pt_zero):.6f}")

    # --- Keras Quantizer Setup ---
    k3_quantizer = k3_quant.Quantizer()
    k3_quantizer.configure(
        bits=bits, 
        perchannel=True, 
        sym=False, 
        mse=False, 
    )
    
    # Input data for Keras (must be Keras Tensor/NumPy array)
    # Keras ops work directly with NumPy arrays or Keras Tensors.
    k3_input_tensor = ops.convert_to_tensor(input_data_np) 
    
    # Run Keras find_params
    k3_quantizer.find_params(k3_input_tensor, weight=True)
    k3_scale = ops.convert_to_numpy(k3_quantizer.scale)
    k3_zero = ops.convert_to_numpy(k3_quantizer.zero)
    print(f"Keras Scale Shape: {k3_scale.shape}, Zero Shape: {k3_zero.shape}")
    print(f"Keras Scale Sum: {np.sum(k3_scale):.6f}, Zero Sum: {np.sum(k3_zero):.6f}")

    # --- Comparison ---
    try:
        # Use a slightly relaxed tolerance for initial testing, as differences are expected
        # You might need to adjust these further based on the output.
        np.testing.assert_allclose(pt_scale, k3_scale, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(pt_zero, k3_zero, atol=1e-5, rtol=1e-5)
        print(f"✅ PASSED: Scale and Zero points match for {test_name}")
        return True
    except AssertionError as e:
        print(f"❌ FAILED: Scale or Zero points do NOT match for {test_name}")
        print(e)
        return False

if __name__ == "__main__":
    np.random.seed(42) # Consistent seed for reproducible data

    # Test Case 1: Simple 2D weight matrix (similar to a single Dense layer without groupsize in effect)
    # Corresponds to a matrix of (output_features, input_features)
    weight_matrix_1 = np.random.rand(512, 128) * 10 - 5 # Values between -5 and 5
    
    # Test Case 2: 2D data that would represent a "group" slice from GPTQ
    # (output_features, groupsize)
    weight_matrix_2 = np.random.rand(512, 128) * 10 - 5 

    # Test Case 3: Small matrix with zeros and edge values
    weight_matrix_3 = np.array([
        [0.0, 1.0, 2.0],
        [-1.0, 0.0, 1.0],
        [-2.0, -1.0, 0.0]
    ], dtype=np.float32)

    # Test Case 4: With negative values where symmetric logic might apply differently if sym was True
    weight_matrix_4 = np.array([
        [-10.0, -5.0, 0.0, 5.0, 10.0],
        [-2.0, -1.0, 0.0, 1.0, 2.0]
    ], dtype=np.float32)

    all_passed = True

    # Run tests for different bits and groupsize configurations
    # Note: groupsize in configure only affects find_params if weight=True and perchannel=True
    # and the find_params logic explicitly uses it for reshaping or min/max.
    # The actual 'x' passed to find_params should already be sliced by groupsize.

    # Test with groupsize=-1 (per-output-feature/per-channel for the full weight dimension)
    # if not run_find_params_test(weight_matrix_1.copy(), bits=4, groupsize=-1, test_name="Weight Matrix 1, bits=4, groupsize=-1"): all_passed = False
    # if not run_find_params_test(weight_matrix_3.copy(), bits=8, groupsize=-1, test_name="Weight Matrix 3 (Edge Values), bits=8, groupsize=-1"): all_passed = False
    
    # Test with groupsize=128 (meaning the input_data_np *is* already a slice for a group)
    # The find_params code should treat it as a per-group-channel quantization.
    if not run_find_params_test(weight_matrix_2.copy(), bits=4, groupsize=128, test_name="Weight Matrix 2 (simulated group slice), bits=4, groupsize=128"): all_passed = False
    
    # Test with more diverse range
    # if not run_find_params_test(weight_matrix_4.copy(), bits=4, groupsize=-1, test_name="Weight Matrix 4 (Range), bits=4, groupsize=-1"): all_passed = False


    if all_passed:
        print("\n🎉 All `find_params` tests PASSED!")
    else:
        print("\n🔥 Some `find_params` tests FAILED. Investigate the numerical differences.")
