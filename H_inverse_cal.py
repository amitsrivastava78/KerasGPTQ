import os
# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
import keras.ops as ops
import numpy as np

def get_h_inv_tensorflow(H_numpy):
    """
    Calculates H_inv using the pure TensorFlow method, mimicking gptqkeras_fixed.py.
    """
    H = tf.convert_to_tensor(H_numpy, dtype=tf.float32)
    
    # Regularize H to ensure it's numerically stable for Cholesky decomposition.
    epsilon = 1e-6 
    H_stable = H + tf.eye(H.shape[0], dtype=H.dtype) * epsilon

    try:
        L = tf.linalg.cholesky(H_stable)
        H_inv = tf.linalg.cholesky_solve(L, tf.eye(H.shape[0], dtype=tf.float32))
        return H_inv
    except tf.errors.InvalidArgumentError:
        print("[TF Fallback] Cholesky failed, using pseudo-inverse.")
        return tf.linalg.pinv(H)

def get_h_inv_keras_ops(H_numpy):
    """
    Calculates H_inv using the Keras 3.0 ops API.
    This version uses ops.linalg.solve, which is the standard Keras 3 way
    to solve a system of linear equations and find an inverse.
    """
    H = ops.convert_to_tensor(H_numpy, dtype='float32')

    # Regularize H for numerical stability.
    epsilon = 1e-6
    H_stable = H + ops.eye(H.shape[0], dtype=H.dtype) * epsilon
    
    try:
        # The standard way to find an inverse is to solve H * X = I for X
        I = ops.eye(H.shape[0], dtype='float32')
        H_inv = ops.linalg.solve(H_stable, I)
        return H_inv
    except Exception as e:
        # This fallback is unlikely but included for safety.
        # We drop to TensorFlow for pinv as it's not in keras.ops
        print(f"[Keras Fallback] Solve failed ({e}), using TF pseudo-inverse.")
        H_tf = tf.convert_to_tensor(H_numpy, dtype=tf.float32)
        H_inv_tf = tf.linalg.pinv(H_tf)
        return ops.convert_to_tensor(H_inv_tf.numpy())


def main():
    """
    Main function to run the comparison test.
    """
    print("--- H_inv Calculation Comparison: TensorFlow vs. Keras 3.0 ---")
    
    # 1. Create a sample positive semi-definite matrix (like a real Hessian)
    matrix_size = 128
    np.random.seed(0)
    # Create a random matrix A, then compute A.T * A to ensure it's positive semi-definite
    A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
    H_numpy = np.dot(A.T, A)
    
    print(f"\nGenerated a {matrix_size}x{matrix_size} positive semi-definite matrix H.")

    # 2. Calculate H_inv using both methods
    h_inv_tf = get_h_inv_tensorflow(H_numpy)
    h_inv_keras = get_h_inv_keras_ops(H_numpy)
    
    # 3. Calculate the scalar sum of all elements in each result
    sum_tf = tf.reduce_sum(h_inv_tf).numpy()
    sum_keras = ops.sum(h_inv_keras).numpy() # Use .numpy() to extract scalar
    
    # 4. Print results and the difference
    print("\n--- Results ---")
    print(f"Sum of H_inv elements (TensorFlow): {sum_tf}")
    print(f"Sum of H_inv elements (Keras ops):  {sum_keras}")
    
    difference = abs(sum_tf - sum_keras)
    print(f"\nAbsolute Difference:                {difference}")
    
    if difference > 1e-5:
        print("\nConclusion: The results are DIFFERENT.")
        print("This confirms that the underlying implementations produce numerically different outputs,")
        print("which will lead to cumulative error in iterative algorithms like GPTQ.")
    else:
        print("\nConclusion: The results are effectively the same.")

if __name__ == "__main__":
    main()