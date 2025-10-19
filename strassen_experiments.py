import numpy as np
import time
import matplotlib.pyplot as plt



def add_matrix(A, B):
    return A + B

def sub_matrix(A, B):
    return A - B

def standard_mult(A, B):
    return A @ B  # Use NumPy standard multiplication

def next_power_of_two(n):

    return 1 if n == 0 else 2**(n - 1).bit_length()

def strassen(A, B, cutoff=64):
    """
    Multiply two matrices using Strassen's algorithm.
    Automatically pads matrices if not of size 2^n.
    cutoff: use standard multiplication for matrices smaller than this size
    """
   
    n = A.shape[0]
    m = next_power_of_two(n)  # size after padding if needed

    # Pad matrices to size m x m
    if m != n:
        A_pad = np.zeros((m, m), dtype=A.dtype)
        B_pad = np.zeros((m, m), dtype=B.dtype)
        A_pad[:n, :n] = A
        B_pad[:n, :n] = B
    else:
        A_pad, B_pad = A, B

    # Base case
    if m <= cutoff:
        return standard_mult(A_pad, B_pad)[:n, :n]

    # Split matrices into quadrants
    k = m // 2
    A11, A12 = A_pad[:k, :k], A_pad[:k, k:]
    A21, A22 = A_pad[k:, :k], A_pad[k:, k:]
    B11, B12 = B_pad[:k, :k], B_pad[:k, k:]
    B21, B22 = B_pad[k:, :k], B_pad[k:, k:]

    # Compute M1 to M7
    M1 = strassen(add_matrix(A11, A22), add_matrix(B11, B22), cutoff)
    M2 = strassen(add_matrix(A21, A22), B11, cutoff)
    M3 = strassen(A11, sub_matrix(B12, B22), cutoff)
    M4 = strassen(A22, sub_matrix(B21, B11), cutoff)
    M5 = strassen(add_matrix(A11, A12), B22, cutoff)
    M6 = strassen(sub_matrix(A21, A11), add_matrix(B11, B12), cutoff)
    M7 = strassen(sub_matrix(A12, A22), add_matrix(B21, B22), cutoff)

    # Combine results
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Stack quadrants
    top = np.hstack((C11, C12))
    bottom = np.hstack((C21, C22))
    C_pad = np.vstack((top, bottom))

    # Remove padding
    return C_pad[:n, :n]

# Example usage
if __name__ == "__main__":
 # Input the size of the matrices
 n = int(input("Enter the size n of the matrices: "))
 
 # Input matrix A
 print("Enter elements of matrix A (row by row):")
 A = np.array([list(map(float, input().split())) for _ in range(n)])
 
 # Validate dimensions
 if A.shape != (n, n):
     raise ValueError("Matrix A must be of size n x n.")
 
 # Input matrix B
 print("Enter elements of matrix B (row by row):")
 B = np.array([list(map(float, input().split())) for _ in range(n)])
 
 # Validate dimensions
 if B.shape != (n, n):
     raise ValueError("Matrix B must be of size n x n.")
 print("\nMatrix A:\n", A)
 print("\nMatrix B:\n", B)
 print("\nMatrix A x B:\n", strassen(A,B))