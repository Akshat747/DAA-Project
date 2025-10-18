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
 np.set_printoptions(threshold=np.inf)
 with open("example.txt", "w") as f:
   
        sizes = [2**i for i in range(2, 11)]  # 4x4 to 1024x1024
        T_strassen = []
        T_numpy = []
        K=1
    # Testing for correctness and difference in run time
        for n in sizes:
            A = np.random.randint(0, 10, (n, n))
            B = np.random.randint(0, 10, (n, n))
            f.write(f"A{(K)} = {A}\n")
            f.write(f"B{K} = {B}\n")
            start = time.perf_counter()
            C = strassen(A, B, cutoff=64)
            f.write(f"A{K} x B{K} = {C}\n")
            end = time.perf_counter()
            T_strassen.append(end - start)

            start1 = time.perf_counter()
            D = standard_mult(A, B)
            end1 = time.perf_counter()
            T_numpy.append(end1 - start1)
            K=K+1

 with open("runtime_table.txt", "w") as f:
     # Write header
     f.write(f"{'Matrix size (n)':>12} {'T_strassen (s)':>15} {'T_numpy (s)':>15}\n")
     f.write("="*44 + "\n")  
     for n, t_s, t_n in zip(sizes, T_strassen, T_numpy):
         f.write(f"{n:12d} {t_s:15.6f} {t_n:15.6f}\n")

        
        

 alpha_theory = np.log2(7)  # ~2.807
 c = np.mean(T_strassen / (sizes**alpha_theory))  # scale constant
 sizes_smooth = np.linspace(sizes[0], sizes[-1], 500)
 T_theory = c * sizes_smooth**alpha_theory
 
 
 log_sizes = np.log(sizes)
 log_T = np.log(T_strassen)
 coeffs = np.polyfit(log_sizes, log_T, 1)  # slope = exponent, intercept = log(c)
 alpha_empirical, logc_empirical = coeffs
 T_bestfit = np.exp(logc_empirical) * sizes_smooth**alpha_empirical
 
 
 plt.figure(figsize=(8,6))
 
 plt.plot(sizes, T_strassen, 'o-', label="Measured runtime")
 plt.plot(sizes_smooth, T_theory, '--', label=r"Theoretical $O(n^{\log_2 7})$")
 plt.plot(sizes_smooth, T_bestfit, '-.', label=rf"Best fit: $O(n^{{{alpha_empirical:.3f}}})$")
 
 plt.xlabel("Matrix size n")
 plt.ylabel("Runtime (seconds)")
 plt.title("Strassen Runtime: Measured vs Theoretical & Best Fit")
 plt.legend()
 plt.grid(True)
 plt.show()
 
 # Generate float matrices for numerical accuracy test
 num_Acc= []
 for i in range (1,100,1):
        A = np.random.rand(i,i) * 10
        B = np.random.rand(i,i) * 10    

        # Compute results
        stan = standard_mult(A, B)
        strass = strassen(A, B, cutoff=64)

        # Measure error (sum of squares)
        error = stan - strass
    
        err = np.sum(error**2)
        num_Acc.append(err) 
 ind=np.arange(1,100)
 with open("numerical_accuracy_table.txt", "w") as f:     
     f.write(f"{'Matrix size (n)':>12} {'Numerical Accuracy':>20}\n")
     f.write("="*32 + "\n")   
     for n, acc in zip(ind, num_Acc):
         f.write(f"{n:12d} {acc:20.6e}\n") 
        
        

       