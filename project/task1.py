import numpy as np
import cmath
import os
import time
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math

T = 1e-6
F_s = 1e6
omega_0 = 2 * np.pi * 1e5
phi = np.pi / 8
A = 1
N = 513
n_0 = -256
iterations = 30


P = N * (N - 1) / 2
Q = N * (N - 1) * (2 * N - 1) / 2

def get_sigma_sqr(SNR: float) -> float :
    return np.pow(A, 2) / (2 * SNR)

def get_sigma(SNR: float) -> float :
    return A / np.sqrt((2 * SNR))

def omega_CRLB(sigma_sqr: float) -> float:
    return (12  * sigma_sqr) / (np.pow(A, 2) * np.pow(T, 2) * N * ((np.pow(N, 2) - 1)))

def phi_CRLB(sigma_sqr: float) -> float:
    return (12 * sigma_sqr) * (np.pow(n_0, 2) * N + 2 * n_0 * P + Q) / (np.pow(A, 2) * np.pow(N, 2) * ((np.pow(N, 2) - 1)))

def F(omega: float, x: np.typing.ArrayLike) -> complex:
    s = 0
    for n in range(N):
        s += x[n] * np.exp(-1j * omega * n * T)
    return s / N

def get_phi_est(omega_est: float, x: np.typing.ArrayLike) -> float:
    return cmath.phase(np.exp(-1j * omega_est * n_0 * T) * F(omega_est, x))

def get_SNR(snr_dB: float) -> float:
    return np.pow(10, snr_dB / 10)


SNRs = [get_SNR(SNR_dB) for SNR_dB in range (-10, 61, 10)]
SNRs_in_dB = [SNR_dB for SNR_dB in range (-10, 61, 10)]
FFT_sizes = [np.pow(2, k) for k in range(10, 21, 2)]
sigma_sqrs = [get_sigma_sqr(SNR) for SNR in SNRs]
omega_CRLBs = [omega_CRLB(sigma_sqr) for sigma_sqr in sigma_sqrs]
phi_CRLBs = [phi_CRLB(sigma_sqr) for sigma_sqr in sigma_sqrs]


def sample(sigma: float, rng: np.random.Generator) -> np.typing.ArrayLike:
    w_r = rng.normal(loc=0, scale=sigma, size=N)
    w_i = rng.normal(loc=0, scale=sigma, size=N)
    x = np.zeros(N, dtype=complex)
    for n in range(n_0, n_0 + N):
        x[n - n_0] = A * np.exp(1j * (omega_0 * n * T + phi)) + w_r[n - n_0] + 1j * w_i[n - n_0]
    return x

def compute_omega_and_phi_estimates(SNR_dB: float, FFT_size: float) :

    rng = np.random.default_rng()
    sigma = get_sigma(get_SNR(SNR_dB))
    omega_estimates = np.zeros(iterations)
    phi_estimates = np.zeros(iterations)

    for n in range(iterations) :
        x_samples = sample(sigma, rng)
        FFT = np.fft.fft(x_samples, FFT_size)
        m_max = np.argmax(np.abs(FFT))

        omega_est = 2 * np.pi * m_max / (FFT_size * T)
        phi_est = get_phi_est(omega_est, x_samples)

        omega_estimates[n] = omega_est
        phi_estimates[n] = phi_est

    return omega_estimates, phi_estimates

def get_omega_variance(SNR_db: float, FFT_size: float) :
    return np.var(compute_omega_and_phi_estimates(SNR_db, FFT_size)[0])

def get_phi_variance(SNR_db: float, FFT_size: float) :
    return np.var(compute_omega_and_phi_estimates(SNR_db, FFT_size)[1])

def get_variance_omega_against_all_snr(FFT_size: float) :
    omega_variances = np.zeros(len(SNRs_in_dB))
    for i in range(len(omega_variances)) :
        omega_variances[i] = get_omega_variance(SNRs_in_dB[i], FFT_size)
    return omega_variances

def get_variance_phi_against_all_snr(FFT_size: float) :
    phi_variances = np.zeros(len(SNRs_in_dB))
    for i in range(len(phi_variances)) :
        phi_variances[i] = get_phi_variance(SNRs_in_dB[i], FFT_size)
    return phi_variances


def plot_variances_omega_against_snr(FFT_size: float):
    omega_variances = get_variance_omega_against_all_snr(FFT_size)

    epsilon = 1e-2
    omega_variancesplot_ = np.clip(omega_variances, epsilon, None)
    omega_CRLBs_plot = np.clip(omega_CRLBs, epsilon, None)
    
    plt.figure(figsize=(10, 6))
    plt.plot(SNRs_in_dB, omega_variancesplot_, marker='o', linestyle='-', label='Omega Variance')
    plt.plot(SNRs_in_dB, omega_CRLBs_plot, marker='s', linestyle='--', label='Omega CRLB')

    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Variance')
    plt.title('Omega Variance and CRLB vs SNR')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_variances_phi_against_snr(FFT_size: float):
    phi_variances = get_variance_phi_against_all_snr(FFT_size)
    
    plt.figure(figsize=(10, 6))
    plt.plot(SNRs_in_dB, phi_variances, marker='o', linestyle='-', label='Phi Variance')
    plt.plot(SNRs_in_dB, phi_CRLBs, marker='s', linestyle='--', label='Phi CRLB')

    plt.yscale('log')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Variance')
    plt.title('Phi Variance and CRLB vs SNR')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    plot_variances_omega_against_snr(FFT_sizes[5])
    #plot_variances_phi_against_snr(FFT_sizes[1])

    

if __name__ == "__main__":
    main()



        



