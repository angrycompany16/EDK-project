import numpy as np
import cmath
import matplotlib.pyplot as plt


T = 1e-6
F_s = 1e6
omega_0 = 2 * np.pi * (1e5)
phi = np.pi / 8
A = 1
N = 513
n_0 = -256
iterations = 1000
timeSteps = np.arange(n_0, n_0 + N)

P = N * (N - 1) / 2
Q = N * (N - 1) * (2 * N - 1) / 2

def get_sigma_sqr(SNR: float) -> float:
    return np.pow(A, 2) / (2 * SNR)

def omega_CRLB(sigma_sqr: float) -> float:
    return (12  * sigma_sqr) / (np.pow(A, 2) * np.pow(T, 2) * N * ((np.pow(N, 2) - 1)))

def phi_CRLB(sigma_sqr: float) -> float:
    return (12 * sigma_sqr) * (np.pow(n_0, 2) * N + 2 * n_0 * P + Q) / (np.pow(A, 2) * np.pow(N, 2) * ((np.pow(N, 2) - 1)))

def F(omega: float, x: np.typing.ArrayLike) -> float:
    s = 0
    for n in range(N):
        s += x[n] * np.exp(-1j * omega * n * T)
    return s / N

def get_phi_est(omega_est: float, x: np.typing.ArrayLike) -> float:
    return cmath.phase(np.exp(-1j * omega_est * n_0 * T) * F(omega_est, x))

def get_SNR(snr_dB: float) -> float:
    return np.pow(10, snr_dB / 10)

SNRs = [get_SNR(SNR_dB) for SNR_dB in range (-10, 61, 10)]
FFT_sizes = [np.pow(2, k) for k in range(10, 21, 2)]


def sample(sigma_sqr: float, rng: np.random.Generator) -> np.typing.ArrayLike:
    n_vals = np.arange(n_0, n_0 + N)
    signal = A * np.exp(1j * (omega_0 * n_vals * T + phi))
    noise = rng.normal(0, np.sqrt(sigma_sqr), size=N) + 1j * rng.normal(0, np.sqrt(sigma_sqr), size=N)
    return signal + noise


def main() :
    rng = np.random.default_rng()
    sigma_sqr = get_sigma_sqr(get_SNR(60))
    FFT_size = np.pow(2, 10)
    testFunction = sample(sigma_sqr, rng)
    FFT_of_function = np.fft.fft(testFunction, FFT_size)
    omegaArray = np.arange(0, FFT_size)


    plt.plot(omegaArray, np.abs(FFT_of_function))
    plt.show

    
if __name__ == "__main__":
    main()