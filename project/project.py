import numpy as np
import cmath

T = 1e-6
F_s = 1e6
omega_0 = 2 * np.pi * 1e5
phi = np.pi / 8
A = 1
N = 513
n_0 = -256
iterations = 100

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

# TODO: Compute all n samples at the same time
def sample(sigma_sqr: float, rng: np.random.Generator) -> np.typing.ArrayLike:
    w_r = rng.normal(loc=0, scale=sigma_sqr, size=N)
    w_i = rng.normal(loc=0, scale=sigma_sqr, size=N)
    x = np.zeros(N, dtype=complex)
    for n in range(n_0, n_0 + N):
        x[n + n_0] = A * np.exp(1j * (omega_0 * n * T + phi)) + w_r[n + n_0] + 1j * w_i[n + n_0]
        print(w_i[n + n_0])
    return x

def main():
    rng = np.random.default_rng()
    # for SNR in SNRs:
    #     print("--------------------------")
    #     for FFT_size in FFT_sizes:
    SNR = get_SNR(60)
    sigma_sqr = get_sigma_sqr(SNR)

    FFT_size = np.pow(2, 20)
    omega_estimates = np.zeros(iterations)
    phi_estimates = np.zeros(iterations)

    for n in range(iterations):
        x_samples = sample(sigma_sqr, rng)
        FFT = np.fft.fft(x_samples, n=FFT_size)
        m_max = np.argmax(np.absolute(FFT))
        print(m_max)
        omega_est = 2 * np.pi * m_max / (FFT_size * T)
        
        phi_est = get_phi_est(omega_est, x_samples)
        omega_estimates[n] = omega_est
        phi_estimates[n] = phi_est
    
    # print("Variance")
    # print(sigma_sqr)

    print("Omega:")

    omega_avg = np.mean(omega_estimates)
    omega_var = np.var(omega_estimates)

    print(omega_var)
    print(omega_CRLB(sigma_sqr))

    print("Phi:")

    phi_avg = np.mean(phi_estimates)
    phi_var = np.var(phi_estimates)

    print(phi_var)
    print(phi_CRLB(sigma_sqr))

    # prev_samples = sample(sigma_sqr, rng)
    # for n in range(iterations):
    #     x_samples = sample(sigma_sqr, rng)
    #     print(x_samples - prev_samples)
    #     prev_samples = x_samples

    # print("--------------------------")

    # loop SNRs
        # loop through FFT sizes
            # loop n = 1000
                # compute omega_fft

            # compute variance omega_fft
            # plot two points: CRLB vs variance omega_fft
        # end up with a plot with one point per FFT size

if __name__ == "__main__":
    main()