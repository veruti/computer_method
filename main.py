import numpy as np

x_min, x_max = 0, 1
t_min, t_max = 0, 1

N, M = 40, 40
h, tau = 1 / N, 1 / M

f = lambda x, t: - x * x * np.sin(t)

u_0, dx_u_0 = lambda x: np.cos(np.pi * x), lambda x: x * x

cray_max = lambda t: 2 * t - 2 * np.sin(t)  + np.cos(np.pi * t)
cray_min = lambda t: 2 * t - np.sin(t)  - np.cos(np.pi * t)

u = lambda x, t: 2*t + (x**2 - 2) * np.sin(t) + np.cos(np.pi * t) * np.cos(np.pi * x)

if __name__ == "__main__":
    print(f"h: {h}")
    print(f"t: {tau}")

    result = np.zeros((M + 1, N + 1))

    for t in range(M + 1):
        result[t, 0] = cray_max(tau * t)
        result[t, N] = cray_min(tau * t)

    for x in range(1, N):
        result[0, x] = u_0(h * x)
        result[1, x] = result[0, x] + tau * dx_u_0(x * h) + tau * tau / 2 * (f(x * h, 0) - np.pi * np.pi * np.cos(np.pi * x * h))

    for t in range(2, M+1):
        for x in range(1, N):
            result[t, x] = 2 * result[t-1, x] - result[t-2, x] + (tau ** 2) / (h ** 2) * (result[t-1, x-1] - 2 * result[t-1, x] + result[t-1, x + 1]) + tau * tau * f(h * x,  tau * (t - 1))

    real_result = np.zeros((M + 1, N + 1))
    for t in range(M + 1):
        for x in range(N + 1):
            real_result[t, x] = u(x*h, t*tau)

    print(result - real_result)
    print(np.abs(result - real_result).max())
    print(np.sqrt(np.square(result - real_result).sum()) * np.square(h * tau))