import numpy as np

a = np.asarray([[0, 0, 0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
                [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]], dtype=float)

c = np.asarray([0, 0.2, 0.3, 0.8, 8 / 9, 1, 1], dtype=float)

b4 = np.asarray([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=float)
b5 = np.asarray([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40], dtype=float)


def kdp45(func, init, h, rng, rtot, num_iter):
    x = np.zeros(shape=num_iter)
    y = np.zeros(shape=num_iter)
    x[0] = init[0]
    y[0] = init[1]

    k = np.zeros(shape=len(init))

    for j in range(num_iter):
        k[0] = h * func(x[0], y[0])
        for i in range(1, 7):
            k[i] = h * func(x[i] + h * c[i], y[i] + a[i] @ k)

        y[j] = y[j-1] + b4 @ k
        y[j] = y[j-1] + b5 @ k

