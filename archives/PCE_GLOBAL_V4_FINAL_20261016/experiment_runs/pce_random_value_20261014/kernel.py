import numpy as np
from numba import njit
@njit(cache=True)
def expectations(theta, fams, masks):
    psi = np.zeros(256, np.complex128)
    psi[0] = 1
    for layer in range(4):
        for q in range(8):
            bit = 1 << 7 - q
            j = 2 * (layer * 8 + q)
            (c, s) = (np.cos(theta[j] / 2), np.sin(theta[j] / 2))
            z0 = np.exp(-0.5j * theta[j + 1])
            z1 = np.exp(0.5j * theta[j + 1])
            for base in range(0, 256, 2 * bit):
                for off in range(bit):
                    a = base + off
                    b = a + bit
                    (x, y) = (psi[a], psi[b])
                    psi[a] = z0 * (c * x - s * y)
                    psi[b] = z1 * (s * x + c * y)
        for q in range(layer % 2, 7, 2):
            a = 1 << 7 - q
            b = 1 << 6 - q
            for j in range(256):
                if j & a and j & b:
                    psi[j] = -psi[j]
    out = np.zeros(len(masks))
    for fam in range(3):
        phi = psi.copy()
        if fam:
            for q in range(8):
                bit = 1 << 7 - q
                for base in range(0, 256, 2 * bit):
                    for off in range(bit):
                        a = base + off
                        b = a + bit
                        (x, y) = (phi[a], phi[b])
                        if fam == 2:
                            y = -1j * y
                        phi[a] = (x + y) / np.sqrt(2)
                        phi[b] = (x - y) / np.sqrt(2)
        parity = np.abs(phi) ** 2
        stride = 1
        while stride < 256:
            for base in range(0, 256, 2 * stride):
                for off in range(stride):
                    a = base + off
                    b = a + stride
                    (x, y) = (parity[a], parity[b])
                    parity[a] = x + y
                    parity[b] = x - y
            stride *= 2
        for j in range(len(masks)):
            if fams[j] == fam:
                out[j] = parity[masks[j]]
    return out
