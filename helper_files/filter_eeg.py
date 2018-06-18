import numpy as np

def filt(raw_signal, fL=1, fH=80, b=.08):
    # print("Hello")
    sampling_rate = 512  # Hz
    fL = fL / sampling_rate
    fH = fH / sampling_rate
    b = b
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.

    n = np.arange(N)

    # low-pass filter
    hlpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)

    # high-pass filter
    hhpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((N - 1) / 2)] += 1

    h = np.convolve(hlpf, hhpf)
    s = list(raw_signal)
    filtered =  np.convolve(s, h)
    # print(filtered)
    return filtered

# a = np.random.random_integers(0,high=1,size=(102,))
# print(a.size,filter(a).size)



