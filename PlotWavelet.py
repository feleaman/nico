import numpy as np
import pywt
import matplotlib.pyplot as plt

n = 1000
x = np.ones(n)
wav = pywt.dwt(x, 'db3')
print(wav)

plt.plot(wav[1])
plt.show()