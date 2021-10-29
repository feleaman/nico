from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from sys import exit

# fig, ax = plt.subplots()
# x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
# ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')
# print(norm.cdf(1) - norm.cdf(-1))
# plt.show()


mu = 3.
std = 0.1

fig, ax = plt.subplots()
x = np.linspace(-5, 5, 5000)
y = norm.pdf(x, mu, std)
ax.plot(x, y)
print(norm.cdf(1, mu, std) - norm.cdf(-1, mu, std))
plt.show()

