import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0, 2 * np.pi, 1000)
k = 6  # Number of petals
r = np.cos(k * theta)  # Radial coordinate

plt.figure(figsize=(6, 6))
plt.polar(theta, r)
plt.title("Rose Curve")
plt.show()

theta = np.linspace(0, 2 * np.pi, 1000)
a = 1  # Constant
r = np.sqrt(a**2 * np.cos(2 * theta))  # Radial coordinate

plt.figure(figsize=(6, 6))
plt.polar(theta, r)
plt.title("Lemniscate of Bernoulli")
plt.show()

theta = np.linspace(0, 2 * np.pi, 1000)
a = 1  # Constant
r = a * (1 - np.cos(theta))  # Radial coordinate

plt.figure(figsize=(6, 6))
plt.polar(theta, r)
plt.title("Cardioid")
plt.show()

theta = np.linspace(0, 3 * np.pi, 1000)
a = 0.2  # Growth factor
r = np.exp(a * theta)  # Radial coordinate

plt.figure(figsize=(6, 6))
plt.polar(theta, r)
plt.title("Logarithmic Spiral")
plt.show()