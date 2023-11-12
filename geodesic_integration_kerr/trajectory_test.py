import numpy as np
from scipy.integrate import solve_ivp
import time
import matplotlib.pyplot as plt
import geodesics

beta_values = np.linspace(np.pi/2, -np.pi/2, 400)
gamma_values = np.linspace(np.pi/2, -np.pi/2, 400)
beta, gamma = np.meshgrid(beta_values, gamma_values)

start_time = time.time()

# в момента е напълно грешно
# може би трябва да се променят интервалите на интегриране в зависимост от достиганите стойности на r
# кривите изглеждат прекалено криви, най-вероятно не смята правилно
# може би самият начин, по който съм дефинирала кривата е грешен, стойностите от scipy.integrate на пръв поглед не изглеждат ужасни
obs = geodesics.Observer([15, np.pi/2, 0], 0.99)
init_q = obs.coord()
# не мога да нацеля прицелен параметър, който да доведе до нестабилната орбита
# това трябва да отиде на ∞
init_p1 = obs.p_init(gamma[19, 8], beta[19, 8])
geo1 = geodesics.Geodesics(init_q, init_p1)
ivp1 = np.array([0, init_q[0], init_q[1], init_q[2], init_p1[1], init_p1[2]])
sol1 = solve_ivp(geo1.hamilton_eqs, [0, -30], ivp1, t_eval=np.linspace(0, -30, 5000))

# това трябва да падне
init_p2 = obs.p_init(gamma[200, 200], beta[200, 200])
geo2 = geodesics.Geodesics(init_q, init_p2)
ivp3 = np.array([0, init_q[0], init_q[1], init_q[2], init_p2[1], init_p2[2]])
sol2 = solve_ivp(geo2.hamilton_eqs, [0, -30], ivp3, t_eval=np.linspace(0, -30, 5000))

init_p3 = obs.p_init(gamma[290, 270], beta[290, 270])
geo3 = geodesics.Geodesics(init_q, init_p3)
ivp3 = np.array([0, init_q[0], init_q[1], init_q[2], init_p3[1], init_p3[2]])
sol3 = solve_ivp(geo3.hamilton_eqs, [0, -30], ivp3, t_eval=np.linspace(0, -30, 5000))

end_time = time.time()
print(end_time - start_time)

r_sol1 = np.asarray(sol1.y[1])
theta_sol1 = np.asarray(sol1.y[2])
phi_sol1 = np.asarray(sol1.y[3])

r_sol2 = np.asarray(sol2.y[1])
theta_sol2 = np.asarray(sol2.y[2])
phi_sol2 = np.asarray(sol2.y[3])

r_sol3 = np.asarray(sol3.y[1])
theta_sol3 = np.asarray(sol3.y[2])
phi_sol3 = np.asarray(sol3.y[3])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

radius = 1.0  # радиус на хоризонта на събитията на черната дупка при а=1
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

photon_r_minus = 2 * (1 + np.cos(2 / 3 * np.arccos(-0.99)))
photon_r_plus = 2 * (1 + np.cos(2 / 3 * np.arccos(0.99)))
print(photon_r_plus)
print(photon_r_minus)

x_curve1 = r_sol1 * np.sin(theta_sol1) * np.cos(phi_sol1)
y_curve1 = r_sol1 * np.sin(theta_sol1) * np.sin(phi_sol1)
z_curve1 = r_sol1 * np.cos(theta_sol1)

x_curve2 = r_sol2 * np.sin(theta_sol2) * np.cos(phi_sol2)
y_curve2 = r_sol2 * np.sin(theta_sol2) * np.sin(phi_sol2)
z_curve2 = r_sol2 * np.cos(theta_sol2)

x_curve3 = r_sol3 * np.sin(theta_sol3) * np.cos(phi_sol3)
y_curve3 = r_sol3 * np.sin(theta_sol3) * np.sin(phi_sol3)
z_curve3 = r_sol3 * np.cos(theta_sol3)

ax.plot(x_curve1, y_curve1, z_curve1, color='r')
ax.plot(x_curve2, y_curve2, z_curve2, color='r')
ax.plot(x_curve3, y_curve3, z_curve3, color='r')
ax.plot_surface(x, y, z, color='b')
ax.view_init(elev=20, azim=30)
ax.set_aspect('equal', adjustable='datalim')
plt.show()