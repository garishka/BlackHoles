from sympy import symbols, Function, sin, diff, exp

r, theta, gamma = symbols("r theta gamma")
# A = (1 - 2 / r / gamma) ** gamma
# B = ((r ** 2 - 2 * r / gamma) / (r ** 2 - 2 * r / gamma + (sin(theta) / gamma) ** 2)) ** (gamma ** 2 - 1)
# C = (r ** 2 - 2 * r / gamma) ** (gamma ** 2) / (r ** 2 - 2 * r / gamma + (sin(theta) / gamma) ** 2) ** (gamma ** 2 - 1)
#
# g00 = - A
# g11 = B / A
# g22 = C / A
# g33 = (r ** 2 - 2 * r / gamma) * sin(theta) ** 2 / A
#
# dg00_dr = diff(g00, r)
# dg11_dr = diff(g11, r)
# dg22_dr = diff(g22, r)
# dg33_dr = diff(g33, r)
#
# dg00_dth = diff(g00, theta)
# dg11_dth = diff(g11, theta)
# dg22_dth = diff(g22, theta)
# dg33_dth = diff(g33, theta)
#
# print(dg33_dr)

# calculations for Gamma Infinity (Chazy-Curzon solution)
a = exp(- 2 / r)
b = exp(- sin(theta) ** 2 / r ** 2)

gi00 = - a
gi11 = b / a
gi22 = r ** 2 / a
gi33 = (r * sin(theta)) ** 2 / a

dgi00_dr = diff(gi00, r)
dgi11_dr = diff(gi11, r)
dgi22_dr = diff(gi22, r)
dgi33_dr = diff(gi33, r)

dgi00_dth = diff(gi00, theta)
dgi11_dth = diff(gi11, theta)
dgi22_dth = diff(gi22, theta)
dgi33_dth = diff(gi33, theta)

print(dgi33_dth)
