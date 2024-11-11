#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 22:26:21 2024

@author: eiriniretsou
"""

import numpy as np
from scipy.integrate import solve_ivp

#Calculating the max torque the rotor would have to apply (based on the dynamic model)
# Constants 
m_exo = 0.045   # Mass of the exoskeleton part attached to the palm (kg)
m_hand = 0.53   # Mass of the palm (kg)
d_exo = 0.046   # Distance from wrist joint to center of mass of exoskeleton (m)
d_hand = 0.056  # Distance from wrist joint to center of mass of hand (m)
g = 9.81        # Gravitational acceleration (m/s^2)
kappa = 0.08     # Spring constant (Nm/rad)
c = 2         # Damping constant (Nm s/rad)

# Calculated constants based on above values
I = m_exo * d_exo**2 + m_hand * d_hand**2   # Moment of inertia (kg m^2)
M = m_exo * d_exo + m_hand * d_hand         # Gravitational term
# Initial conditions
theta_0 = 0          # Initial position (radians)
theta_dot_0 = 0      # Initial angular velocity (rad/s)
initial_conditions = [theta_0, theta_dot_0]

# Define the differential equation without external torque
def equation_of_motion(t, y):
    theta, theta_dot = y
    theta_double_dot = (- kappa * theta - M * g * np.cos(theta) - c * theta_dot) / I
    return [theta_dot, theta_double_dot]

# Time span for simulation
time_span = (0, 20)                # Simulate for 10 seconds
time_eval = np.linspace(0, 10, 1000)  # Increase points for better resolution

# Solve the differential equation
solution = solve_ivp(
    equation_of_motion,
    time_span,
    initial_conditions,
    t_eval=time_eval
)

# Extract theta (position), theta_dot (velocity), theta_double_dot (acceleration)
theta = solution.y[0]
theta_dot = solution.y[1]
time = solution.t

# Calculate angular acceleration (theta_double_dot)
theta_double_dot = np.gradient(theta_dot, time)

# Calculate required motor torque at each time point (without external torque)
torque_required = I * theta_double_dot + kappa * theta + M * g * np.cos(theta) - c * theta_dot

# Find the maximum absolute torque required
max_torque = np.max(np.abs(torque_required))

# Print maximum torque with one decimal place
print(f"The maximum torque required by the motor is: {max_torque:.1f} Nm")