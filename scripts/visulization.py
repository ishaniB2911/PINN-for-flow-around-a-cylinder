import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('../build/flow_field_final.csv')

# Reshape for plotting
nx, ny = 60, 30
x = df['x'].values.reshape(ny, nx)
y = df['y'].values.reshape(ny, nx)
speed = df['speed'].values.reshape(ny, nx)
u = df['u'].values.reshape(ny, nx)
v = df['v'].values.reshape(ny, nx)

# Plot velocity magnitude
plt.figure(figsize=(12, 6))
plt.contourf(x, y, speed, levels=20, cmap='jet')
plt.colorbar(label='Velocity Magnitude')
plt.quiver(x[::3, ::3], y[::3, ::3], u[::3, ::3], v[::3, ::3], color='white')
circle = plt.Circle((0, 0), 0.5, color='black', fill=True)
plt.gca().add_patch(circle)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Flow Around Cylinder - PINN Solution')
plt.axis('equal')
plt.tight_layout()
plt.savefig('../results/flow_field.png', dpi=150)
plt.show()