# Physics-Informed Neural Network (PINN) for Fluid Flow Around a Cylinder

> **Applied AI + Computational Fluid Dynamics in Modern C++**
> A high performance implementation of a Physics-Informed Neural Network (PINN) solving Navier-Stokes equations for a stead 2D flow around a cylinder-without traditional CFD Solvers.
>
> ---
## Why This Project Stands Out

- **Bridges AI and Physics:** Demonstrates advanced use of neural networks to learn governing PDEs.
- **High-Performance C++:** Implemented from scratch—no TensorFlow/PyTorch—showing strong algorithmic and optimization skills.
- **Real-World Relevance:** PINNs are cutting-edge for aerospace, energy, and scientific computing.

---

## Tech Stack & Keywords

- **Languages:** C++, CMake  
- **Concepts:** Neural Networks, PDE Solvers, Computational Fluid Dynamics (CFD), Physics-Informed Machine Learning  
- **Skills:** Numerical Methods, Gradient Approximation, Modular Design, Performance Optimization  
- **Tools:** Git, CMake, Python (for visualization)  

---

## Core Features

- **Custom Neural Network Engine:**  
  - Architecture: `2 → 32 → 32 → 32 → 3` (outputs: `u, v, p`)
  - Activations: `tanh` (hidden), `linear` (output)
- **Physics Loss:**  
  - Enforces **continuity** and **momentum** equations for incompressible flow
- **Boundary Conditions:**  
  - Inlet velocity, cylinder no-slip, top/bottom walls
- **Training:**  
  - Random collocation sampling
  - Gradient updates via finite-difference approximations
- **Output:**  
  - CSV flow fields for visualization in Python/MATLAB

---
## Build & Run

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
./pinn_flow_cylinder


