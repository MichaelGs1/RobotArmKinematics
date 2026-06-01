# 🤖 Kinematics Package

A high-performance Python package for computing **forward and inverse kinematics** of N-DOF robotic manipulators. Built with speed in mind using Numba JIT compilation.

## 📋 Overview

This package provides tools for:
- **Forward Kinematics (FK)**: Compute end-effector pose from joint angles
- **Inverse Kinematics (IK)**: Find joint angles from desired end-effector pose
- **Jacobian Computation**: Calculate kinematic relationships for velocity/force transformation
- **Gravity Compensation**: Compute required torques to compensate gravitational effects
- **Trajectory Generation**: Create linear and circular motion paths
- **Visualization**: 3D plotting of robot configurations and workspaces

## ⚡ Features

- ✅ Fast JIT-compiled core algorithms using Numba
- ✅ Support for multiple robot configurations (UR10, UR20, Doosan M0609, Kuka IIwa, ...)
- ✅ Denavit-Hartenberg (Khalil) parameterization
- ✅ IK with both position and orientation constraints
- ✅ Dynamic analysis with gravity effects
- ✅ Manipulability and force analysis with ellipsoid visualization

## 📦 Installation

```bash
pip install .
```

## 🚀 Quick Start

```python
from kinematics.config import UR10Config
from kinematics.core import fk, ik
import numpy as np

# Load robot configuration
config = UR10Config()

# Forward kinematics: compute end-effector pose
q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])  # joint angles (rad)
T = fk(q, config.parameter_a, config.parameter_d, 
        config.parameter_alpha, config.parameter_theta, config.parameter_tcp)

print("End-effector pose:\n", T)

# Inverse kinematics: find joint angles for desired pose
target_pose = T.copy()
success, q_solution = ik(target_pose, q, 
                         config.parameter_a, config.parameter_d,
                         config.parameter_alpha, config.parameter_theta, 
                         config.parameter_tcp, config.parameter_qmin, 
                         config.parameter_qmax)

print(f"IK Success: {success}")
print(f"Joint angles (rad): {q_solution}")
```

## 🔧 Denavit-Hartenberg (Khalil) Convention

This package uses the **modified Denavit-Hartenberg (Khalil)** convention for kinematic parameterization.

### DH Parameters

Each joint is described by 4 parameters that define the transformation from one frame to the next:

```
┌─────────────────────────────────────────────────────────────────┐
│  Parameter  │  Symbol  │      Description                       │
├─────────────────────────────────────────────────────────────────┤
│  Distance   │    a     │  Translation along X-axis (m)          │
│  Offset     │    d     │  Translation along Z-axis (m)          │
│  Twist      │  alpha   │  Rotation around X-axis (rad)          │
│  Angle      │  theta   │  Rotation around Z-axis (rad)          │
│             │          │  (includes joint variable q)           │
└─────────────────────────────────────────────────────────────────┘

```

Homogeneous Transformation (Khalil Convention):
$$T_i^{i-1} = Rot(x_{i-1}, α_i) × Trans(x_{i-1}, a_i) × Rot(z_i, θ_i) × Trans(z_{i-1}, d_i)$$


### Transformation Matrix Visualization

```
Frame i-1              Frame i
    │                    │
    │                    │
    └─────[Joint i]──────┘
         (actuated by q)
    
    Khalil DH Matrix:
    ┌                                             ┐
    │ cos(θ)         -sin(θ)       0        a     │
    │ sin(θ)cos(α) cos(θ)cos(α) -sin(α) -d·sin(α) │
    │ sin(θ)sin(α) cos(θ)sin(α)  cos(α)  d·cos(α) │
    │    0              0          0        1     │
    └                                             ┘
    
    Where:
    - a = translation along X-axis
    - d = translation along Z-axis
    - α = rotation around X-axis
    - θ = rotation around Z-axis (includes joint q)
```

## 📚 Module Structure

```
src/kinematics/
├── core/
│   ├── core.py              # FK, IK, Jacobian computations
│   └── trajectory.py        # Trajectory generation
├── config/
│   ├── config.py            # Base configuration class
│   ├── ur10/
│   │   └── config_ur10.py   # UR10 robot configuration (6-Dof)
│   ├── ur20/
│   │   └── config_ur20.py   # UR20 robot configuration (6-Dof)
│   └── doosan_m0609/
│       └── config_doosan_m0609.py  # Doosan M0609 configuration (6-Dof)
│   └── kuka_iiwa/
│       └── config_kuka_iiwa.py  # Kuka iiwa 14 820 configuration (7-Dof)
└── utils/
    ├── utils_compute.py     # Rotation matrices, conversions
    └── utils_graph.py       # 3D visualization utilities
```

## 🤖 Adding a New Robot Configuration

To add support for a new robot, follow these steps:

### Step 1: Create Configuration File

Create a new directory under `src/kinematics/config/YOUR_ROBOT_NAME/`:

```python
# src/kinematics/config/your_robot_name/config_your_robot.py

import numpy as np
from kinematics.config.config import BaseConfig

class YourRobotConfig(BaseConfig):
    """Configuration for Your Robot."""
    
    def __init__(self):
        # Denavit-Hartenberg (Khalil) parameters
        a = np.array([a1, a2, a3, a4, a5, a6])          # Translations along X
        d = np.array([d1, d2, d3, d4, d5, d6])          # Translations along Z
        alpha = np.array([a1, a2, a3, a4, a5, a6])      # Twists (rotations around X)
        theta = np.array([t1, t2, t3, t4, t5, t6])      # Angle offsets
        
        # Joint limits (radians)
        qmin = np.array([q1_min, q2_min, q3_min, q4_min, q5_min, q6_min])
        qmax = np.array([q1_max, q2_max, q3_max, q4_max, q5_max, q6_max])
        
        # Tool Center Point (4x4 transformation from last frame to tool)
        tcp = np.eye(4)
        tcp[:3, 3] = [x_offset, y_offset, z_offset]
        
        # Optional: dynamic parameters
        masses = np.array([m1, m2, m3, m4, m5, m6])  # Link masses (kg)
        cog = np.array([[cog_x1, cog_y1, cog_z1],     # Centers of gravity
                        [cog_x2, cog_y2, cog_z2],
                        # ... etc
                       ])
        
        super().__init__(a, d, alpha, theta, qmin, qmax, tcp=tcp, 
                        masses=masses, cog=cog)
```

### Step 2: Register in Config Module

Update `src/kinematics/config/__init__.py`:

```python
from .your_robot_name.config_your_robot import YourRobotConfig

__all__ = [..., "YourRobotConfig"]
```

### Step 3: Use Your Configuration

```python
from kinematics.config import YourRobotConfig
from kinematics.core import fk, ik

config = YourRobotConfig()
# Now use fk, ik, etc. with your robot!
```

## 📖 API Documentation

### Core Kinematics Functions

#### Forward Kinematics
```python
T = fk(q, a, d, alpha, theta, tcp)
```
Computes end-effector pose from joint angles.

#### Inverse Kinematics
```python
success, q = ik(target_pose, q_init, a, d, alpha, theta, tcp, 
                q_min, q_max, epsilon_pos=1e-4, epsilon_orient=1e-3)
```
Iteratively solves for joint angles given desired end-effector pose.

#### Jacobian
```python
J = get_jacobian(q, a, d, alpha, theta, tcp)
```
Computes 6×6 Jacobian matrix for velocity/force transformations.

#### Gravity Compensation
```python
tau = get_torque_gravity(q, a, d, alpha, theta, tcp, masses, cog)
```
Calculates joint torques needed to counteract gravity.

### Trajectory Generation

#### Linear Motion
```python
trajectory = compute_linear_trajectory(T_base, direction, distance)
```

#### Circular Motion
```python
trajectory = compute_circular_trajectory(T_base, center, axis, angle)
```

## 📊 Visualization

```python
from kinematics.core import fk
from kinematics.utils.utils_graph import create_graph, plot_robot_3d, plot_tcp

# Create 3D plot
ax = create_graph(title="My Robot")

# Plot robot configuration
q = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
T01, T02, T03, T04, T05, T06 = get_dh_mat(q, a, d, alpha, theta)
plot_robot_3d(ax, [T01, T02, T03, T04, T05, T06])

# Plot end-effector frame
T = fk(q, a, d, alpha, theta, tcp)
plot_tcp(ax, T)
```

## 🧪 Examples

See the `examples/` directory for complete working examples:
- `example_kinematics.py` - FK/IK demonstrations
- `example_compute.py` - Jacobian and force computations
- `example_trajectories.py` - Trajectory generation
- `example_config.py` - Configuration file usage

## 📝 Performance

All core computations use Numba JIT compilation for high performance:
- **Forward Kinematics**: < 1 ms
- **Inverse Kinematics**: 1-50 ms depending on convergence
- **Jacobian**: < 0.5 ms

## 🧬 Dependencies

- `numpy` - Numerical computations
- `numba` - JIT compilation for performance
- `matplotlib` - 3D visualization
- `scipy` - Testing
- `tqdm` - Visualization

## 📄 License

MIT

## 👤 Authors

Michaël Gross
