[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.11 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)  

# 🤖 Kinematics Package

A high-performance Python package for computing **forward and inverse kinematics** of N-DOF robotic manipulators. Built with speed in mind using Numba JIT compilation.

## 📋 Overview

This package provides tools for:
- **Forward Kinematics (FK)**: Compute end-effector pose from joint angles
- **Inverse Kinematics (IK)**: Find joint angles from desired end-effector pose
- **Jacobian Computation**: Calculate kinematic relationships for velocity/force transformation
- **Gravity Compensation**: Compute required torques to compensate gravitational effects
- **Trajectory Generation**: Create linear and circular motion paths
- **Manipulability Analysis**: Compute and visualize force/velocity ellipsoids
- **Multi-Convention Support**: Use either Standard DH or Modified DH (Khalil) parameterization

## ⚡ Features

- ✅ Fast JIT-compiled core algorithms using Numba
- ✅ Object-oriented API with `RobotArmKinematics` class
- ✅ Support for both **Standard DH** and **Modified DH (Khalil)** conventions
- ✅ Support for multiple robot configurations (UR10, UR20, Doosan M0609, Kuka IIwa, ...)
- ✅ IK with both position and orientation constraints
- ✅ Dynamic analysis with gravity effects
- ✅ Manipulability and force analysis with ellipsoid visualization

## 📦 Installation

### Uv

```bash
git clone https://github.com/MichaelGs1/Kinematics.git
cd Kinematics
uv sync
```

### Pip

```bash
git clone https://github.com/MichaelGs1/Kinematics.git
cd Kinematics
pip install .
```

### Optional dev dependencies :

### Uv

```bash
uv sync --extra dev
```

### Pip

```bash
pip install "pytest>=7.0" "ruff>=0.3.0" "mypy>=1.8.0" "pre_commit>=4.6.0"
```


## 🚀 Quick Start

```python
from kinematics.config.ur10 import UR10Config
from kinematics.core.robot import RobotArmKinematics
import numpy as np

# Load robot configuration
config = UR10Config()
robot = RobotArmKinematics(config)

# Forward kinematics: compute end-effector pose
q = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])  # joint angles (rad)
T = robot.fk(q)

print("End-effector pose:\n", T)

# Inverse kinematics: find joint angles for desired pose
target_pose = T.copy()
success, q_solution = robot.ik(target_pose, q)

print(f"IK Success: {success}")
print(f"Joint angles (rad): {q_solution}")

# Compute Jacobian
J = robot.get_jacobian(q)
print(f"Jacobian shape: {J.shape}")

# Compute gravity compensation torques
tau_gravity = robot.get_gravity_torque(q)
print(f"Gravity torques (N.m): {tau_gravity}")
```

## 🔧 Denavit-Hartenberg Conventions

This package supports two DH parameterization conventions:
- **Standard DH (DH)**: Classical Denavit-Hartenberg convention (Craig)
- **Modified DH (Khalil)**: Modified convention (often used in robotics research)

You can specify which convention to use when creating a robot configuration via the `RepresentationType` enum.

### Standard DH (Craig) Convention

Each joint is described by 4 parameters:

```
┌─────────────────────────────────────────────────────────────────┐
│  Parameter  │  Symbol  │      Description                       │
├─────────────────────────────────────────────────────────────────┤
│  Distance   │    d     │  Translation along Z-axis (m)          │
│  Offset     │    a     │  Translation along X-axis (m)          │
│  Twist      │  alpha   │  Rotation around X-axis (rad)          │
│  Angle      │  theta   │  Rotation around Z-axis (rad) + q      │
└─────────────────────────────────────────────────────────────────┘
```

Homogeneous Transformation (in order):
$$T_i^{i-1} = Rot(z, θ_i) × Trans(z, d_i) × Trans(x, a_i) × Rot(x, α_i)$$

Transformation Matrix:
```
    ┌                                                ┐
    │ cos(θ)  -sin(θ)cos(α)   sin(θ)sin(α)  a·cos(θ) │
    │ sin(θ)   cos(θ)cos(α)  -cos(θ)sin(α)  a·sin(θ) │
    │    0          sin(α)         cos(α)       d    │
    │    0             0              0          1   │
    └                                                ┘
```

### Modified DH (Khalil) Convention

Each joint is described by 4 parameters:

```
┌─────────────────────────────────────────────────────────────────┐
│  Parameter  │  Symbol  │      Description                       │
├─────────────────────────────────────────────────────────────────┤
│  Distance   │    d     │  Translation along Z-axis (m)          │
│  Offset     │    a     │  Translation along X-axis (m)          │
│  Twist      │  alpha   │  Rotation around X-axis (rad)          │
│  Angle      │  theta   │  Rotation around Z-axis (rad) + q      │
└─────────────────────────────────────────────────────────────────┘
```

**Important:** In Khalil convention, `alpha` and `a` refer to the **previous** joint frame (frame i-1), not the current frame.

Homogeneous Transformation (in order):
$$T_i^{i-1} = Rot(x, α_{i-1}) × Trans(x, a_{i-1}) × Rot(z, θ_i) × Trans(z, d_i)$$

Transformation Matrix:
```
    ┌                                             ┐
    │ cos(θ)         -sin(θ)       0        a     │
    │ sin(θ)cos(α) cos(θ)cos(α) -sin(α) -d·sin(α) │
    │ sin(θ)sin(α) cos(θ)sin(α)  cos(α)  d·cos(α) │
    │    0              0          0        1     │
    └                                             ┘
```

### Key Differences

| Aspect | Standard DH | Modified DH (Khalil) |
|--------|------------|----------------------|
| Frame attachment | At joint i end | At joint i+1 beginning |
| Transformation order | Z-rotation, Z-translation, X-translation, X-rotation | X-rotation, X-translation, Z-rotation, Z-translation |
| alpha & a reference | Current frame (i) | Previous frame (i-1) |
| Use case | Classical robotics texts | Research, KUKA robots |

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
from kinematics.config.config import BaseConfig, RepresentationType

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
        masses = np.array([m1, m2, m3, m4, m5, m6, m_tool])  # Link masses (kg)
        cog = np.array([[cog_x1, cog_y1, cog_z1],     # Centers of gravity
                        [cog_x2, cog_y2, cog_z2],
                        # ... 7 rows total (including tool)
                       ])
        
        joint_velocity_max = np.array([v1_max, v2_max, v3_max, v4_max, v5_max, v6_max])
        torque_max = np.array([tau1_max, tau2_max, tau3_max, tau4_max, tau5_max, tau6_max])
        
        super().__init__(
            a=a, d=d, alpha=alpha, theta=theta,
            representation_type=RepresentationType.DH_KHALIL,  # or RepresentationType.DH
            qmin=qmin, qmax=qmax, tcp=tcp,
            q_point_max=joint_velocity_max,
            torque_max=torque_max,
            masses=masses, cog=cog
        )
```

### Step 2: Create __init__.py

```python
# src/kinematics/config/your_robot_name/__init__.py

from .config_your_robot import YourRobotConfig

__all__ = ["YourRobotConfig"]
```

### Step 3: Use Your Configuration

```python
from kinematics.config.your_robot_name import YourRobotConfig
from kinematics.core.robot import RobotArmKinematics

config = YourRobotConfig()
robot = RobotArmKinematics(config)

# Now use all robot methods!
T = robot.fk(q)
success, q_sol = robot.ik(T, q)
```

## 📖 API Documentation

### Creating a Robot Instance

```python
from kinematics.config.ur10 import UR10Config
from kinematics.core.robot import RobotArmKinematics

config = UR10Config()
robot = RobotArmKinematics(config)
```

### Core Kinematics Methods

#### Forward Kinematics
```python
T = robot.fk(q)
```
Computes end-effector pose (4×4 homogeneous matrix) from joint angles.

**Parameters:**
- `q` (np.ndarray): Joint angles in radians, shape (n,)

**Returns:**
- `T` (np.ndarray): 4×4 homogeneous transformation matrix

#### Inverse Kinematics
```python
success, q = robot.ik(target_pose, q_init, epsilon_pos=1e-4, 
                      epsilon_orient=1e-3, max_iter=1000, alpha_fix=0.2)
```
Iteratively solves for joint angles given desired end-effector pose.

**Parameters:**
- `target_pose` (np.ndarray): Desired 4×4 homogeneous transformation matrix
- `q_init` (np.ndarray): Initial joint angle guess, shape (n,)
- `epsilon_pos` (float): Position error threshold (m), default 1e-4
- `epsilon_orient` (float): Orientation error threshold (rad), default 1e-3
- `max_iter` (int): Maximum iterations, default 1000
- `alpha_fix` (float): Step size damping [0, 1], default 0.2

**Returns:**
- `success` (bool): Whether IK converged
- `q` (np.ndarray): Joint angles solution, shape (n,)

#### Link Transformation Matrices
```python
transforms = robot.get_link_matrix(q)
```
Computes homogeneous transformations from base to each joint frame.

**Parameters:**
- `q` (np.ndarray): Joint angles in radians, shape (n,)

**Returns:**
- `transforms` (np.ndarray): Array of n 4×4 transformation matrices

#### Jacobian Matrix
```python
J = robot.get_jacobian(q)
```
Computes 6×n Jacobian matrix relating joint velocities to end-effector velocities/angular velocities.

**Parameters:**
- `q` (np.ndarray): Joint angles in radians, shape (n,)

**Returns:**
- `J` (np.ndarray): 6×n Jacobian matrix

#### Gravity Compensation Torques
```python
tau = robot.get_gravity_torque(q)
```
Calculates joint torques needed to counteract gravity effects.

**Parameters:**
- `q` (np.ndarray): Joint angles in radians, shape (n,)

**Returns:**
- `tau` (np.ndarray): Joint torques in N.m, shape (n,)

#### Force from Torques
```python
force = robot.get_force_tcp(q, tau)
```
Converts joint torques to end-effector forces and moments using Jacobian transpose.

**Parameters:**
- `q` (np.ndarray): Joint angles in radians, shape (n,)
- `tau` (np.ndarray): Joint torques in N.m, shape (n,)

**Returns:**
- `force` (np.ndarray): End-effector force/moment vector [Fx, Fy, Fz, Mx, My, Mz], shape (6,)

### Manipulability Analysis

#### Force Ellipsoid
```python
A = robot.compute_force_ellipsoid(J, normalized=False)
```
Computes force transmission ellipsoid matrix.

**Parameters:**
- `J` (np.ndarray): Jacobian submatrix (3×n for translation)
- `normalized` (bool): Normalize by maximum torque, default False

**Returns:**
- `A` (np.ndarray): 3×3 ellipsoid matrix

#### Velocity Ellipsoid
```python
A = robot.compute_velocity_ellipsoid(J, normalized=False)
```
Computes velocity manipulation ellipsoid matrix.

**Parameters:**
- `J` (np.ndarray): Jacobian submatrix (3×n for translation)
- `normalized` (bool): Normalize by maximum velocity, default False

**Returns:**
- `A` (np.ndarray): 3×3 ellipsoid matrix

#### Ellipsoid Amplitude
```python
amplitude = robot.get_amplitude_ellipsoid(A, direction)
```
Computes radius of ellipsoid along a given direction.

**Parameters:**
- `A` (np.ndarray): 3×3 ellipsoid matrix
- `direction` (np.ndarray): Direction vector, shape (3,)

**Returns:**
- `amplitude` (float): Radius along the given direction

### Trajectory Generation

Trajectory generation functions are available for motion planning:

```python
from kinematics.core.trajectory import compute_linear_trajectory, compute_circular_trajectory

# Linear trajectory from current pose
T_current = robot.fk(q)
trajectory = compute_linear_trajectory(T_current, direction=[0, 0, 1], distance=0.5)

# Circular trajectory
trajectory = compute_circular_trajectory(T_current, center=[0, 0, 0], 
                                       axis=[1, 0, 0], angle=np.pi)
```

## 📊 Visualization

```python
from kinematics.core.robot import RobotArmKinematics
from kinematics.config.ur10 import UR10Config
from kinematics.utils.utils_graph import create_graph, plot_robot_3d, plot_tcp
import numpy as np

# Create robot
config = UR10Config()
robot = RobotArmKinematics(config)

# Create 3D plot
ax = create_graph(title="UR10 Robot")

# Compute configuration
q = np.array([0, -1.57, 1.57, -1.57, -1.57, 0])
transforms = robot.get_link_matrix(q)

# Plot robot configuration
plot_robot_3d(ax, transforms)

# Plot end-effector frame
T = robot.fk(q)
plot_tcp(ax, T)
```

## 🧪 Examples

See the `examples/` directory for complete working examples:
- `example_config.py` - Configuration file usage and robot instantiation
- `example_kinematics.py` - FK/IK demonstrations with RobotArmKinematics
- `example_trajectories.py` - Trajectory generation
- `example_compute.py` - Test of computation
- `example_compute.py` - Time of computation

## 📝 Performance

All core computations use Numba JIT compilation for high performance:
- **Forward Kinematics**: < 10 ns
- **Jacobian**: < 10 ns
- **Inverse Kinematics**: 1-15 ms depending on convergence

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
ALTEN SA
