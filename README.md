# xarm-lite6-ctc-vs-pd-trajectory-control

Robotics Control Challenge for the xArm Lite 6 manipulator.  
This project implements task-space trajectory tracking using inverse kinematics and compares two control strategies: **PD control** and **Computed Torque Control (CTC)**.

---

## Project Overview

The robot executes a Cartesian trajectory defined by several waypoints with two different height levels.  
The trajectory is converted into joint-space references using inverse kinematics.

The objective is to compare the tracking performance of:

- PD Controller
- Computed Torque Controller (CTC)

---

## Control Architecture

Task-space trajectory  
↓  
Inverse kinematics  
↓  
Joint references  
↓  
Controller (PD or CTC)  
↓  
Robot execution  
↓  
Data logging and analysis  

---

## Controllers

### PD Controller

The proportional-derivative controller is defined as:

τ = Kp(qd − q) + Kd(q̇d − q̇)

This controller is simple and widely used in industrial manipulators.

### Computed Torque Control (CTC)

Computed torque control compensates for the robot dynamics:

τ = M(q)(q̈d + Kv ė + Kp e) + C(q,q̇)q̇ + G(q)

This approach improves tracking accuracy by canceling nonlinear dynamics.

---

## Repository Structure

ctc_controller_cpp_node.cpp  
Computed Torque controller implementation

pd_controller_cpp_node.cpp  
PD controller implementation

ik_solver.py  
Inverse kinematics solver

ik_solver_kdl.py  
KDL-based inverse kinematics

data_logger_cpp_node.cpp  
Robot data logger

metrics.py  
Tracking error metrics

plots.py  
Result visualization

---

## Experiments

The following experiments are performed:

1. PD controller without disturbances  
2. CTC controller without disturbances  
3. PD controller with disturbances  
4. CTC controller with disturbances  

The performance is evaluated using trajectory tracking error and waypoint accuracy.

---

## Author

Robotics Control Challenge  
xArm Lite 6 trajectory tracking project
