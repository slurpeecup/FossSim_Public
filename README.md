# FossSIM

## It does the math for you!
TODO: A better README. 

Here's the basics:<br>
FossSIM, a (currently barebones) Python library supporting Fossen dynamics for generalized watercraft, <br>particularly the BlueROV2.
This repository depends on Thor Fossen's Python Vehicle Simulator for <br>benchmarking, which is included as a submodule.

Currently contains:
 - `fossen_solver.py` provides math helpers (should move to a different file) and a solver for <br>Fossen's equations of motion
 - `fossen_controlled_solver.py` provides a barebones (and likely incorrect as of now) <br>thruster allocator, pose, velocity, and wrench utilities, a channel-wise PD controller, and an <br>additive disturbance wrapper as a starting point for integrating with the extremum_seeking package.
 - `torpedo_shootout.py` provides an over-complicated benchmark for FossSIM against the torpedo <br>submarine controller implementation found in Thor Fossen's PythonVehicleSimulator. <br> <br> Torpedo_Shootout contains some useful examples for frame transformation, vehicle-specific solving, <br>actuator implementation, and 3D plotting.

## Install Guide
To get started, make sure to run 

```bash
git submodule init
git submodule update
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
