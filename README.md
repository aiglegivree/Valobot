# Valobot

**1st place in the EPFL Robopoly robotics competition.** Valobot is a team-built mobile robot that uses an onboard camera to follow a line and navigate corridors autonomously. The team designed the robot's mechanics, electronics, and control software together.

<img src="docs/valobot.jpg" alt="Valobot competition robot" width="480">

## Competition build

Start with [Code_fsm_v1](Code_fsm_v1/). Despite the version number in its name, this is where the team integrated the finite-state controller and saved the final run. The camera supplies visual observations, the controller selects a navigation action, and the motor interface applies it. Evan Ayoub contributed to controller integration and tuned line following and corridor navigation using recorded runs.

## Hardware

- [Mechanical CAD](CAD/) includes the chassis assembly, wheel and motor-mount designs. [CAD_V1](CAD_V1/) contains the CATIA assembly and component models.
- [Electronics prototype](Elec/elec_proto/) contains the KiCad board design. The [component list](Components.xlsx) records the hardware used during development.
