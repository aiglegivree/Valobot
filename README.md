# Valobot

**1st place in the EPFL Robopoly robotics competition.** Valobot is a team-built mobile robot that uses an onboard camera to follow a line and navigate corridors autonomously. The team designed the robot's mechanics, electronics, and control software together.

<img src="docs/valobot.jpg" alt="Valobot competition robot" width="480">

[Watch the final run](Code_fsm_v1/run_recordings/final_run.mp4) · [Browse the navigation software](Code_fsm_v1/) · [View the project portfolio](https://aiglegivree.github.io/evanayoub/#projects)

## Competition build

Start with [Code_fsm_v1](Code_fsm_v1/). Despite the version number in its name, this is where the team integrated the finite-state controller and saved the final run. The camera supplies visual observations, the controller selects a navigation action, and the motor interface applies it. Evan Ayoub contributed to controller integration and tuned line following and corridor navigation using recorded runs.

## Hardware

- [Mechanical CAD](CAD/) includes the chassis assembly, wheel and motor-mount designs. [CAD_V1](CAD_V1/) contains the CATIA assembly and component models.
- [Electronics prototype](Elec/elec_proto/) contains the KiCad board design. The [component list](Components.xlsx) records the hardware used during development.

## Development history

The [pre-cleanup snapshot](https://github.com/aiglegivree/Valobot/tree/81965a84623f7e89856ed3ef548d561cf87e1eee) preserves the earlier controllers, standalone experiments, hardware checks, and test recordings.
