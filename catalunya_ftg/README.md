# Catalunya Follow-the-Gap files

This folder contains a copy of the controller, HUD, track centerline, run logs, tuning state, and RViz layout used for Catalunya. For installation and the normal run sequence, follow the [repository setup guide](../README.md).

## RViz layout topics

`catalunya.rviz` has no filesystem dependency on this folder; it subscribes to ROS topics published by the simulator and controller:

| Display | Topic or frame | Publisher |
|---|---|---|
| Map | `/map` | `nav2_map_server` from the simulator launch |
| Ego model | `/ego_robot_description` and TF | `robot_state_publisher` and Gym bridge |
| Opponent model | `/opp_robot_description` and TF | Available when `num_agent: 2`; disabled by default in RViz |
| LiDAR | `/scan` | Gym bridge |
| Road surface, start line, trail | `/track_surface`, `/start_finish_line`, `/car_trail` | Follow-the-Gap controller |
| HUD | `/race_hud` | Follow-the-Gap controller (requires Pillow) |

The same layout is installed at `src/f1tenth_gym_ros/launch/catalunya.rviz` and opened by the default simulator launch. Keep the two copies aligned if editing the layout.

The robot model dimensions follow the Gym vehicle parameters: 0.58 m long, 0.31 m wide, with a 0.3302 m wheelbase. The LiDAR frame is at the vehicle pose origin, matching the Gym scan origin.
