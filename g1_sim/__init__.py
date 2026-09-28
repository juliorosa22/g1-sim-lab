"""G1 Isaac Sim sensor-testing package (Stage 1: Sense).

Modules:
    config       -- every tunable value (paths, mount points, topics, FOVs)
    stage_utils  -- small USD/stage helpers shared across modules
    robot        -- G1Robot: loading, head_link validation, spawn positioning
    environment  -- RoomEnvironment: loading a sample scene (e.g. Simple_Room)
    sensors      -- LidarSensor, CameraSensor, ImuSensor

Note: this package imports omni/pxr modules, which only work once
isaacsim.SimulationApp has been constructed -- always import from g1_sim
*after* creating the SimulationApp instance (see run_sensors.py).
"""
