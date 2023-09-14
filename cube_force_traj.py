import numpy as np
from pydrake.all import *
import time
from simple_block_world import *  # Assuming you have this custom module

# ======================
# WORLD
# ======================

dt = 0.001
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

add_ground(plant)
color = np.array([0.0, 0.0, 1.0, 0.3])
cube_name = "cube1"
edge_length = 0.1
mass = 1.0
add_cube(plant, cube_name, color, edge_length, mass)

plant.Finalize()

# Connect to visualizer
params = DrakeVisualizerParams(role=Role.kProximity, show_hydroelastic=True)
DrakeVisualizer(params=params).AddToBuilder(builder, scene_graph)

diagram = builder.Build()

diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)

q0_cube1 = np.array([1., 0., 0., 0., 0.0, 0., 0.05])
v0 = np.zeros(plant.num_velocities())
x0 = np.hstack((q0_cube1, v0))

initial_force = np.array([1000, 0, 0])  # In Newtons

# Update initial velocity based on the force
# v_new = v_old + (F/m) * dt
v0[3] += (initial_force[0] / mass) * dt # for x-axis
v0[4] += (initial_force[1] / mass) * dt # for y-axis
v0[5] += (initial_force[2] / mass) * dt # for z-axis

# Update the state with new velocity
x0 = np.hstack((q0_cube1, v0))

diagram_context.SetDiscreteState(x0)
diagram.ForcedPublish(diagram_context)


simulator = Simulator(diagram, diagram_context)
simulator.set_publish_every_time_step(False)
simulator.set_target_realtime_rate(0.1)
simulator.Initialize()

simulator.AdvanceTo(1.0)
