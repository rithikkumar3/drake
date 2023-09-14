import numpy as np
from pydrake.all import *
import time
from simple_block_world import *

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
#print_plant_info(plant, scene_graph)


# Connect to visualizer
params = DrakeVisualizerParams(role=Role.kProximity, show_hydroelastic=True)
DrakeVisualizer(params=params).AddToBuilder(builder, scene_graph)

diagram = builder.Build()

diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)
print("Size of continuous state vector:", plant_context.get_continuous_state_vector().size())


# ======================
# SIMULATION SETUP
# ======================

# Set the initial state
q0_cube1 = np.array([1., 0., 0., 0., 0.0, 0., 0.05])


# this initial velocity is basically the push we are providing to the cube
v0 = np.array([0., 0., 0., 1.0, 1.0, 1.0])
x0 = np.hstack((q0_cube1, v0)) 

diagram_context.SetDiscreteState(x0)
#plant_context.SetContinuousState(x0)
diagram.ForcedPublish(diagram_context)

simulator = Simulator(diagram, diagram_context)
simulator.set_publish_every_time_step(False)
simulator.set_target_realtime_rate(1.0)
simulator.Initialize()

simulator.AdvanceTo(1.0)
