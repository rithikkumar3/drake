import numpy as np
import matplotlib.pyplot as plt

from pydrake.common.cpp_param import List
from pydrake.common.value import Value
from pydrake.geometry import (
    DrakeVisualizer,
    DrakeVisualizerParams,
    Role,
)
from pydrake.all import *

from simple_block_world import *


class ForcePublisher(LeafSystem):
    def __init__(self, body, p_WT, v_WT, fm):
        LeafSystem.__init__(self)

        self.body = body
        self.p_WT = p_WT
        self.v_WT = v_WT
        self.fm = fm
        self.num_states = len(self.p_WT)

        self.pt_num = 0

        forces_cls = Value[List[ExternallyAppliedSpatialForce_[float]]]
        self.DeclareAbstractOutputPort("spatial_forces",
                                       lambda: forces_cls(),
                                       self.publish_force)

    def publish_force(self, context, spatial_forces_vector):
        print("publishing force.")
        fp = self.p_WT[self.pt_num]
        fv = self.v_WT[self.pt_num]
        fm = self.fm[self.pt_num]

        print("fp_val: ", fp)
        print("fv_val: ", fv)
        print("fm_val: ", fm)

        if self.pt_num < self.num_states-1:
            self.pt_num = self.pt_num+1

        force = ExternallyAppliedSpatialForce_[float]()

        force.body_index = self.body.index()

        spatial_force = SpatialForce(
            tau=[0, 0, 0],
            f=fv*fm)

        force.p_BoBq_B = fp
        force.F_Bq_W = spatial_force

        spatial_forces_vector.set_value([force])


# ======================
# WORLD
# ======================
dt = 0.01
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)

add_ground(plant)
color = np.array([0.0, 0.0, 1.0, 0.3])
cube_name = "cube"
edge_length = 0.1
mass = 1.0
add_cube(plant, cube_name, color, edge_length, mass)

plant.Finalize()
print_plant_info(plant, scene_graph)

# Connect to visualizer
params = DrakeVisualizerParams(role=Role.kProximity, show_hydroelastic=True)
DrakeVisualizer(params=params).AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# ======================
# ADD FORCE
# ======================

p_WT = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
v_WT = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
fm = np.array([[150.0], [50.0], [10.0], [5.0], [0.0]])

body = plant.GetBodyByName(cube_name)
force_pub = builder.AddSystem(ForcePublisher(body, p_WT, v_WT, fm))
builder.Connect(force_pub.get_output_port(
    0), plant.get_applied_spatial_force_input_port())
logger = LogVectorOutput(plant.get_state_output_port(), builder, dt)


diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(diagram_context)


# ======================
# SIMULATE
# ======================

# Set the initial state
q0 = np.array([1., 0., 0., 0., 0.0, 0., edge_length/2.0])
v0 = np.zeros(plant.num_velocities())
x0 = np.hstack((q0, v0))

diagram_context.SetDiscreteState(x0)
diagram.ForcedPublish(diagram_context)


simulator = Simulator(diagram, diagram_context)
simulator.set_publish_every_time_step(True)
simulator.set_target_realtime_rate(0.5)
simulator.set_publish_at_initialization(True)
simulator.Initialize()
simulator.AdvanceTo(1.0)


context = simulator.get_mutable_context()
log = logger.FindMutableLog(context)
t = log.sample_times()
x = log.data()
x = x.transpose()
# print("x:\n", x[:, 4])

plt.scatter(t, x[:, 4], label="X")
plt.scatter(t, x[:, 5], label="Y")
plt.scatter(t, x[:, 6], label="Z")

# Create the legend
plt.legend()

plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")

plt.show()
