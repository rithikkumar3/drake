from pydrake.all import *
import numpy as np

def main():
    builder = DiagramBuilder()
    multibody_plant, scene_graph = AddMultibodyPlantSceneGraph(
        builder=builder, time_step=0.1)
    body = multibody_plant.AddRigidBody("sphere", SpatialInertia(
        mass=1.0,
        p_PScm_E=np.array([0., 0., 0.]),
        # refer drake sphere paramters documentation, why inertia?
        G_SP_E=UnitInertia(1.0, 1.0, 1.0)
    ))
    shape = Sphere(0.5)  # radius
    multibody_plant.RegisterVisualGeometry(
        body, RigidTransform(), shape, "sphere_vis", [0.5, 0.5, 0.5, 0.5])
    multibody_plant.Finalize()
    DrakeVisualizer().AddToBuilder(builder, scene_graph)
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    simulator = Simulator(diagram, diagram_context)
    simulator.set_publish_every_time_step(False)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(0.1)

if __name__ == '__main__':
    main()