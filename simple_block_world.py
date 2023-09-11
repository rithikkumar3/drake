import numpy as np
from pydrake.all import *
import time

# fix the print precision for numpy
np.set_printoptions(precision=7, suppress=True)

# Contact model parameters
dissipation = 5.0              # controls "bounciness" of collisions: lower is bouncier
# controls "squishiness" of collisions: lower is squishier
hydroelastic_modulus = 5e6
resolution_hint = 0.05         # smaller means a finer mesh
mu_static = 0.3
mu_dynamic = 0.2

# Hydroelastic, Point, or HydroelasticWithFallback
contact_model = ContactModel.kHydroelasticWithFallback
mesh_type = HydroelasticContactRepresentation.kTriangle  # Triangle or Polygon


def get_proximity_properties():
    props = ProximityProperties()
    friction = CoulombFriction(0.7*mu_static, 0.7*mu_dynamic)
    AddCompliantHydroelasticProperties(
        resolution_hint, hydroelastic_modulus, props)
    AddContactMaterial(dissipation=dissipation,
                       friction=friction, properties=props)
    return props


def add_ground(plant: MultibodyPlant):
    # Add the ground as a big box.
    ground_idx = plant.AddModelInstance("ground")
    ground_box = plant.AddRigidBody(
        "ground", ground_idx, SpatialInertia(1, np.array([0, 0, 0]), UnitInertia(1, 1, 1)))
    X_WG = RigidTransform([0, 0, -0.1])
    props = get_proximity_properties()
    plant.RegisterCollisionGeometry(
        ground_box, RigidTransform(), Box(2, 2, 0.1), "ground",
        props)
    plant.RegisterVisualGeometry(
        ground_box, RigidTransform(), Box(2, 2, 0.1), "ground",
        [0.4, 0.5, 0.5, 1.])
    plant.WeldFrames(plant.world_frame(), ground_box.body_frame(), X_WG)


def add_sphere(plant: MultibodyPlant, name: str = "sphere", color: np.array = [1.0, 0.0, 0.0, 0.5], radius=0.1, mass=0.1):
    # Add a sphere with compliant hydroelastic contact
    I = SpatialInertia.SolidSphereWithMass(mass, radius)
    sphere_idx = plant.AddModelInstance(name)
    sphere = plant.AddRigidBody(name, sphere_idx, I)
    X_sphere = RigidTransform()
    friction = CoulombFriction(0.7*mu_static, 0.7*mu_dynamic)
    props = get_proximity_properties()
    plant.RegisterCollisionGeometry(sphere, X_sphere, Sphere(radius),
                                    "ball_collision", props)

    plant.RegisterVisualGeometry(
        sphere, X_sphere, Sphere(radius), "ball_visual", color)

    # Add some spots to visualize the ball's roation
    spot_color1 = np.array([0.5, 0.0, 0.0, 0.5])
    spot_color2 = np.array([0.0, 0.5, 0.0, 0.5])
    spot_color3 = np.array([0.0, 0.0, 0.5, 0.5])

    spot_radius = 0.08*radius
    spot = Sphere(spot_radius)
    spot_offset = radius - 0.45*spot_radius

    plant.RegisterVisualGeometry(
        sphere, RigidTransform(RotationMatrix(), np.array([radius, 0, 0])),
        spot, "sphere_x+", spot_color1)
    # plant.RegisterVisualGeometry(
    #     sphere, RigidTransform(RotationMatrix(), np.array([-radius, 0, 0])),
    #     spot, "sphere_x-", spot_color)
    plant.RegisterVisualGeometry(
        sphere, RigidTransform(RotationMatrix(), np.array([0, radius, 0])),
        spot, "sphere_y+", spot_color2)
    # plant.RegisterVisualGeometry(
    #     sphere, RigidTransform(RotationMatrix(), np.array([0, -radius, 0])),
    #     spot, "sphere_y-", spot_color)
    plant.RegisterVisualGeometry(
        sphere, RigidTransform(RotationMatrix(), np.array([0, 0, radius])),
        spot, "sphere_z+", spot_color3)
    # plant.RegisterVisualGeometry(
    #     sphere, RigidTransform(RotationMatrix(), np.array([0, 0, -radius])),
    #     spot, "sphere_z-", spot_color)

    return sphere_idx


def add_cube(plant: MultibodyPlant, name: str = "cube", color: np.array = [0.0, 0.0, 1.0, 0.5], edge_length=0.1, mass=0.1):
    # Add a cube with compliant hydroelastic contact
    cube_idx = plant.AddModelInstance(name)

    # Calculate inertia for cube (uniform density assumed)
    I = SpatialInertia(mass=mass, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia.SolidBox(edge_length, edge_length, edge_length))

    # Add the rigid body for cube
    cube = plant.AddRigidBody(name, cube_idx, I)

    # The pose of the cube
    X_cube = RigidTransform()

    # Friction coefficients 
    friction = CoulombFriction(0.7, 0.6)
    
    # Proximity properties 
    props = get_proximity_properties()

    # Register collision geometry
    plant.RegisterCollisionGeometry(cube, X_cube, Box(edge_length, edge_length, edge_length),
                                    name + "_collision", props)

    # Register visual geometry
    plant.RegisterVisualGeometry(
        cube, X_cube, Box(edge_length, edge_length, edge_length), name + "_visual", color)

    return cube_idx

def add_box(plant: MultibodyPlant):
    # Add boxes
    masses = [1.]
    box_sizes = [np.array([0.2, 0.2, 0.2])]

    assert isinstance(masses, list)
    assert isinstance(box_sizes, list)
    assert len(masses) == len(box_sizes)

    num_boxes = len(masses)
    boxes = []
    boxes_geometry_id = []
    props = get_proximity_properties()
    for i in range(num_boxes):
        box_name = f"box{i}"
        box_idx = plant.AddModelInstance(box_name)
        box_body = plant.AddRigidBody(
            box_name, box_idx, SpatialInertia(
                masses[i], np.array([0, 0, 0]), UnitInertia(1, 1, 1)))
        boxes.append(box_body)
        box_shape = Box(box_sizes[i][0], box_sizes[i][1], box_sizes[i][2])
        box_geo = plant.RegisterCollisionGeometry(
            box_body, RigidTransform(
            ), box_shape, f"{box_name}_box",
            props)
        boxes_geometry_id.append(box_geo)
        plant.RegisterVisualGeometry(
            box_body, RigidTransform(
            ), box_shape, f"{box_name}_box",
            [0.8, 1.0, 0.0, 0.5])


def finalize_plant(plant: MultibodyPlant):
    # Choose contact model
    plant.set_contact_surface_representation(mesh_type)
    plant.set_contact_model(contact_model)
    plant.Finalize()


def print_plant_info(plant: MultibodyPlant, scene_graph):
    inspector = scene_graph.model_inspector()

    print("=== Frames")
    for frame_id in inspector.GetAllFrameIds():
        print(inspector.GetName(frame_id))

    print("=== Geometries")
    for geometry_id in inspector.GetAllGeometryIds():
        print(inspector.GetName(geometry_id))

    print("=== Bodies")
    for i in range(plant.num_bodies()):
        idx = BodyIndex(i)
        body = plant.get_body(idx)
        print(body.name())

    print("Num actuators: ", plant.num_actuators())
    print("Num positions: ", plant.num_positions())
    print("Num velocities: ", plant.num_velocities())


def print_contact_pairs(contact_pairs, inspector):
    print("All collision pairings:")
    for pair_id, pair in enumerate(contact_pairs):
        print(f"\tPair {pair_id}")
        print(f"\t\t{pair[0]}: {inspector.GetName(pair[0])}")
        print(f"\t\t{pair[1]}: {inspector.GetName(pair[1])}")
