import pybullet as p
import numpy as np
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary

class PlanningAviary(CtrlAviary):
    
    def _addObstacles(self):
        """Our own obstacle implementation for the planning environment."""

        x = np.arange(1, 5, 1)
        y = np.arange(1, 5, 1)
        X, Y = np.meshgrid(x, y)

        height = 0.5
        radius = 0.1

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                position = [X[i, j], Y[i, j], height / 2]
                cylinder = CylinderObstacle(position, radius, height)
                cylinder.add_to_simulation(self.CLIENT)

        cylinder.add_to_simulation(self.CLIENT)

class CylinderObstacle:
    def __init__(self, position, radius, height):
        self.position = position
        self.radius = radius
        self.height = height

    def add_to_simulation(self, client):
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=[1, 0, 0, 1],
            physicsClientId=client
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height,
            physicsClientId=client
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.position,
            physicsClientId=client
        )

