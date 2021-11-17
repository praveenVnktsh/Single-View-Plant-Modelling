
from typing import List
from vector import Vector
import numpy as np
import cv2
dt = 0.1
class Node:

    def __init__(self, x, y, coeffs = [1, 1], oppositeNode = None ):
        self.x = x

        self.y = y
        self.oppositeNode = oppositeNode
        self.connections : set[Node] = set()
        self.vector = Vector(x, y)
        self.coeffs = coeffs
        self.fallen = False
    def connect(self, node, terminate=False):

        self.connections.add(node)
        if not terminate:
            node.connect(self, terminate = True)

    def disconnect(self, node):
        self.connections.remove(node)
        node.connections.remove(self)


    def computeGradientForce(self, image):
        y = int(self.y)
        x = int(self.x)
        pel = image[y, x]
        grads = np.array([
            image[y, x+1] - pel,
            image[y+1, x+1] - pel,
            image[y+1, x] - pel,
            image[y+1, x-1] - pel,
            image[y, x-1] - pel,
            image[y-1, x-1] - pel,
            image[y-1, x] - pel,
            image[y-1, x + 1] - pel,
        ])

        direction = np.argmin(grads)

        vectors = [
            Vector( 1, 0).normalize(),
            Vector( 1, 1).normalize(),
            Vector( 0, 1).normalize(),
            Vector(-1, 1).normalize(),
            Vector(-1, 0).normalize(),
            Vector(-1,-1).normalize(),
            Vector( 0,-1).normalize(),
            Vector( 1,-1).normalize(),
        ]
        return vectors[direction] * float(pel)

    def computeInternalForce(self):
        repulsiveForce = Vector(0, 0)
        for n in self.connections :
            disp = n.vector  - self.vector

            norm  = disp.norm()
            thresh = [2, 5]
            if thresh[0] < norm < thresh[1]:
                mult = 0
            elif norm < thresh[0]:
                mult = -1
            else:
                mult = 1

            repulsiveForce += mult * (disp) 
        return repulsiveForce

    def constrainForce(self, force, image):
        y = int(self.y)
        x = int(self.x)
        pel = image[y, x]
        grads = np.array([
            image[y, x+1] - pel,
            image[y+1, x+1] - pel,
            image[y+1, x] - pel,    
            image[y+1, x-1] - pel,
            image[y, x-1] - pel,
            image[y-1, x-1] - pel,
            image[y-1, x] - pel,
            image[y-1, x + 1] - pel,
        ])
        grads =np.abs(grads)
        vectors = [
            Vector( 1, 0).normalize(),
            Vector( 1, 1).normalize(),
            Vector( 0, 1).normalize(),
            Vector(-1, 1).normalize(),
            Vector(-1, 0).normalize(),
            Vector(-1,-1).normalize(),
            Vector( 0,-1).normalize(),
            Vector( 1,-1).normalize(),
        ]
        direction = np.argmin(grads)
        
        return (vectors[direction] * force) * vectors[direction]


        





    def computeForce(self, image):
        # if self.coeffs != [1, 0] and image[int(self.y), int(self.x)] == 0:
        #     self.fallen = True
        # else:
        #     self.fallen = False

        # if self.fallen:
            # return self.constrainForce(self.computeInternalForce(), image)

        externalForce = self.computeGradientForce(image) 
        internalForce = self.computeInternalForce()

        externalcoeff = self.coeffs[0]

        # if self.fallen:
        #     externalcoeff = 30


        return externalForce * externalcoeff  +   internalForce * self.coeffs[1]
    

    def step(self, image, t, offsetForce = Vector(0, 0)):
        force = self.computeForce(image)
        
        
        # if not self.fixed:
        # b = 1
        # force -= (force * dt).normalize() * b
        force += offsetForce * 0.1
        self.y += force[1] * dt
        self.x += force[0] * dt
        
        self.vector = Vector(self.x, self.y)
        
        
        cv2.circle(t, (int(self.x), int(self.y)), 3, (0, 0, 255), -1)
        cv2.line(t, (int(self.x), int(self.y)), (int(self.x + force[0] * 2), int(self.y + force[1]* 2)), (0, 128, 0), 1)
        for n in self.connections:
            cv2.line(t, (int(self.x), int(self.y)), (int(n.x ), int(n.y )), (128, 128, 128), 1)



        return t
        

    def __str__(self):
        return "({}, {})".format(self.x, self.y)
        