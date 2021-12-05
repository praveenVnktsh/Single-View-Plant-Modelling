from typing import List
from utils import scaleAndShow
from node import Node

import numpy as np

from vector import Vector
dt = 0.001

import cv2

def getRope(pt1, pt2, mask = None, slackstop = 50, slacksbtm = 50, gradation = False):

    rope = [Node(pt1[0], pt1[1])]
    
    for i in range(slackstop):
        rope.append(Node(pt1[0], pt1[1]))

    rope.append(Node(pt2[0], pt2[1]))
    
    for i in range(slacksbtm):
        rope.append(Node(pt2[0], pt2[1]))
    
    for i in range(1, len(rope)):
        rope[i].connect(rope[i-1])
        rope[i].coeffs = [4, 0.5]


    if gradation:
        hwidth = len(rope)/2.
        scalea = 5
        scaleb = 2
        for i in range(len(rope)):
                
            ca = (abs(i)/len(rope))  * scalea + 1
            cb = (1 - abs(i)/len(rope))  * scaleb + 5
            rope[i].coeffs = [ca, cb]

        rope[0].coeffs = [1, 0]
        rope[-1].coeffs = [1, 0]


    return rope


def getCircle(centroid):
    leaf = []
    dist = 100
    centroid = np.array(centroid, dtype= int)
    theta = np.arange(0, 2*np.pi, 0.3)
    leaf = [
        Node(centroid[0] + 60*np.sin(t), centroid[1] + 60*np.cos(t)) for t in theta
    ]

    for i in range(1, len(leaf)):
        leaf[i].connect(leaf[i-1])

    leaf[0].connect(leaf[-1])

    # r1 = getRope(centroid + np.array([-dist, -dist]), centroid  + np.array([dist, -dist]), slackstop = 3, slacksbtm = 3)
    # r2 = getRope(centroid + np.array([dist, -dist + 1]), centroid  + np.array([dist, dist]), slackstop = 3, slacksbtm = 2)
    # r3 = getRope(centroid + np.array([dist, dist + 1]), centroid  + np.array([dist, -dist]), slackstop = 3, slacksbtm = 2)

    # r3 = getRope((880, 302), (880, 328), slackstop = 3, slacksbtm = 3)
    # r4 = getRope((730, 328), (722, 275), slackstop = 3, slacksbtm = 3)
    
    # r1[0].connect(r3[-1])
    # r2[0].connect(r1[-1])
    # r3[0].connect(r2[-1])
    # r3[-1].connect(backwardRope[0])
    # r4[0].connect(backwardRope[-1])


    
    # leaf += r1
    # leaf += r2
    # leaf += r3

    for node in leaf:
        node.coeffs = [1, 1]

    return leaf


class Leaf:

    def __init__(self, centroid = None, base = None):
        

        self.tiplength = 30

        self.thetaT = np.radians(75)
        self.thetaB = np.radians(75)

        self.baseTriangleWidth = 30
        self.topTriangleWidth = 30

        self.maxWidth = 50
        self.edgeWidth = 20
        
        self.theta = np.radians(30)
        self.length = 100
        self.ratio = 0.3

        self.stem = Node(300, 300)

        center  = centroid
        # center = (400, 223)

        # self.leaf : List[Node] = getCircle(center)
        self.leaf = []
        
        # self.leaf += r4

        self.stem : List[Node] = getRope(center, base, slackstop = 4, slacksbtm = 4, gradation = True)
        # self.stem[0].connect(self.leaf[0])
        # self.leaf[0].coeffs = [1, 1]



    def step(self, leafGrad, stemGrad, vizimg):
     
        # mainCentroid = Vector(0, 0)
        # for node in self.leaf:
        #     mainCentroid += node.vector

        # mainCentroid /= len(self.leaf)
        
        # for i, node in enumerate(self.leaf):
        #     if i != len(self.leaf) - 1:
        #         centroid = (self.leaf[i-1].vector + self.leaf[i].vector + self.leaf[i+1].vector)/3.
        #     else:
        #         centroid = (self.leaf[i-1].vector + self.leaf[i].vector )/2.
        #     normalForce = (node.vector - centroid ) * 0.5
        #     normalForce += 0.2 * (node.vector - mainCentroid)
        #     # vizimg = node.step(leafGrad, vizimg, normalForce,   color = (255, 0, 255))
        # cv2.circle(vizimg, (int(mainCentroid[0]), int(mainCentroid[1])), 5, (255, 0,0 ), -1)

        distances = 0
        for node in self.stem:
            vizimg, dist = node.step(stemGrad, vizimg, color = (0, 0, 255))
            distances += dist

        return vizimg, distances

    def isConverged(self, gradimg, stemGrad, vizimg):
        i = 0
        # scaleAndShow(vizimg, waitkey=0)
        distances = []
        while True:
            self.disp = 0

            t = vizimg.copy()
            for _ in range(10):
                t, distance = self.step(gradimg, stemGrad, t)

            
            distances.append(distance)            
            if len(distances) > 30:
                distances.pop(0)
            
            # print(np.mean(distances))
                if np.mean(distances) < 70:
                    break
            i += 1
            scaleAndShow(t, waitkey=1)

        return t

            
        



    