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
        rope[i].coeffs = [1, 1]


    if gradation:
        hwidth = len(rope)/2.
        scalea = 5
        scaleb = 2
        for i in range(len(rope)):
                
            ca = (abs(hwidth - i)/hwidth)  * scalea + 1
            cb = (1 - abs(hwidth - i)/hwidth)  * scaleb + 1
            rope[i].coeffs = [ca, cb]

        rope[0].coeffs = [1, 0]
        rope[-1].coeffs = [1, 0]


    return rope


def getCircle(centroid):
    leaf = []
    dist = 100
    centroid = np.array(centroid, dtype= int)
    forwardRope = getRope(centroid + np.array([-dist, -dist]), centroid  + np.array([dist, -dist]), slackstop = 2, slacksbtm = 2)
    backwardRope = getRope(centroid + np.array([dist, -dist + 1]), centroid  + np.array([dist, dist]), slackstop = 2, slacksbtm = 2)
    # r3 = getRope((880, 302), (880, 328), slackstop = 3, slacksbtm = 3)
    # r4 = getRope((730, 328), (722, 275), slackstop = 3, slacksbtm = 3)
    
    backwardRope[-1].connect(forwardRope[0])
    forwardRope[-1].connect(backwardRope[0])
    # r3[-1].connect(backwardRope[0])
    # r4[0].connect(backwardRope[-1])

    
    leaf += forwardRope
    leaf += backwardRope
    # leaf += r3

    return leaf


class Leaf:

    def __init__(self, ):
        

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



        self.leaf : List[Node] = getCircle((34, 351))
        
        # self.leaf += r4

        self.stem : List[Node] = getRope((51, 100), (200, 600), slackstop = 20, slacksbtm = 20, gradation = True)
        self.stem[0].connect(self.leaf[0])
        self.leaf[0].coeffs = [0, 1]



    def step(self, leafGrad, stemGrad, vizimg):
     
        centroid = Vector(0, 0)
        for node in self.leaf:
            centroid += node.vector

        centroid /= len(self.leaf)
        
        for node in self.leaf:
            normalForce = (node.vector - centroid) * 0.5
            vizimg = node.step(leafGrad, vizimg, normalForce,   color = (255, 0, 255))


        for node in self.stem:
            vizimg = node.step(stemGrad, vizimg, color = (255, 255, 0))

        cv2.circle(vizimg, (int(centroid[0]), int(centroid[1])), 5, (255, 0,0 ), -1)
        return vizimg

    def isConverged(self, gradimg, stemGrad, vizimg):
        i = 0
        scaleAndShow(vizimg, waitkey=0)
        while True:
            self.disp = 0

            t = vizimg.copy()
            t = self.step(gradimg, stemGrad, t)
            if i >= 0:
                wait = 1
            else:
                wait = 100

            i += 1
            scaleAndShow(t, waitkey=wait)


            
        



    