

from typing import List
from utils import scaleAndShow
from vector import Vector
from node import Node
from node import dt
import cv2
import numpy as np
class Rope:
    def __init__(self, nodes : List[Node]):

        self.nodes = nodes

        hwidth = len(nodes)/2.
        scalea = 5
        scaleb = 2
        for i in range(len(nodes)):
            
            ca = (abs(hwidth - i)/hwidth)  * scalea + 1
            cb = (1 - abs(hwidth - i)/hwidth)  * scaleb + 1
            self.nodes[i].coeffs = [ca, cb]

        self.nodes[0].coeffs = [1, 0]
        self.nodes[-1].coeffs = [1, 0]
        self.disp = Vector(0, 0)

    def step(self, gradimg, vizimg):
        
        for node in self.nodes:
            vizimg = node.step(gradimg, vizimg)

        return vizimg


    def isConverged(self, image, vizimg):
        distances = []
        while True:
            self.disp = 0

            for i in range(15):
                t = vizimg.copy()
                t = self.step(image,t)
                for node in self.nodes:
                    self.disp += (node.computeForce(image) * dt).norm()

            # if i % == 0:
            scaleAndShow(t, waitkey=1)

            

            distances.append(self.disp)
            threshdist = abs(np.average(distances[::-1][:8]) - distances[-1])
            if len(distances) > 8 and (threshdist) < 5:
                break


        print('converged')

        return t

    

    def removeNode(self):
        dists = []
        for i in range(1, len(self.nodes) - 1):
            dist = (self.nodes[i].vector - self.nodes[i-1].vector).norm()
            dist += (self.nodes[i].vector - self.nodes[i+1].vector).norm()
            dists.append(dist)
        # ind = np.random.randint(1, len(self.nodes))

        ind = np.argmax(dists) + 1
        removenode = self.nodes[ind]
        prevnode = self.nodes[ind - 1]
        nextnode = self.nodes[ind + 1]
        removenode.disconnect(prevnode)
        removenode.disconnect(nextnode)
        self.nodes = self.nodes[:ind] + self.nodes[ind+1:]
        prevnode.connect(nextnode)
        

    def addNode(self):

        dists = []
        for i in range(1, len(self.nodes)):
            dist = (self.nodes[i].vector - self.nodes[i-1].vector).norm()
            dists.append(dist)

        ind = np.argmax(dists)
        endnode = self.nodes[ind]
        prevnode = self.nodes[ind + 1]
        endnode.disconnect(prevnode)
        newnode = Node((endnode.x + prevnode.x)/2, (endnode.y + prevnode.y)/2)
        newnode.connect(prevnode)
        endnode.connect(newnode)

        list1 = self.nodes[:ind + 1]
        list2 = self.nodes[ind + 1:]
        self.nodes = list1 + [newnode] + list2
        



def getRope(pt1, pt2, mask = None, slackstop = 50, slacksbtm = 50):

    rope = [Node(pt1[0], pt1[1])]
    
    for i in range(slackstop):
        rope.append(Node(pt1[0], pt1[1]))

    rope.append(Node(pt2[0], pt2[1]))
    
    for i in range(slacksbtm):
        rope.append(Node(pt2[0], pt2[1]))
    
    for i in range(1, len(rope)):
        rope[i].connect(rope[i-1])
    rope = Rope(rope)
    return rope