

from typing import List
from node import Node
import numpy as np

from utils import scaleAndShow

class Model:


    def __init__(self, leaves):

        self.nodes : List[Node] = []

        for leaf in leaves:
            self.nodes += leaf.stem

    def step(self, vizimg, gradimg):
        distance = 0
        for node in self.nodes:
            vizimg, dist = node.step(gradimg, vizimg, color = (0, 0, 255))  
            distance += dist

        return vizimg, distance

        
    def converge(self, vizimg, gradimg):
        
        distances = []
        while True:

            for _ in range(10):
                t = vizimg.copy()
                t, dist = self.step(t, gradimg)

            distances.append(dist)

            if len(distances) > 30:
                distances.pop(0)
                if np.mean(distances) < 70:
                    break

            scaleAndShow(t, waitkey=1)

        return t