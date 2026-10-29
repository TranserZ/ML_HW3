import numpy as np


class GD:
    def __init__(self):
        pass

    def update(self, w, grad, lr):
        return w - lr * grad