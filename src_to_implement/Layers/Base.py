class BaseLayer:
    def __init__(self , weights = None):
        self.weights = weights
        self.testing_phase = False
        self.trainable = False
