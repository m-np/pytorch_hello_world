"""Create a base class for NN layers."""
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self):
        pass

    def inference(self):
        pass
    