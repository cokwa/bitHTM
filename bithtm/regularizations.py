import numpy as np


class ExponentialBoosting:
    def __init__(
        self, output_dim, active_outputs,
        intensity=0.2, momentum=0.99
    ):
        self.density = active_outputs / output_dim
        self.intensity = intensity
        self.momentum = momentum

        self.duty_cycle = np.zeros(output_dim, dtype=np.float32)

    def process(self, input_activation):
        factor = np.exp(-(self.intensity / self.density) * self.duty_cycle)
        return factor * input_activation

    def update(self, active_input):
        self.duty_cycle *= self.momentum
        self.duty_cycle[active_input] += 1.0 - self.momentum


class GlobalInhibition:
    def __init__(self, active_outputs):
        self.active_outputs = active_outputs

    def process(self, input_activation):
        return np.argpartition(input_activation, -self.active_outputs)[-self.active_outputs:]