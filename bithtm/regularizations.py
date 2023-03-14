import numpy as np


class ExponentialBoosting:
    def __init__(
        self, output_dim, active_units,
        intensity=0.2, momentum=0.99
    ):
        self.density = active_units / output_dim
        self.intensity = intensity
        self.momentum = momentum

        self.duty_cycle = np.zeros(output_dim, dtype=np.float32)

    def process(self, input):
        factor = np.exp(self.intensity * -self.duty_cycle / self.density)
        return factor * input

    def update(self, active):
        self.duty_cycle *= self.momentum
        self.duty_cycle[active] += 1.0 - self.momentum


class GlobalInhibition:
    def __init__(self, active_units):
        self.active_units = active_units

    def process(self, input):
        return np.argpartition(input, -self.active_units)[-self.active_units:]