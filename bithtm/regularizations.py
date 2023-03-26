import numpy as np


class ExponentialBoosting:
    def __init__(
        self, output_dim, active_outputs,
        intensity=0.3, momentum=0.99
    ):
        self.density = active_outputs / output_dim
        self.intensity = intensity
        self.momentum = momentum

        self.duty_cycle = np.zeros(output_dim, dtype=np.float32)

    def process(self, output_activation):
        factor = np.exp(-(self.intensity / self.density) * self.duty_cycle)
        return factor * output_activation

    def update(self, active_output):
        self.duty_cycle *= self.momentum
        self.duty_cycle[active_output] += 1.0 - self.momentum


class GlobalInhibition:
    def __init__(self, active_outputs):
        self.active_outputs = active_outputs

    def process(self, output_activation):
        return np.argpartition(output_activation, -self.active_outputs)[-self.active_outputs:]


class LocalInhibition:
    def __init__(self, input_dim, output_dim, window_size, active_outputs_per_neighborhood=None, density=None, flattened=True):
        assert (active_outputs_per_neighborhood is None) ^ (density is None)
        if density is not None:
            raise NotImplementedError()

        assert type(input_dim) == type(output_dim) == type(window_size)
        if type(input_dim) == int:
            input_dim = (input_dim, )
            output_dim = (output_dim, )
            window_size = (window_size, )
        assert len(input_dim) == len(output_dim) == len(window_size)
        assert (np.array(window_size) <= np.array(input_dim)).all()

        stride = (np.array(input_dim) - np.array(window_size)) / (np.array(output_dim) - 1)
        assert (stride <= np.array(window_size)).all()

        self.active_outputs_per_neighborhood = active_outputs_per_neighborhood
        self.flattened = flattened
        
        # TODO
        self.neighborhood = np.random.randint(0, np.prod(output_dim), (np.prod(output_dim), 100))

    def process(self, output_activation, epsilon=1e-8):
        activation_threshold = np.partition(output_activation[self.neighborhood], -self.active_outputs_per_neighborhood, axis=1)[:, -self.active_outputs_per_neighborhood]
        active_output, = np.where(output_activation - activation_threshold > -epsilon)
        return active_output