import numpy as np

class SpatialPooler:
    def __init__(self, input_size, column_size, active_column_size):
        self.input_size = input_size
        self.column_size = column_size
        self.active_column_size = active_column_size

        self.sparsity = self.active_column_size / self.column_size

        self.duty_cycle_momentum = 0.99

        self.permanence_threshold = 0.0
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.2

        self.activation = np.zeros(self.column_size, dtype=np.bool)
        self.overlaps = np.zeros(self.column_size, dtype=np.uint)
        self.duty_cycle = np.zeros(self.column_size, dtype=np.float32)

        self.permanence = np.random.randn(self.column_size, self.input_size)

    def run(self, input):
        weight = self.permanence > self.permanence_threshold
        self.overlaps = np.count_nonzero(input & weight, axis=1)
        
        boosting = np.exp(-self.duty_cycle / self.sparsity)
        sorted = (boosting * self.overlaps).argsort()
        self.active = sorted[-self.active_column_size:]

        self.activation = np.zeros(self.column_size, dtype=np.bool)
        self.activation[self.active] = True

        self.duty_cycle *= self.duty_cycle_momentum
        self.duty_cycle[self.active] += 1.0 - self.duty_cycle_momentum

        # not canon
        self.permanence[self.active] += input * (self.permanence_increment - self.permanence_decrement) + self.permanence_decrement
        
if __name__ == '__main__':
    input = np.random.randn(10, 10) > 0
    sp = SpatialPooler(10, 10, 1)

    for step in range(100):
        for i in range(10):
            sp.run(input[i])
            print(i, ':', end=' ')
            for j in range(10):
                print('o' if sp.activation[j] else ' ', end='')
            print()
