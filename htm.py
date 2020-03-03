import numpy as np

class SpatialPooler:
    def __init__(self, input_size, columns, active_columns):
        self.input_size = input_size
        self.columns = columns
        self.active_columns = active_columns

        self.sparsity = self.active_columns / self.columns

        self.duty_cycle_momentum = 0.99

        self.permanence_threshold = 0.0
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.2

        self.activation = np.zeros(self.columns, dtype=np.bool)
        self.overlaps = np.zeros(self.columns, dtype=np.uint)
        self.duty_cycle = np.zeros(self.columns, dtype=np.float32)

        self.active = np.zeros(self.active_columns, dtype=np.uint)

        self.permanence = np.random.randn(self.columns, self.input_size)

    def run(self, input):
        weight = self.permanence > self.permanence_threshold
        self.overlaps = np.count_nonzero(input & weight, axis=1)
        
        boosting = np.exp(-self.duty_cycle / self.sparsity)
        sorted = (boosting * self.overlaps).argsort()
        self.active = sorted[-self.active_columns:]

        self.activation.fill(False)
        self.activation[self.active] = True

        self.duty_cycle *= self.duty_cycle_momentum
        self.duty_cycle[self.active] += 1.0 - self.duty_cycle_momentum

        # not canon
        self.permanence[self.active] += input * (self.permanence_increment - self.permanence_decrement) + self.permanence_decrement
        
class TemporalMemory:
    def __init__(self, columns, cells):
        self.columns = columns
        self.cells = cells

        self.cell_active = np.zeros((columns, cells), dtype=np.bool)
        self.cell_predictive = np.zeros_like(self.cell_active)

        self.segment_activation = np.zeros((columns, cells, 0), np.uint)

        self.synapse_target = np.zeros((columns, cells, 0), np.uint)
        self.synapse_permanence = np.zeros((columns, cells, 0), np.float32)

    def run(self, active_column):
        self.cell_active.fill(False)

        active_column_cell_predictive = self.cell_predictive[active_column]

        bursting = np.count_nonzero(active_column_cell_predictive, axis=1) == 0
        
        self.cell_active[active_column] = np.expand_dims(bursting, axis=1) | active_column_cell_predictive

        self.cell_predictive.fill(False)

        active_synapse_target = self.synapse_target[active_column]
        active_synapse_permanence = self.synapse_permanence[active_column]

class HierarchicalTemporalMemory:
    def __init__(self, input_size, columns, cells, active_columns=None):
        if active_columns is None:
            active_columns = int(columns * 0.02)

        self.spatial_pooler = SpatialPooler(input_size, columns, active_columns)
        self.temporal_memory = TemporalMemory(columns, cells)

    def run(self, input):
        self.spatial_pooler.run(input)
        self.temporal_memory.run(self.spatial_pooler.active)

if __name__ == '__main__':
    input = np.random.randn(10, 10) > 0
    htm = HierarchicalTemporalMemory(10, 10, 4, active_columns=1)

    for step in range(100):
        for i in range(10):
            htm.run(input[i])
            print(i, ':', end=' ')
            for j in range(10):
                print('o' if htm.spatial_pooler.activation[j] else ' ', end='')
            print()
