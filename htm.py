import numpy as np

class SpatialPooler:
    def __init__(self, input_size, columns, active_columns):
        self.input_size = input_size
        self.columns = columns
        self.active_columns = active_columns

        self.sparsity = self.active_columns / self.columns

        self.boosting_intensity = 0.1
        self.duty_cycle_inertia = 0.99

        self.permanence_threshold = 0.0
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.2

        self.activation = np.zeros(self.columns, dtype=np.bool)
        self.overlaps = np.zeros(self.columns, dtype=np.long)
        self.duty_cycle = np.zeros(self.columns, dtype=np.float32)

        self.active = np.zeros(self.active_columns, dtype=np.long)

        self.permanence = np.random.randn(self.columns, self.input_size)

    def run(self, input):
        weight = self.permanence > self.permanence_threshold
        self.overlaps = np.sum(input & weight, axis=1)
        
        boosting = np.exp(self.boosting_intensity * -self.duty_cycle / self.sparsity)
        sorted = (boosting * self.overlaps).argsort()
        self.active = sorted[-self.active_columns:]

        self.activation[:] = False
        self.activation[self.active] = True

        self.duty_cycle *= self.duty_cycle_inertia
        self.duty_cycle[self.active] += 1.0 - self.duty_cycle_inertia

        self.permanence[self.active] += input * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
        
class TemporalMemory:
    def __init__(self, columns, cells):
        self.columns = columns
        self.cells = cells

        self.segment_active_threshold = 10
        self.segment_matching_threshold = 10

        self.synapse_sample_size = 20

        self.permanence_invalid = -1.0
        self.permanence_initial = 0.01
        self.permanence_threshold = 0.5
        self.permanence_increment = 0.3
        self.permanence_decrement = 0.05
        self.permanence_punishment = 0.01

        self.cell_segments = np.zeros((columns, cells), dtype=np.long)
        self.cell_active = np.zeros((columns, cells), dtype=np.bool)
        self.cell_predictive = np.zeros_like(self.cell_active)

        self.segments_capacity = 0
        self.segment_activation = np.zeros((columns, cells, 0), dtype=np.int)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.segment_active = np.zeros((columns, cells, 0), dtype=np.bool)
        self.segment_matching = np.zeros_like(self.segment_active)

        self.cell_synapse_capacity = 0
        self.cell_synapse_target = np.zeros((columns, cells, 0), dtype=np.long)

        self.segment_synapse_capacity = 0
        self.segment_synapse_target = np.zeros((columns, cells, 0, 0), dtype=np.long)
        self.segment_synapse_permanence = np.zeros((columns, cells, 0, 0))

    def run(self, active_column):
        cell_predictive = self.cell_predictive[active_column]
        cell_bursting = ~np.any(cell_predictive, axis=1)[:, None]
        cell_active = cell_predictive | cell_bursting
        self.cell_active[:, :] = False
        self.cell_active[active_column] = cell_active
        
        active_cell = np.nonzero(cell_active)
        active_cell = (active_column[active_cell[0]], active_cell[1])
        target_segment, target_segment_active_cells = np.unique(self.cell_synapse_target[active_cell], return_counts=True)
        valid_target_segment = target_segment[target_segment_active_cells >= min(self.segment_active_threshold, self.segment_matching_threshold)]
        segment_synapse_target = self.segment_synapse_target.reshape(-1, self.segment_synapse_capacity)[valid_target_segment]
        segment_synapse_permanence = self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[valid_target_segment]
        self.segment_activation[:, :, :] = 0
        self.segment_activation.reshape(-1, self.segment_synapse_capacity)[valid_target_segment] = np.sum(segment_synapse_target, axis=1)

    @staticmethod
    def get_exponential_capacity(capacity):
        return 2 ** int(np.ceil(np.log2(capacity)))

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
    input = np.random.randn(10, 1000) > 1.0
    htm = HierarchicalTemporalMemory(1000, 2048, 32)

    import time

    prev_time = time.time()

    for epoch in range(100):
        for i in range(len(input)):
            htm.run(input[i])
            print('epoch {}, pattern {}: correctly predicted columns: {}'.format(epoch, i, np.sum(np.any(htm.temporal_memory.cell_active & ~np.all(htm.temporal_memory.cell_active, axis=1)[:, None], axis=1))))

    print('{}s'.format(time.time() - prev_time))
