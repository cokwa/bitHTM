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

        self.cell_active = np.zeros((self.columns, self.cells), dtype=np.bool)
        self.cell_predictive = np.zeros_like(self.cell_active)
        self.cell_segments = np.zeros((self.columns, self.cells), dtype=np.long)

        self.segment_capacity = 1
        self.segment_index = np.arange(self.cells * self.segment_capacity, dtype=np.long).reshape(1, self.cells, self.segment_capacity)
        self.segment_activation = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.int)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.segment_active = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.bool)
        self.segment_matching = np.zeros_like(self.segment_active)
        self.segment_synapses = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.long)

        self.cell_synapse_capacity = 0
        self.cell_synapse_cell = np.zeros((self.columns, self.cells, 0), dtype=np.long)

        self.segment_synapse_capacity = 1
        self.segment_synapse_cell = np.zeros((self.columns, self.cells, self.segment_capacity, self.segment_synapse_capacity), dtype=np.long)
        self.segment_synapse_permanence = np.zeros((self.columns, self.cells, self.segment_capacity, self.segment_synapse_capacity), dtype=np.float32)

        self.prev_target_cell = np.zeros(0, dtype=np.long)

    def run(self, active_column):
        cell_predictive = self.cell_predictive[active_column]
        column_bursting = ~np.any(cell_predictive, axis=1)

        segment_potential = self.segment_potential[active_column].reshape(len(active_column), -1)
        column_best_matching_segment = np.argmax(segment_potential, axis=1)
        column_least_used_cell = np.argmin(self.cell_segments[active_column], axis=1)
        column_grow_segment = segment_potential[(np.arange(len(active_column), dtype=np.long), column_best_matching_segment)] == 0
        segment_learning = self.segment_active[active_column] | ((self.segment_index == column_best_matching_segment[:, None, None]) & (column_bursting & ~column_grow_segment)[:, None, None])
        segment_growing_column = np.nonzero(column_grow_segment)[0]
        segment_growing_cell = column_least_used_cell[segment_growing_column]
        learning_segment = np.nonzero(segment_learning)
        punished_segment = np.nonzero(self.segment_active.reshape(-1, self.segment_capacity)[self.prev_target_cell])
        learning_segment = (active_column[learning_segment[0]], learning_segment[1], learning_segment[2])
        punished_segment = (self.prev_target_cell[punished_segment[0]], punished_segment[1])
        
        # TODO: what about invalid synapses?
        learning_segment_synapse_cell = self.segment_synapse_cell[learning_segment]
        punished_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_capacity, self.segment_synapse_capacity)[punished_segment]
        self.segment_synapse_permanence[learning_segment] += self.cell_active.reshape(-1)[learning_segment_synapse_cell] * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
        self.segment_synapse_permanence.reshape(-1, self.segment_capacity, self.segment_synapse_capacity)[punished_segment] -= self.cell_active.reshape(-1)[punished_segment_synapse_cell] * self.permanence_punishment

        segment_growing_column = active_column[segment_growing_column]
        cell_new_segment = self.cell_segments[(segment_growing_column, segment_growing_cell)]
        learning_segment = ( np.concatenate([learning_segment[0], segment_growing_column]),
                             np.concatenate([learning_segment[1], segment_growing_cell]),
                             np.concatenate([learning_segment[2], cell_new_segment]) )

        max_segments = np.max(cell_new_segment) + 1 if len(cell_new_segment) > 0 else 0
        if max_segments > self.segment_capacity:
            segment_capacity = max_segments
            self.segment_index = np.arange(self.cells * segment_capacity, dtype=np.long).reshape(1, self.cells, segment_capacity)
            self.segment_activation = np.zeros((self.columns, self.cells, segment_capacity), dtype=np.int)
            self.segment_potential = np.zeros_like(self.segment_activation)

            segment_synapses = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.long)
            segment_synapse_cell = np.zeros((self.columns, self.cells, segment_capacity, self.segment_synapse_capacity), dtype=np.long)
            segment_synapse_permanence = np.zeros((self.columns, self.cells, segment_capacity, self.segment_synapse_capacity), dtype=np.float32)
            segment_synapses[:, :, :self.segment_capacity] = self.segment_synapses
            segment_synapse_cell[:, :, :self.segment_capacity, :] = self.segment_synapse_cell
            segment_synapse_permanence[:, :, :self.segment_capacity, :] = self.segment_synapse_permanence
            self.segment_synapses = segment_synapses
            self.segment_synapse_cell = segment_synapse_cell
            self.segment_synapse_permanence = segment_synapse_permanence
            self.segment_capacity = segment_capacity

        self.cell_segments[(segment_growing_column, segment_growing_cell)] += 1

        learning_segment_new_synapses = self.synapse_sample_size - self.segment_potential[learning_segment]
        # exclude already connected cells

        max_synapses = np.max(self.segment_synapses[learning_segment] + learning_segment_new_synapses) if len(learning_segment[0]) > 0 else 0
        if max_synapses > self.segment_synapse_capacity:
            segment_synapse_capacity = max_synapses
            
            self.segment_synapse_capacity = segment_synapse_capacity

        self.segment_synapses[learning_segment] += learning_segment_new_synapses

        cell_active = cell_predictive | column_bursting[:, None]
        self.cell_active[:, :] = False
        self.cell_active[active_column] = cell_active

        active_cell = np.nonzero(cell_active)
        active_cell = (active_column[active_cell[0]], active_cell[1])

        cell_targeted = np.zeros(self.columns * self.cells, dtype=np.bool)
        cell_targeted[self.cell_synapse_cell[active_cell]] = True # TODO: what about invalid synapses? - easy
        target_cell = np.nonzero(cell_targeted)[0]
        
        segment_synapse_target = self.segment_synapse_cell.reshape(-1, self.segment_capacity, self.segment_synapse_capacity)[target_cell]
        segment_synapse_permanence = self.segment_synapse_permanence.reshape(-1, self.segment_capacity, self.segment_synapse_capacity)[target_cell]
        segment_synapse_weight = segment_synapse_permanence > self.permanence_threshold

        self.segment_activation[:, :, :] = 0
        self.segment_potential[:, :, :] = 0
        self.segment_activation.reshape(-1, self.segment_capacity)[target_cell] = np.sum(segment_synapse_target & segment_synapse_weight, axis=2)
        self.segment_potential.reshape(-1, self.segment_capacity)[target_cell] = np.sum(segment_synapse_target, axis=2)
        self.segment_active = self.segment_activation >= self.segment_active_threshold
        self.segment_matching = self.segment_potential >= self.segment_matching_threshold
        self.cell_predictive = np.any(self.segment_activation, axis=2)

        self.prev_target_cell = target_cell

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
