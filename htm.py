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
    def __init__(self, columns, active_columns, cells):
        self.columns = columns
        self.active_columns = active_columns
        self.cells = cells

        self.segment_activation_threshold = 15
        self.segment_potential_threshold = 15

        self.permanence_initial = 0.0
        self.permanence_threshold = 0.5
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.2

        self.active_column_cell_index = np.arange(self.active_columns * self.cells).reshape(self.active_columns, self.cells) % self.cells

        self.cell_active = np.zeros((self.columns, self.cells), dtype=np.bool)
        self.cell_predictive = np.zeros_like(self.cell_active)
        self.cell_winner = np.zeros_like(self.cell_active)

        self.segments = np.zeros((self.columns, self.cells), dtype=np.long)
        self.active_column_segment_index = np.zeros((self.active_columns, self.cells, 1), dtype=np.uint)
        self.segment_activation = np.zeros((self.columns, self.cells, 1), dtype=np.uint)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.segment_active = np.zeros((self.columns, self.cells, 1), dtype=np.bool)
        self.segment_matching = np.zeros_like(self.segment_active)

        self.synapses = np.zeros((self.columns, self.cells, 1), dtype=np.long)
        self.synapse_target = np.zeros((self.columns, self.cells, 1, 0), dtype=np.uint)
        self.synapse_permanence = np.zeros((self.columns, self.cells, 1, 0), dtype=np.float32)

    def run(self, active_column):
        self.cell_active.fill(False)

        active_column_cell_predictive = self.cell_predictive[active_column]
        bursting = ~np.any(active_column_cell_predictive, axis=1)[:, None]
        self.cell_active[active_column] = bursting | active_column_cell_predictive
        
        active_column_segments = self.segments[active_column]
        active_column_segment_matching = self.segment_matching[active_column]

        least_used_cell = np.argmin(active_column_segments, axis=1)[:, None] == self.active_column_cell_index
        self.cell_winner[active_column] = (least_used_cell & bursting) | active_column_cell_predictive

        cell_new_segment = ~np.any(active_column_segment_matching, axis=(1, 2))[:, None] & least_used_cell
        segment_new = cell_new_segment[:, :, None] & (active_column_segments[:, :, None] == self.active_column_segment_index)
        self.segments[active_column] += cell_new_segment

        max_segments = np.max(self.segments[active_column])
        if max_segments >= self.synapses.shape[2]:
            self.reserve_segment(2 ** int(np.ceil(np.log2(max_segments + 1))))

        segment_learning = active_column_segment_matching | segment_new

        #print(segment_new)

        synapse_target_column = self.synapse_target // self.cells
        synapse_target_cell = self.synapse_target % self.cells
        target_cell_active = self.cell_active[(synapse_target_column, synapse_target_cell)]

        synapse_weight = self.synapse_permanence > self.permanence_threshold
        synapse_valid = self.synapse_permanence >= 0.0
        self.segment_activation = np.count_nonzero(target_cell_active & synapse_weight, axis=3)
        self.segment_potential = np.count_nonzero(target_cell_active & synapse_valid, axis=3)
        self.segment_active = self.segment_activation > self.segment_activation_threshold
        self.segment_matching = self.segment_potential > self.segment_potential_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

    def reserve_segment(self, capacity):
        self.active_column_segment_index = np.arange(self.active_columns * self.cells * capacity).reshape(self.active_columns, self.cells, capacity) % capacity

        new_synapses = np.zeros((self.columns, self.cells, capacity), dtype=np.long)
        new_synapses[:, :, :self.synapses.shape[2]] = self.synapses
        self.synapses = new_synapses

        new_synapse_target = np.zeros((self.columns, self.cells, capacity, self.synapse_target.shape[3]), dtype=np.uint)
        new_synapse_target[:, :, :self.synapse_target.shape[2], :] = self.synapse_target
        self.synapse_target = new_synapse_target

        new_synapse_permanence = np.zeros((self.columns, self.cells, capacity, self.synapse_permanence.shape[3]), dtype=np.uint)
        new_synapse_permanence[:, :, :self.synapse_permanence.shape[2], :] = self.synapse_permanence
        self.synapse_permanence = new_synapse_permanence

class HierarchicalTemporalMemory:
    def __init__(self, input_size, columns, cells, active_columns=None):
        if active_columns is None:
            active_columns = int(columns * 0.02)

        self.spatial_pooler = SpatialPooler(input_size, columns, active_columns)
        self.temporal_memory = TemporalMemory(columns, active_columns, cells)

    def run(self, input):
        self.spatial_pooler.run(input)
        self.temporal_memory.run(self.spatial_pooler.active)

if __name__ == '__main__':
    input = np.random.randn(10, 10) > 0
    htm = HierarchicalTemporalMemory(10, 20, 4, active_columns=2)

    for step in range(100):
        for i in range(10):
            htm.run(input[i])
            print(i, ':', end=' ')
            for j in range(20):
                print('o' if htm.spatial_pooler.activation[j] else ' ', end='')
            print()
