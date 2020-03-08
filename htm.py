import numpy as np

class SpatialPooler:
    def __init__(self, input_size, columns, active_columns):
        self.input_size = input_size
        self.columns = columns
        self.active_columns = active_columns

        self.sparsity = self.active_columns / self.columns

        self.boosting_intensity = 0.1
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
        
        boosting = np.exp(self.boosting_intensity * -self.duty_cycle / self.sparsity)
        sorted = (boosting * self.overlaps).argsort()
        self.active = sorted[-self.active_columns:]

        # activation, duty cycle and permanence indexing is only for cpu threshold or mask for gpu instead

        self.activation.fill(False)
        self.activation[self.active] = True

        self.duty_cycle *= self.duty_cycle_momentum
        self.duty_cycle[self.active] += 1.0 - self.duty_cycle_momentum

        # not canon
        self.permanence[self.active] += input * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
        
class TemporalMemory:
    def __init__(self, columns, cells):
        self.columns = columns
        self.cells = cells

        self.segment_activation_threshold = 10
        self.segment_potential_threshold = 10

        self.synapse_sample_size = 20

        self.permanence_invalid = -1.0
        self.permanence_initial = 0.01
        self.permanence_threshold = 0.5
        self.permanence_increment = 0.3
        self.permanence_decrement = 0.4
        self.permanence_punishment = 0.1

        self.cell_index = np.arange(self.cells).reshape(1, self.cells)
        self.cell_active = np.zeros((self.columns, self.cells), dtype=np.bool)
        self.cell_predictive = np.zeros_like(self.cell_active)
        self.cell_winner = np.zeros_like(self.cell_active)
        
        self.cell_prev_active = np.zeros_like(self.cell_active)
        self.cell_prev_winner = np.zeros_like(self.cell_active)

        self.segment_capacity = 1
        self.segments = np.zeros((self.columns, self.cells), dtype=np.uint)
        self.segment_index_in_cell = np.zeros((1, 1, 1), dtype=np.uint)
        self.segment_index_in_column = np.arange(self.cells, dtype=np.uint).reshape(1, self.cells, 1)
        self.segment_index = np.arange(self.columns * self.cells, dtype=np.uint).reshape(self.columns, self.cells, 1)
        self.segment_activation = np.zeros((self.columns, self.cells, 1), dtype=np.uint)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.segment_active = np.zeros((self.columns, self.cells, 1), dtype=np.bool)
        self.segment_matching = np.zeros_like(self.segment_active)
        self.segment_learning = np.zeros_like(self.segment_active)
        
        self.synapse_capacity = 0
        self.synapses = np.zeros((self.columns, self.cells, 1), dtype=np.uint)
        self.synapse_target = np.zeros((self.columns, self.cells, 1, 0), dtype=np.uint)
        self.synapse_permanence = np.zeros((self.columns, self.cells, 1, 0), dtype=np.float32)

    def run(self, column_active):
        self.cell_prev_active = self.cell_active
        self.cell_prev_winner = self.cell_winner

        cell_column_active = column_active[:, None]
        cell_column_bursting = ~np.any(self.cell_predictive, axis=1)[:, None]
        self.cell_active = (cell_column_bursting | self.cell_predictive) & cell_column_active
        
        column_segment_matching = np.any(self.segment_matching, axis=(1, 2))
        cell_least_used = np.argmin(self.segments, axis=1)[:, None] == self.cell_index
        cell_new_segment = ~column_segment_matching[:, None] & cell_least_used & cell_column_active
        segment_new = cell_new_segment[:, :, None] & (self.segments[:, :, None] == self.segment_index_in_cell)
        self.segments += cell_new_segment

        segment_best_matching = column_segment_matching[:, None, None] & (np.argmax(self.segment_potential.reshape(self.columns, -1), axis=1)[:, None, None] == self.segment_index_in_column)
        self.segment_learning = (self.segment_active | segment_best_matching | segment_new) & cell_column_active[:, :, None]
        self.cell_winner = np.any(self.segment_learning, axis=2)

        # TODO: move this to a better place
        max_segments = np.max(self.segments)
        if max_segments >= self.segment_capacity:
            self.reserve_segment(TemporalMemory.get_exponential_capacity(max_segments + 1))
            
        synapse_valid = self.synapse_permanence >= 0.0
        synapse_target_active = self.cell_active.reshape(-1)[self.synapse_target] & synapse_valid
        synapse_target_prev_active = self.cell_prev_active.reshape(-1)[self.synapse_target] & synapse_valid

        synapse_learning = self.segment_learning[:, :, :, None]
        synapse_punished = (self.cell_predictive & ~cell_column_active)[:, :, None, None] & synapse_target_prev_active
        learning = synapse_learning * (np.float32(synapse_target_prev_active) * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement)
        punishment = np.float32(synapse_punished) * -self.permanence_punishment
        self.synapse_permanence += learning + punishment

        learning_segment = np.nonzero(self.segment_learning)
        new_synapses = np.maximum(self.synapse_sample_size - self.segment_potential[learning_segment], 0)

        synapse_weight = self.synapse_permanence > self.permanence_threshold
        self.segment_activation = np.count_nonzero(synapse_target_active & synapse_weight, axis=3)
        self.segment_potential = np.count_nonzero(synapse_target_active, axis=3)
        self.segment_active = self.segment_activation >= self.segment_activation_threshold
        self.segment_matching = self.segment_potential >= self.segment_potential_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

        cur_synapse_valid = synapse_valid[learning_segment]
        cur_synapse_target = self.synapse_target[learning_segment][cur_synapse_valid]
        cell_available = np.ones((len(learning_segment[0]), self.columns, self.cells), dtype=np.bool)
        cell_available.reshape(-1, self.columns * self.cells)[(np.nonzero(cur_synapse_valid)[0], cur_synapse_target)] = False
        cell_available_winner = self.cell_prev_winner[None, :, :] & cell_available

        available_winners = np.count_nonzero(cell_available_winner, axis=(1, 2))
        new_synapses = np.minimum(new_synapses, available_winners)

        cur_synapses = self.synapses[learning_segment]
        cur_synapses_start = np.max(cur_synapses)
        cur_synapses_end = cur_synapses_start + np.max(new_synapses)
        if cur_synapses_end > self.synapse_capacity:
            self.reserve_synapse(TemporalMemory.get_exponential_capacity(cur_synapses_end))

        self.synapses[learning_segment] = cur_synapses_end
        '''
        max_available_winners = np.max(available_winners)
        available_winner_cell = np.nonzero(cell_available_winner)
        learning_segment_index = np.arange(len(available_winner_cell[0]))[None, :]
        learning_segment_mask = available_winner_cell[0][None, :] == np.arange(len(learning_segment[0]))[:, None]
        maxed_mask_target_index = len(available_winner_cell[0]) - (len(available_winner_cell[0]) - learning_segment_index) * learning_segment_mask
        zeroed_mask_target_index = learning_segment_index * learning_segment_mask
        synapse_taret_offset = np.min(maxed_mask_target_index, axis=1) if np.all(maxed_mask_target_index.shape) else np.zeros((len(learning_segment[0])), dtype=np.uint)
        synapse_target_last = np.max(zeroed_mask_target_index, axis=1) if np.all(zeroed_mask_target_index.shape) else np.zeros((len(learning_segment[0])), dtype=np.uint)
        new_synapse_target = available_winner_cell[1][(synapse_taret_offset[:, None] + np.arange(max_available_winners)[None, :]) % len(available_winner_cell[0])]
        new_synapse_valid = (np.arange(max_available_winners)[None, :] <= synapse_target_last[:, None])
        #shuffled_index = np.random.permutation(new_synapse_target.shape[0] * new_synapse_target.shape[1]).reshape(new_synapse_target.shape) % new_synapse_target.shape[1]
        #new_synapse_target = new_synapse_target[:, shuffled_index]
        #new_synapse_valid = new_synapse_valid[:, shuffled_index]
        new_synapse_target = new_synapse_target[:, :cur_synapses_end - cur_synapses_start]
        new_synapse_valid = new_synapse_valid[:, :cur_synapses_end - cur_synapses_start]
        '''
        # HACK: very inefficient, just find a better way to pick valid indices while retaining the shape
        new_synapse_target = np.arange(np.prod(cell_available_winner.shape)).reshape(cell_available_winner.shape[0], -1) % np.prod(cell_available_winner.shape[1:])
        new_synapse_target.reshape(*cell_available_winner.shape)[~cell_available_winner] = -1
        new_synapse_target[:, ::-1].sort()
        new_synapse_target = new_synapse_target[:, :cur_synapses_end - cur_synapses_start]
        #np.apply_along_axis(np.random.shuffle, 1, new_synapse_target)
        new_synapse_valid = new_synapse_target >= 0
        new_synapse_target[~new_synapse_valid] = 0
        
        self.synapse_target[:, :, :, cur_synapses_start:cur_synapses_end][learning_segment] = new_synapse_target
        self.synapse_permanence[:, :, :, cur_synapses_start:cur_synapses_end][learning_segment] = new_synapse_valid * (self.permanence_initial - self.permanence_invalid) + self.permanence_invalid

    @staticmethod
    def get_exponential_capacity(capacity):
        return 2 ** int(np.ceil(np.log2(capacity)))

    def reserve_segment(self, capacity):
        self.segment_index_in_cell = np.arange(capacity).reshape(1, 1, capacity)
        self.segment_index_in_column = np.arange(self.cells * capacity).reshape(1, self.cells, capacity)

        new_segment_potential = np.zeros((self.columns, self.cells, capacity), dtype=np.bool)
        new_segment_potential[:, :, :self.segment_potential.shape[2]] = self.segment_potential
        self.segment_potential = new_segment_potential

        new_segment_learning = np.zeros((self.columns, self.cells, capacity), dtype=np.bool)
        new_segment_learning[:, :, :self.segment_learning.shape[2]] = self.segment_learning
        self.segment_learning = new_segment_learning

        new_synapses = np.zeros((self.columns, self.cells, capacity), dtype=np.long)
        new_synapses[:, :, :self.synapses.shape[2]] = self.synapses
        self.synapses = new_synapses

        new_synapse_target = np.zeros((self.columns, self.cells, capacity, self.synapse_target.shape[3]), dtype=np.uint)
        new_synapse_target[:, :, :self.synapse_target.shape[2], :] = self.synapse_target
        self.synapse_target = new_synapse_target

        new_synapse_permanence = np.full((self.columns, self.cells, capacity, self.synapse_permanence.shape[3]), self.permanence_invalid, dtype=np.float32)
        new_synapse_permanence[:, :, :self.synapse_permanence.shape[2], :] = self.synapse_permanence
        self.synapse_permanence = new_synapse_permanence
        
        self.segment_capacity = capacity

    def reserve_synapse(self, capacity):
        new_synapse_target = np.zeros((self.columns, self.cells, self.synapse_target.shape[2], capacity), dtype=np.uint)
        new_synapse_target[:, :, :, :self.synapse_target.shape[3]] = self.synapse_target
        self.synapse_target = new_synapse_target

        new_synapse_permanence = np.full((self.columns, self.cells, self.synapse_permanence.shape[2], capacity), self.permanence_invalid, dtype=np.float32)
        new_synapse_permanence[:, :, :, :self.synapse_permanence.shape[3]] = self.synapse_permanence
        self.synapse_permanence = new_synapse_permanence
        
        self.synapse_capacity = capacity

class HierarchicalTemporalMemory:
    def __init__(self, input_size, columns, cells, active_columns=None):
        if active_columns is None:
            active_columns = int(columns * 0.02)

        self.spatial_pooler = SpatialPooler(input_size, columns, active_columns)
        self.temporal_memory = TemporalMemory(columns, cells)

    def run(self, input):
        self.spatial_pooler.run(input)
        self.temporal_memory.run(self.spatial_pooler.activation)

if __name__ == '__main__':
    input = np.random.randn(10, 1000) > 1.0
    htm = HierarchicalTemporalMemory(1000, 2048, 32)

    import time

    prev_time = time.time()

    for epoch in range(100):
        for i in range(len(input)):
            htm.run(input[i])
            print('epoch {}, pattern {}: correctly predicted columns: {}'.format(epoch, i, np.count_nonzero(np.any(htm.temporal_memory.cell_active & ~np.all(htm.temporal_memory.cell_active, axis=1)[:, None], axis=1))))

    print('{}s'.format(time.time() - prev_time))
