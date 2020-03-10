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

        self.activation[:] = False
        self.activation[self.active] = True

        self.duty_cycle *= self.duty_cycle_momentum
        self.duty_cycle[self.active] += 1.0 - self.duty_cycle_momentum

        self.permanence[self.active] += input * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
        
class TemporalMemory:
    def __init__(self, columns, cells):
        self.columns = columns
        self.cells = cells

        self.segment_activation_threshold = 10
        self.segment_potential_threshold = 10

        self.segment_activation_threshold = 10

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
        self.cell_segments = np.zeros((self.columns, self.cells), dtype=np.long)
        self.cell_best_matching_segment = np.full((self.columns, self.cells), -1, dtype=np.long)

        self.cell_synapses = np.zeros((self.columns, self.cells), dtype=np.long)
        
        self.prev_winner_cell = np.zeros(0, dtype=np.long)
        self.winner_cell = np.zeros(0, dtype=np.long)

        self.cell_prev_active = np.zeros_like(self.cell_active)
        self.cell_prev_winner = np.zeros_like(self.cell_active)

        self.segment_capacity = 1
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

        #self.cell_index = np.arange(self.columns * self.cells)

        self.segments = 0
        self.segments_capacity = 0
        self.segment_index = np.arange(self.segments_capacity)
        self.segment_cell = np.full(self.segments_capacity, -1, dtype=np.long)
        self.segment_activation = np.zeros(self.segments_capacity, dtype=np.long)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.active_segment = np.zeros(0, dtype=np.long)

        self.synapses = 0
        self.synapses_capacity = 0
        self.synapse_segment = np.full((self.columns, self.cells, self.synapses_capacity), -1, dtype=np.long)
        self.synapse_permanence = np.full(self.synapse_segment.shape, self.permanence_initial, dtype=np.float32)

    def run(self, column_active, active_column):
        self.prev_winner_cell = self.winner_cell

        cell_predictive = self.cell_predictive[active_column]
        cell_column_bursting = ~np.any(cell_predictive, axis=1)[:, None]
        cell_active = cell_column_bursting | cell_predictive
        self.cell_active[:, :] = False
        self.cell_active[active_column] = cell_active

        cell_best_matching_segment = self.cell_best_matching_segment[active_column]
        cell_segment_matching = cell_best_matching_segment >= 0
        least_used_cell = np.argmin(self.cell_segments[active_column], axis=1)
        cell_new_segment = ~np.any(cell_segment_matching, axis=1)[:, None] & (self.cell_index == least_used_cell[:, None])
        cell_winner = cell_predictive | cell_segment_matching | cell_new_segment
        self.cell_winner[:, :] = False
        self.cell_winner[active_column] = cell_winner

        new_segment_cell = self.get_active_column_cell_index(active_column, np.nonzero(cell_new_segment))
        active_cell_offset = active_column[:, None] * self.cells
        prev_active_segment_cell = self.segment_cell[self.active_segment][None, :]
        correct_segment = self.active_segment[np.any((active_cell_offset <= prev_active_segment_cell) & (prev_active_segment_cell <= active_cell_offset + self.cells), axis=0)]
        new_segment = self.segments + np.arange(len(new_segment_cell))
        learning_segment = np.unique(np.concatenate([cell_best_matching_segment[cell_segment_matching].reshape(-1), correct_segment, new_segment]))
        
        self.winner_cell = self.get_active_column_cell_index(active_column, np.nonzero(cell_winner))
        
        self.prev_winner_cell = self.winner_cell # for debug only

        if len(self.prev_winner_cell) > 0:
            segments = self.segments + len(new_segment)
            if segments > self.segments_capacity:
                self.segments_capacity = TemporalMemory.get_exponential_capacity(segments)

                segment_cell = np.full(self.segments_capacity, -1, dtype=np.long)
                segment_cell[:self.segments] = self.segment_cell[:self.segments]
                self.segment_cell = segment_cell

                self.segment_potential = np.append(self.segment_potential, np.zeros(segments - self.segments, dtype=self.segment_potential.dtype))

            segment_cell[self.segments:segments] = new_segment_cell
            self.segments = segments
            
            candidate_cell_synapse_segment = self.synapse_segment.reshape(self.columns * self.cells, -1)[self.prev_winner_cell]
            segment_new_synapses = np.maximum(self.synapse_sample_size - self.segment_potential[learning_segment], 0)
            segment_cell_candidate = np.all(learning_segment[:, None, None] != candidate_cell_synapse_segment[None, :, :], axis=2)
            shuffled_index = np.arange(np.prod(segment_cell_candidate.shape)).reshape(segment_cell_candidate.shape) % segment_cell_candidate.shape[1]
            np.apply_along_axis(np.random.shuffle, 1, shuffled_index)
            segment_cell_candidate &= shuffled_index < segment_new_synapses[:, None]
            candidate_cells = np.nonzero(segment_cell_candidate)

            cell_synapses = self.cell_synapses.reshape(self.columns * self.cells, -1)[self.prev_winner_cell] + np.sum(segment_cell_candidate, axis=0)
            synapses = np.max(cell_synapses)
            if synapses > self.synapses_capacity:
                self.synapses_capacity = TemporalMemory.get_exponential_capacity(synapses)
                
                synapse_segment = np.full((self.columns, self.cells, self.synapses_capacity), -1, dtype=np.long)
                synapse_segment[:, :, :self.synapses] = self.synapse_segment[:, :, :self.synapses]
                self.synapse_segment = synapse_segment

                synapse_permanence = np.full(self.synapse_segment.shape, self.permanence_initial, dtype=np.float32)
                synapse_permanence[:, :, :self.synapses] = self.synapse_permanence[:, :, :self.synapses]
                self.synapse_permanence = synapse_permanence

            synapse_segment_cell = self.prev_winner_cell[candidate_cells[1]]

            print(np.unique(candidate_cells[1], return_inverse=True))

            synapse_segment_synapse = self.cell_synapses.reshape(-1)[synapse_segment_cell] + np.unique(candidate_cells[1], return_counts=True)[1]
            synapse_segment.reshape(self.columns * self.cells, -1)[(synapse_segment_cell, synapse_segment_synapse)] = learning_segment[candidate_cells[0]]
            self.cell_synapses.reshape(self.columns * self.cells, -1)[self.prev_winner_cell] = cell_synapses
            self.synapses = synapses

        active_cell = self.get_active_column_cell_index(active_column, np.nonzero(cell_active))
        active_synapse_segment = self.synapse_segment.reshape(self.columns * self.cells, -1)[active_cell].reshape(-1)
        active_synapse_weight = self.synapse_permanence.reshape(self.columns * self.cells, -1)[active_cell].reshape(-1) > self.permanence_threshold
        affected_segment = np.unique(active_synapse_segment[active_synapse_segment >= 0])
        segment_synapse_active = affected_segment[:, None] == active_synapse_segment[None, :] # can be optimized by np.unique
        segment_activation = np.sum(segment_synapse_active & active_synapse_weight[None, :], axis=1)
        segment_potential = np.sum(segment_synapse_active, axis=1)
        self.segment_activation[:] = 0
        self.segment_potential[:] = 0
        self.segment_activation[affected_segment] = segment_activation
        self.segment_potential[affected_segment] = segment_potential

        self.cell_predictive[:] = False
        self.cell_best_matching_segment[:] = -1
        self.active_segment = affected_segment[segment_activation >= self.segment_activation_threshold]
        matching_segment = affected_segment[segment_potential >= self.segment_activation_threshold]

        if matching_segment.size > 0:
            matching_segment_cell = self.segment_cell[matching_segment]
            affected_cell = np.unique(matching_segment_cell)
            cell_segment_matching = affected_cell[:, None] == matching_segment_cell[None, :] # can be optimized by np.unique
            cell_segment_activation = self.segment_activation[matching_segment][None, :] * cell_segment_matching
            cell_segment_potential = self.segment_potential[matching_segment][None, :] * cell_segment_matching
            cell_best_matching_segment = self.segment_index[matching_segment][np.argmax(cell_segment_potential, axis=1)]
            self.cell_predictive.reshape(-1)[affected_cell] = np.any(cell_segment_activation >= self.segment_activation_threshold, axis=1)
            self.cell_best_matching_segment.reshape(-1)[affected_cell] = cell_best_matching_segment

        quit()

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
        learning = synapse_learning * (synapse_target_prev_active.astype(np.float32) * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement)
        punishment = synapse_punished.astype(np.float32) * -self.permanence_punishment
        self.synapse_permanence += learning + punishment

        learning_segment = np.nonzero(self.segment_learning)
        new_synapses = np.maximum(self.synapse_sample_size - self.segment_potential[learning_segment], 0)

        synapse_weight = self.synapse_permanence > self.permanence_threshold
        self.segment_activation = np.count_nonzero(synapse_target_active & synapse_weight, axis=3)
        self.segment_potential = np.count_nonzero(synapse_target_active, axis=3)
        self.segment_active = self.segment_activation >= self.segment_activation_threshold
        self.segment_matching = self.segment_potential >= self.segment_potential_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

        learning_segments = len(learning_segment[0])
        cur_synapse_valid = synapse_valid[learning_segment]
        cur_synapse_target = self.synapse_target[learning_segment][cur_synapse_valid]
        cell_available = np.ones((learning_segments, self.columns * self.cells), dtype=np.bool)
        cell_available[(np.nonzero(cur_synapse_valid)[0], cur_synapse_target)] = False
        cell_available_winner = self.cell_prev_winner.reshape(1, -1) & cell_available

        available_winners = np.count_nonzero(cell_available_winner, axis=1)
        new_synapses = np.minimum(new_synapses, available_winners)

        cur_synapses = self.synapses[learning_segment]
        cur_synapses_start = np.max(cur_synapses)
        cur_synapses_end = cur_synapses_start + np.max(new_synapses)
        if cur_synapses_end > self.synapse_capacity:
            self.reserve_synapse(TemporalMemory.get_exponential_capacity(cur_synapses_end))

        self.synapses[learning_segment] = cur_synapses_end

        available_winner_cell = np.nonzero(cell_available_winner)
        new_synapse_target = (available_winner_cell[1][None, :] + 1) * (np.arange(learning_segments)[:, None] == available_winner_cell[0][None, :])
        new_synapse_target[:, ::-1].sort()
        new_synapse_target = new_synapse_target[:, :cur_synapses_end - cur_synapses_start]
        new_synapse_valid = new_synapse_target > 0
        new_synapse_target -= 1
        
        self.synapse_target[:, :, :, cur_synapses_start:cur_synapses_end][learning_segment] = new_synapse_target
        self.synapse_permanence[:, :, :, cur_synapses_start:cur_synapses_end][learning_segment] = new_synapse_valid * (self.permanence_initial - self.permanence_invalid) + self.permanence_invalid

    def get_active_column_cell_index(self, active_column, active_column_cell):
        return active_column[active_column_cell[0]] * self.cells + active_column_cell[1]

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
        self.temporal_memory.run(self.spatial_pooler.activation, self.spatial_pooler.active)

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
