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
    def __init__(self, columns, active_columns, cells):
        self.columns = columns
        self.active_columns = active_columns
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
        
        '''
        self.synapse_capacity = 0
        self.synapses = np.zeros((self.columns, self.cells, 1), dtype=np.uint)
        self.synapse_target = np.zeros((self.columns, self.cells, 1, 0), dtype=np.uint)
        self.synapse_permanence = np.zeros((self.columns, self.cells, 1, 0), dtype=np.float32)
        '''

        self.synapse_capacity = 0
        self.synapses = np.zeros((self.columns, self.cells), dtype=np.uint)
        self.synapse_target = np.zeros((self.columns, self.cells, 0), dtype=np.uint)
        self.synapse_permanence = np.zeros((self.columns, self.cells, 0), dtype=np.float32)

    def run(self, column_active, active_column):
        self.cell_prev_active, self.cell_active = self.cell_active, self.cell_prev_active
        self.cell_prev_winner, self.cell_winner = self.cell_winner, self.cell_prev_winner

        cell_predictive = self.cell_predictive[active_column]
        cell_column_bursting = ~np.any(cell_predictive, axis=1)[:, None]
        cell_active = cell_column_bursting | cell_predictive
        self.cell_active.fill(0)
        self.cell_active[active_column] = cell_active

        segments = self.segments[active_column]
        segment_matching = self.segment_matching[active_column]
        column_segment_matching = np.any(segment_matching, axis=(1, 2))
        cell_least_used = np.argmin(segments, axis=1)[:, None] == self.cell_index
        cell_new_segment = ~column_segment_matching[:, None] & cell_least_used
        segment_new = cell_new_segment[:, :, None] & (segments[:, :, None] == self.segment_index_in_cell)
        self.segments[active_column] += cell_new_segment

        segment_active = self.segment_active[active_column]
        segment_potential = self.segment_potential[active_column]
        segment_best_matching = column_segment_matching[:, None, None] & (np.argmax(segment_potential.reshape(self.active_columns, -1), axis=1)[:, None, None] == self.segment_index_in_column)
        segment_learning = segment_active | segment_best_matching | segment_new
        self.segment_learning.fill(0)
        self.segment_learning[active_column] = segment_learning

        cell_winner = np.any(segment_learning, axis=2)
        self.cell_winner.fill(0)
        self.cell_winner[active_column] = cell_winner

        max_segments = np.max(segments)
        if max_segments >= self.segment_capacity:
            self.reserve_segment(TemporalMemory.get_exponential_capacity(max_segments + 1))

        active_cell = np.nonzero(cell_active)
        active_cell = (active_column[active_cell[0]], active_cell[1])
        synapse_permanence = self.synapse_permanence[active_cell]
        synapse_valid = synapse_permanence >= 0.0
        synapse_target = self.synapse_target[active_cell][synapse_valid]
        cell_target_segment = np.zeros((len(active_cell[0]), self.columns * self.cells * self.segment_capacity), dtype=np.bool)

        synapse_permanence = synapse_permanence[synapse_valid]
        synapse_weight = synapse_permanence > self.permanence_threshold
        synapse_valid_target = (np.arange(synapse_target.shape[0])[:, None], np.nonzero(synapse_valid)[0])

        cell_target_segment[synapse_valid_target] = True
        self.segment_potential = np.count_nonzero(cell_target_segment, axis=0).reshape((self.columns, self.cells, self.segment_capacity))

        cell_target_segment[synapse_valid_target] = synapse_weight
        self.segment_activation = np.count_nonzero(cell_target_segment, axis=0).reshape((self.columns, self.cells, self.segment_capacity))

        self.segment_active = self.segment_activation >= self.segment_activation_threshold
        self.segment_matching = self.segment_potential >= self.segment_potential_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

        print(max_segments, self.segment_capacity, self.segment_potential.shape)

        winner_cell = np.nonzero(cell_winner)
        winner_cell = (active_column[winner_cell[0]], winner_cell[1])
        synapse_permanence = self.synapse_permanence[winner_cell]
        synapse_valid = synapse_permanence >= 0.0
        synapse_target = self.synapse_target[winner_cell][synapse_valid]
        cell_candidate_segment = np.ones((len(winner_cell[0]), self.columns * self.cells * self.segment_capacity), dtype=np.bool)
        synapse_valid_target = (np.arange(synapse_target.shape[0])[:, None], np.nonzero(synapse_valid)[0])
        cell_candidate_segment[synapse_valid_target] = False
        cell_candidate_learning_segment = cell_candidate_segment.reshape((-1, self.columns, self.cells, self.segment_capacity))[:, active_column][:, segment_learning]
        
        new_synapses = np.minimum(np.maximum(self.synapse_sample_size - np.count_nonzero(~cell_candidate_segment, axis=1), 0), np.count_nonzero(cell_candidate_learning_segment, axis=1))

        cur_synapses = self.synapses[winner_cell]
        cur_synapses_start = np.max(cur_synapses)
        cur_synapses_end = cur_synapses_start + np.max(new_synapses)
        if cur_synapses_end > self.synapse_capacity:
            self.reserve_synapse(TemporalMemory.get_exponential_capacity(cur_synapses_end))


        quit()
        
        synapse_permanence = self.synapse_permanence[active_column][cell_active]
        synapse_valid = synapse_permanence >= 0.0
        synapse_target = self.synapse_target[active_column][cell_active][synapse_valid][None, None, None, :]
        synapse_permanence = synapse_permanence[synapse_valid][None, None, None, :]
        synapse_weight = synapse_permanence > self.permanence_threshold
        synapse_valid = True

        segment_targeted = self.segment_index[:, :, :, None] == synapse_target
        self.segment_activation = np.count_nonzero(segment_targeted, axis=3)
        self.segment_potential = np.count_nonzero(segment_targeted & synapse_weight, axis=3)
        self.segment_active = self.segment_activation >= self.segment_activation_threshold
        self.segment_matching = self.segment_potential >= self.segment_potential_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

        synapse_permanence = self.synapse_permanence[active_column][cell_winner]
        synapse_valid = synapse_permanence >= 0.0
        synapse_target = self.synapse_target[active_column][cell_winner][synapse_valid][None, None, None, :]
        synapse_valid = True
        
        cell_candidate 

        new_synapses = np.maximum(self.synapse_sample_size - segment_matching, 0)

        quit()
        




        
        
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

        synapse_target_column = self.synapse_target // self.cells
        synapse_target_cell = self.synapse_target % self.cells
        synapse_valid = self.synapse_permanence >= 0.0
        synapse_target_active = self.cell_active[(synapse_target_column, synapse_target_cell)] & synapse_valid
        synapse_target_prev_active = self.cell_prev_active[(synapse_target_column, synapse_target_cell)] & synapse_valid

        synapse_learning = self.segment_learning[:, :, :, None]
        synapse_punished = (self.cell_predictive & ~cell_column_active)[:, :, None, None] & synapse_target_prev_active
        learning = synapse_learning * (np.float32(synapse_target_prev_active) * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement)
        punishment = np.float32(synapse_punished) * -self.permanence_punishment
        self.synapse_permanence += learning + punishment

        new_synapses = np.max(self.synapse_sample_size - self.segment_matching[self.segment_learning], 0)

        synapse_weight = self.synapse_permanence > self.permanence_threshold
        self.segment_activation = np.count_nonzero(synapse_target_active & synapse_weight, axis=3)
        self.segment_potential = np.count_nonzero(synapse_target_active, axis=3)
        self.segment_active = self.segment_activation >= self.segment_activation_threshold
        self.segment_matching = self.segment_potential >= self.segment_potential_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

        cur_synapse_valid = synapse_valid[self.segment_learning]
        cur_synapse_target = self.synapse_target[self.segment_learning][cur_synapse_valid]
        cur_synapse_target_column = cur_synapse_target // self.cells
        cur_synapse_target_cell = cur_synapse_target % self.cells
        cell_available = np.ones((np.count_nonzero(self.segment_learning), self.columns, self.cells), dtype=np.bool)
        cell_available[(np.nonzero(cur_synapse_valid)[0], cur_synapse_target_column, cur_synapse_target_cell)] = False
        cell_available_winner = self.cell_prev_winner[None, :, :] & cell_available

        new_synapses = np.minimum(new_synapses, np.count_nonzero(cell_available_winner, axis=(1, 2)))

        cur_synapses = self.synapses[self.segment_learning]
        cur_synapses_start = np.max(cur_synapses)
        cur_synapses_end = cur_synapses_start + np.max(new_synapses)
        if cur_synapses_end > self.synapse_capacity:
            self.reserve_synapse(TemporalMemory.get_exponential_capacity(cur_synapses_end))

        self.synapses[self.segment_learning] = cur_synapses_end

        # HACK: very inefficient, just find a better way to pick valid indices while retaining the shape
        new_synapse_target = np.arange(np.prod(cell_available_winner.shape)).reshape(cell_available_winner.shape[0], -1) % np.prod(cell_available_winner.shape[1:])
        np.random.shuffle(new_synapse_target)
        new_synapse_target.reshape(*cell_available_winner.shape)[~cell_available_winner] = -1
        new_synapse_target[:, ::-1].sort()
        new_synapse_target[new_synapse_target < 0] = 0
        
        self.synapse_target[self.segment_learning, cur_synapses_start:cur_synapses_end] = new_synapse_target[:, :cur_synapses_end - cur_synapses_start]
        self.synapse_permanence[self.segment_learning, cur_synapses_start:cur_synapses_end] = (self.permanence_initial - self.permanence_invalid) + self.permanence_invalid
        
    @staticmethod
    def get_exponential_capacity(capacity):
        return 2 ** int(np.ceil(np.log2(capacity)))

    def reserve_segment(self, capacity):
        self.segment_index_in_cell = np.arange(capacity).reshape(1, 1, capacity)
        self.segment_index_in_column = np.arange(self.cells * capacity).reshape(1, self.cells, capacity)

        '''
        new_segment_matching = np.zeros((self.columns, self.cells, capacity), dtype=np.bool)
        new_segment_matching[:, :, :self.segment_matching.shape[2]] = self.segment_matching
        self.segment_matching = new_segment_matching

        new_segment_learning = np.zeros((self.columns, self.cells, capacity), dtype=np.bool)
        new_segment_learning[:, :, :self.segment_learning.shape[2]] = self.segment_learning
        self.segment_learning = new_segment_learning

        new_synapses = np.zeros((self.columns, self.cells, capacity), dtype=np.long)
        new_synapses[:, :, :self.synapses.shape[2]] = self.synapses
        self.synapses = new_synapses

        new_synapse_target = np.zeros((self.columns, self.cells, capacity, self.synapse_target.shape[3]), dtype=np.uint)
        new_synapse_target[:, :, :self.synapse_target.shape[2], :] = self.synapse_target
        self.synapse_target = new_synapse_target

        new_synapse_permanence = np.zeros((self.columns, self.cells, capacity, self.synapse_permanence.shape[3]), dtype=np.float32)
        new_synapse_permanence[:, :, :self.synapse_permanence.shape[2], :] = self.synapse_permanence
        self.synapse_permanence = new_synapse_permanence
        '''
        
        synapse_target_column = self.synapse_target // (self.cells * self.segment_capacity)
        synapse_target_cell = self.synapse_target // self.segment_capacity % self.cells
        synapse_target_segment = self.synapse_target % self.segment_capacity
        self.synapse_target = synapse_target_column * (self.cells * capacity) + synapse_target_cell * capacity + synapse_target_segment

        self.segment_capacity = capacity

    def reserve_synapse(self, capacity):
        '''
        self.synapse_capacity = capacity

        new_synapse_target = np.zeros((self.columns, self.cells, self.synapse_target.shape[2], capacity), dtype=np.uint)
        new_synapse_target[:, :, :, :self.synapse_target.shape[3]] = self.synapse_target
        self.synapse_target = new_synapse_target

        new_synapse_permanence = np.full((self.columns, self.cells, self.synapse_permanence.shape[2], capacity), self.permanence_invalid, dtype=np.float32)
        new_synapse_permanence[:, :, :, :self.synapse_permanence.shape[3]] = self.synapse_permanence
        self.synapse_permanence = new_synapse_permanence
        '''

        new_synapse_target = np.zeros((self.columns, self.cells, capacity), dtype=np.uint)
        new_synapse_target[:, :, :self.synapse_capacity] = self.synapse_target
        self.synapse_target = new_synapse_target

        new_synapse_permanence = np.full((self.columns, self.cells, capacity), self.permanence_invalid, dtype=np.float32)
        new_synapse_permanence[:, :, :self.synapse_capacity] = self.synapse_permanence
        self.synapse_permanence = new_synapse_permanence

        self.synapse_capacity = capacity

class HierarchicalTemporalMemory:
    def __init__(self, input_size, columns, cells, active_columns=None):
        if active_columns is None:
            active_columns = int(columns * 0.02)

        self.spatial_pooler = SpatialPooler(input_size, columns, active_columns)
        self.temporal_memory = TemporalMemory(columns, active_columns, cells)

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
            print('epoch {}, pattern {}: correctly predicted columns: {}'.format(epoch, i, np.count_nonzero(htm.temporal_memory.cell_active & ~np.all(htm.temporal_memory.cell_active, axis=1)[:, None])))

    print('{}s'.format(time.time() - prev_time))
