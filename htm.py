import numpy as np

class SpatialPooler:
    def __init__(self, input_size, columns, active_columns):
        self.input_size = input_size
        self.columns = columns
        self.active_columns = active_columns

        self.sparsity = self.active_columns / self.columns

        self.boosting_intensity = 0.5
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
        self.segment_activation = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.long)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.segment_active = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.bool)
        self.segment_matching = np.zeros_like(self.segment_active)
        self.segment_synapses = np.zeros((self.columns, self.cells, self.segment_capacity), dtype=np.long)

        self.cell_synapse_capacity = 0
        self.cell_synapse_cell = np.full((self.columns, self.cells, self.cell_synapse_capacity), -1, dtype=np.long)

        self.segment_synapse_capacity = 1
        self.segment_synapse_cell = np.full((self.columns, self.cells, self.segment_capacity, self.segment_synapse_capacity), -1, dtype=np.long)
        self.segment_synapse_permanence = np.zeros((self.columns, self.cells, self.segment_capacity, self.segment_synapse_capacity), dtype=np.float32)

        self.prev_winner_cell = np.zeros(0, dtype=np.long)
        self.prev_target_cell = np.zeros(0, dtype=np.long)

    def run(self, active_column):
        cell_predictive = self.cell_predictive[active_column]
        column_bursting = ~np.any(cell_predictive, axis=1)

        segment_potential = self.segment_potential[active_column].reshape(len(active_column), -1)
        column_best_matching_segment = np.argmax(segment_potential, axis=1)
        column_least_used_cell = np.argmin(self.cell_segments[active_column], axis=1)
        column_grow_segment = segment_potential[(np.arange(len(active_column), dtype=np.long), column_best_matching_segment)] == 0
        segment_learning = self.segment_active[active_column] | ((self.segment_index == column_best_matching_segment[:, None, None]) & (column_bursting & ~column_grow_segment)[:, None, None])

        learning_segment = np.nonzero(segment_learning)
        learning_segment = active_column[learning_segment[0]] * (self.cells * self.segment_capacity) + learning_segment[1] * self.segment_capacity + learning_segment[2]
        learning_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[learning_segment]
        learning_segment_synapse_cell_valid = np.nonzero(learning_segment_synapse_cell >= 0)
        learning_segment_synapse_cell = learning_segment_synapse_cell[learning_segment_synapse_cell_valid]
        self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[(learning_segment[learning_segment_synapse_cell_valid[0]], learning_segment_synapse_cell_valid[1])] += self.cell_active.reshape(-1)[learning_segment_synapse_cell] * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement
        
        punished_segment = np.nonzero(self.segment_active.reshape(-1, self.segment_capacity)[self.prev_target_cell])
        punished_segment = self.prev_target_cell[punished_segment[0]] * self.segment_capacity + punished_segment[1]
        punished_segment_synapse_cell = self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[punished_segment]
        punished_segment_synapse_cell_valid = np.nonzero(punished_segment_synapse_cell >= 0)
        punished_segment_synapse_cell = punished_segment_synapse_cell[punished_segment_synapse_cell_valid]
        self.segment_synapse_permanence.reshape(-1, self.segment_synapse_capacity)[(punished_segment[punished_segment_synapse_cell_valid[0]], punished_segment_synapse_cell_valid[1])] -= self.cell_active.reshape(-1)[punished_segment_synapse_cell] * self.permanence_punishment

        growing_segment_column = np.nonzero(column_grow_segment)[0]
        growing_segment_cell = column_least_used_cell[growing_segment_column]
        winner_cell = cell_predictive.copy()
        winner_cell[(growing_segment_column, growing_segment_cell)] = True
        winner_cell = np.nonzero(winner_cell)
        winner_cell = active_column[winner_cell[0]] * self.cells + winner_cell[1]

        if len(self.prev_winner_cell) > 0:
            growing_segment_column = active_column[growing_segment_column]
            growing_segment = self.cell_segments[(growing_segment_column, growing_segment_cell)]

            max_cell_segments = np.max(growing_segment) + 1 if len(growing_segment) > 0 else 0
            if max_cell_segments > self.segment_capacity:
                segment_capacity = max_cell_segments
                self.segment_index = np.arange(self.cells * segment_capacity, dtype=np.long).reshape(1, self.cells, segment_capacity)
                self.segment_activation = np.zeros((self.columns, self.cells, segment_capacity), dtype=np.long)
                self.segment_potential = np.zeros_like(self.segment_activation)

                segment_synapses = np.zeros((self.columns, self.cells, segment_capacity), dtype=np.long)
                segment_synapse_cell = np.full((self.columns, self.cells, segment_capacity, self.segment_synapse_capacity), -1, dtype=np.long)
                segment_synapse_permanence = np.zeros((self.columns, self.cells, segment_capacity, self.segment_synapse_capacity), dtype=np.float32)
                segment_synapses[:, :, :self.segment_capacity] = self.segment_synapses
                segment_synapse_cell[:, :, :self.segment_capacity, :] = self.segment_synapse_cell
                segment_synapse_permanence[:, :, :self.segment_capacity, :] = self.segment_synapse_permanence

                self.segment_capacity = segment_capacity
                self.segment_synapses = segment_synapses
                self.segment_synapse_cell = segment_synapse_cell
                self.segment_synapse_permanence = segment_synapse_permanence

            learning_segment = np.concatenate([learning_segment, growing_segment_column * (self.cells * self.segment_capacity) + growing_segment_cell * self.segment_capacity + growing_segment])
            segment_candidate = np.sort(np.concatenate([np.tile(self.prev_winner_cell, (len(learning_segment), 1)), np.tile(self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[learning_segment], 2)], axis=1), axis=1)
            segment_winner_targeted = segment_candidate[:, :-1] == segment_candidate[:, 1:]
            segment_candidate[:, :-1][segment_winner_targeted] = -1
            segment_candidate[:, 1:][segment_winner_targeted] = -1
            segment_index = np.arange(segment_candidate.shape[0])[:, None]
            candidate_index = np.arange(segment_candidate.shape[1])
            shuffled_candidate_index = np.tile(candidate_index, (segment_candidate.shape[0], 1))
            np.apply_along_axis(np.random.shuffle, 1, shuffled_candidate_index)
            segment_candidate[:, candidate_index] = segment_candidate[(segment_index, shuffled_candidate_index)]
            
            segment_new_synapses = np.maximum(np.minimum(self.synapse_sample_size - self.segment_potential.reshape(-1)[learning_segment], np.sum(segment_candidate >= 0, axis=1)), 0)
            new_synapse_segment = np.nonzero(segment_new_synapses)[0]
            if len(new_synapse_segment) > 0:
                learning_segment = learning_segment[new_synapse_segment]
                segment_candidate = segment_candidate[new_synapse_segment]
                segment_new_synapses = segment_new_synapses[new_synapse_segment]
                shuffled_candidate_index = shuffled_candidate_index[new_synapse_segment]
                segment_synapses = self.segment_synapses.reshape(-1)[learning_segment]

                max_segment_synapses = np.max(segment_synapses + segment_new_synapses) if len(learning_segment) > 0 else 0
                if max_segment_synapses > self.segment_synapse_capacity:
                    segment_synapse_capacity = max_segment_synapses
                    segment_synapse_cell = np.full((self.columns, self.cells, self.segment_capacity, segment_synapse_capacity), -1, dtype=np.long)
                    segment_synapse_permanence = np.zeros((self.columns, self.cells, self.segment_capacity, segment_synapse_capacity), dtype=np.float32)
                    segment_synapse_cell[:, :, :, :self.segment_synapse_capacity] = self.segment_synapse_cell
                    segment_synapse_permanence[:, :, :, :self.segment_synapse_capacity] = self.segment_synapse_permanence
                    self.segment_synapse_capacity = segment_synapse_capacity
                    self.segment_synapse_cell = segment_synapse_cell
                    self.segment_synapse_permanence = segment_synapse_permanence

                segment_target = np.nonzero(segment_candidate >= 0)
                segment_target_offset = np.ones(len(segment_target[0]), dtype=np.bool)
                segment_target_offset[1:] = segment_target[0][1:] != segment_target[0][:-1]
                segment_target_offset = np.nonzero(segment_target_offset)[0]
                segment_target_end = np.where(segment_new_synapses > 0, segment_target[1][segment_target_offset + segment_new_synapses - 1], 0)
                segment_target_offset[0] = len(segment_target[0])
                segment_new_synapse = np.arange(len(segment_target[0]), dtype=np.long) % segment_target_offset[segment_target[0]]
                segment_target_valid = np.nonzero(segment_target[1] <= segment_target_end[segment_target[0]])
                segment_target = (segment_target[0][segment_target_valid], segment_target[1][segment_target_valid])
                segment_new_synapse = segment_synapses[segment_target[0]] + segment_new_synapse[segment_target_valid]

                segment_target_segment = learning_segment[segment_target[0]]
                segment_target_candidate = segment_candidate[segment_target]
                self.segment_synapse_cell.reshape(-1, self.segment_synapse_capacity)[(segment_target_segment, segment_new_synapse)] = segment_target_candidate

                self.cell_segments[(growing_segment_column, growing_segment_cell)] += 1
                self.segment_synapses.reshape(-1)[learning_segment] += segment_new_synapses
                
                # TODO: they're not candidates at this point
                candidate_target = (shuffled_candidate_index[segment_target], segment_target[0])
                candidate_synapse_cell = np.full((segment_candidate.shape[1], segment_candidate.shape[0]), -1, dtype=np.long)
                candidate_synapse_cell[candidate_target] = segment_target_candidate
                candidate_valid = np.nonzero(np.any(candidate_synapse_cell >= 0, axis=1))[0]

                candidate_synapse_cell_candidate = candidate_synapse_cell[candidate_valid]
                candidate_synapse_cell_candidate_valid = np.nonzero(candidate_synapse_cell_candidate >= 0)
                candidate_synapse_cell_candidate[(candidate_synapse_cell_candidate_valid[0], 0)] = candidate_synapse_cell_candidate[candidate_synapse_cell_candidate_valid]
                candidate_synapse_cell_candidate = candidate_synapse_cell_candidate[:, 0]

                candidate_synapse_cell[candidate_target] = segment_target_segment // self.segment_capacity
                candidate_synapse_cell = candidate_synapse_cell[candidate_valid]
                candidate_synapse_cell = np.concatenate([candidate_synapse_cell, self.cell_synapse_cell.reshape(self.columns * self.cells, -1)[candidate_synapse_cell_candidate]], axis=1)
                candidate_synapse_cell = np.sort(candidate_synapse_cell, axis=1)
                candidate_synapse_cell[:, 1:][candidate_synapse_cell[:, 1:] == candidate_synapse_cell[:, :-1]] = -1
                candidate_synapse_cell_valid = candidate_synapse_cell >= 0

                candidate_synapses = np.sum(candidate_synapse_cell_valid, axis=1)
                max_cell_synapses = np.max(candidate_synapses)
                if max_cell_synapses > self.cell_synapse_capacity:
                    cell_synapse_capacity = max_cell_synapses
                    cell_synapse_cell = np.full((self.columns, self.cells, cell_synapse_capacity), -1, dtype=np.long)
                    cell_synapse_cell[:, :, :self.cell_synapse_capacity] = self.cell_synapse_cell
                    self.cell_synapse_capacity = cell_synapse_capacity
                    self.cell_synapse_cell = cell_synapse_cell

                candidate_synapse_cell_valid = np.nonzero(candidate_synapse_cell_valid)
                candidate_synapse_cell_offset = np.ones(len(candidate_synapse_cell_valid[0]), dtype=np.bool)
                candidate_synapse_cell_offset[1:] = candidate_synapse_cell_valid[0][1:] != candidate_synapse_cell_valid[0][:-1]
                candidate_synapse_cell_offset = np.nonzero(candidate_synapse_cell_offset)[0]
                candidate_synapse_cell_offset[0] = len(candidate_synapse_cell_valid[0])
                candidate_synapse_cell_index = np.arange(len(candidate_synapse_cell_valid[0]), dtype=np.long) % candidate_synapse_cell_offset[candidate_synapse_cell_valid[0]]
                candidate_synapse_cell_candidate = candidate_synapse_cell_candidate[candidate_synapse_cell_valid[0]]
                self.cell_synapse_cell.reshape(-1, self.cell_synapse_capacity)[(candidate_synapse_cell_candidate, candidate_synapse_cell_index)] = candidate_synapse_cell[candidate_synapse_cell_valid]

        cell_active = cell_predictive | column_bursting[:, None]
        self.cell_active[:, :] = False
        self.cell_active[active_column] = cell_active

        active_cell = np.nonzero(cell_active)
        active_cell = (active_column[active_cell[0]], active_cell[1])

        cell_targeted = np.zeros(self.columns * self.cells, dtype=np.bool)
        active_cell_synapse_cell = self.cell_synapse_cell[active_cell]
        active_cell_synapse_cell = active_cell_synapse_cell[active_cell_synapse_cell >= 0]
        cell_targeted[active_cell_synapse_cell] = True
        target_cell = np.nonzero(cell_targeted)[0]
        
        segment_synapse_cell_active = self.cell_active.reshape(-1)[self.segment_synapse_cell.reshape(-1, self.segment_capacity, self.segment_synapse_capacity)[target_cell]]
        segment_synapse_permanence = self.segment_synapse_permanence.reshape(-1, self.segment_capacity, self.segment_synapse_capacity)[target_cell]
        segment_synapse_weight = segment_synapse_permanence > self.permanence_threshold

        self.segment_activation[:, :, :] = 0
        self.segment_potential[:, :, :] = 0
        self.segment_activation.reshape(-1, self.segment_capacity)[target_cell] = np.sum(segment_synapse_cell_active & segment_synapse_weight, axis=2)
        self.segment_potential.reshape(-1, self.segment_capacity)[target_cell] = np.sum(segment_synapse_cell_active, axis=2)
        self.segment_active = self.segment_activation >= self.segment_active_threshold
        self.segment_matching = self.segment_potential >= self.segment_matching_threshold
        self.cell_predictive = np.any(self.segment_active, axis=2)

        self.prev_winner_cell = winner_cell
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
