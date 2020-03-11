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
        self.cell_synapses = np.zeros((self.columns, self.cells), dtype=np.long)
        self.cell_best_matching_segment = np.full((self.columns, self.cells), -1, dtype=np.long)
        
        self.prev_active_cell = np.zeros(0, dtype=np.long)
        self.prev_winner_cell = np.zeros(0, dtype=np.long)
        self.winner_cell = np.zeros(0, dtype=np.long)

        self.segments = 0
        self.segments_capacity = 0
        self.segment_index = np.arange(self.segments_capacity)
        self.segment_cell = np.full((self.segments_capacity, ), -1, dtype=np.long)
        self.segment_activation = np.zeros(self.segments_capacity, dtype=np.long)
        self.segment_potential = np.zeros_like(self.segment_activation)
        self.active_segment = np.zeros(0, dtype=np.long)

        self.synapses = 0
        self.synapses_capacity = 0
        self.synapse_segment = np.full((self.columns, self.cells, self.synapses_capacity), -1, dtype=np.long)
        self.synapse_permanence = np.full(self.synapse_segment.shape, self.permanence_initial, dtype=np.float32)

    def run(self, active_column):
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
        self.cell_segments[active_column] += cell_new_segment

        new_segment_cell = self.get_active_column_cell_index(active_column, np.nonzero(cell_new_segment))
        active_cell_offset = active_column[:, None] * self.cells
        prev_active_segment_cell = self.segment_cell[self.active_segment][None, :]
        segment_correct = np.any((active_cell_offset <= prev_active_segment_cell) & (prev_active_segment_cell <= active_cell_offset + self.cells), axis=0)
        correct_segment = self.active_segment[segment_correct]
        incorrect_segment = self.active_segment[~segment_correct]
        new_segment = self.segments + np.arange(len(new_segment_cell))
        learning_segment = np.unique(np.concatenate([cell_best_matching_segment[cell_segment_matching].reshape(-1), correct_segment, new_segment]))

        prev_active_synapse_segment = self.synapse_segment.reshape(self.columns * self.cells, -1)[self.prev_active_cell]
        prev_active_synapse_learning = np.any(prev_active_synapse_segment[:, :, None] == learning_segment[None, None, :], axis=2)
        prev_active_synapse_punished = np.any(prev_active_synapse_segment[:, :, None] == incorrect_segment[None, None, :], axis=2)
        learning = np.where(prev_active_synapse_learning, np.array([self.permanence_increment]), np.array([self.permanence_decrement]))
        punishment = np.where(prev_active_synapse_punished, np.array([self.permanence_punishment]), np.array([0.0]))
        self.synapse_permanence.reshape(self.columns * self.cells, -1)[self.prev_active_cell] += learning + punishment

        self.winner_cell = self.get_active_column_cell_index(active_column, np.nonzero(cell_winner))
        
        if len(self.prev_winner_cell) > 0:
            segments = self.segments + len(new_segment)
            if segments > self.segments_capacity:
                self.segments_capacity = TemporalMemory.get_exponential_capacity(segments)
                
                self.segment_index = np.arange(self.segments_capacity)

                segment_cell = np.full((self.segments_capacity, ), -1, dtype=np.long)
                segment_cell[:self.segments] = self.segment_cell[:self.segments]
                self.segment_cell = segment_cell

                self.segment_activation = np.concatenate([self.segment_activation, np.zeros(self.segments_capacity - len(self.segment_activation), dtype=np.long)])
                self.segment_potential = np.concatenate([self.segment_potential, np.zeros(self.segments_capacity - len(self.segment_potential), dtype=np.long)])

            self.segment_cell[self.segments:segments] = new_segment_cell
            self.segments = segments
            
            candidate_cell_synapse_segment = self.synapse_segment.reshape(self.columns * self.cells, -1)[self.prev_winner_cell]
            segment_cell_candidate = np.all(learning_segment[:, None, None] != candidate_cell_synapse_segment[None, :, :], axis=2)
            segment_new_synapses = np.maximum(self.synapse_sample_size - np.sum(~segment_cell_candidate, axis=1), 0)
            
            max_new_synapses = np.max(segment_new_synapses)
            if max_new_synapses > 0:
                random_connect = np.tile(np.arange(segment_cell_candidate.shape[1]), (segment_cell_candidate.shape[0], 1)) < segment_new_synapses[:, None]
                np.apply_along_axis(np.random.shuffle, 1, random_connect)
                segment_cell_candidate &= random_connect
                
                candidate_cell_new_synapse = np.nonzero(segment_cell_candidate.T)
                if len(candidate_cell_new_synapse[0]) > 0:
                    cell_synapses = self.cell_synapses.reshape(-1)[self.prev_winner_cell] + np.sum(segment_cell_candidate, axis=0)
                    synapses = np.max(cell_synapses)
                    if synapses > self.synapses_capacity:
                        self.synapses_capacity = TemporalMemory.get_exponential_capacity(synapses)
                        
                        synapse_segment = np.full((self.columns, self.cells, self.synapses_capacity), -1, dtype=np.long)
                        synapse_segment[:, :, :self.synapses] = self.synapse_segment[:, :, :self.synapses]
                        self.synapse_segment = synapse_segment

                        synapse_permanence = np.full(self.synapse_segment.shape, self.permanence_initial, dtype=np.float32)
                        synapse_permanence[:, :, :self.synapses] = self.synapse_permanence[:, :, :self.synapses]
                        self.synapse_permanence = synapse_permanence

                    synapse_segment_cell = self.prev_winner_cell[candidate_cell_new_synapse[0]]
                    candidate_cell, synapse_segment_synapses = np.unique(synapse_segment_cell, return_counts=True)
                    synapse_segment_synapse_offset = np.argmin(np.where(candidate_cell[:, None] == synapse_segment_cell[None, :], np.arange(len(synapse_segment_cell)), np.array([len(synapse_segment_cell)], dtype=np.long)), axis=1)
                    max_segment_synapses = np.max(synapse_segment_synapses)
                    masked_synapse_index = np.arange(max_segment_synapses)
                    valid_synapse_index = np.nonzero(masked_synapse_index[None, :] < synapse_segment_synapses[:, None])
                    synapse_segment_synapse_index = np.empty_like(candidate_cell_new_synapse[0])
                    synapse_segment_synapse_index[(synapse_segment_synapse_offset[:, None] + masked_synapse_index[None, :])[valid_synapse_index]] = masked_synapse_index[valid_synapse_index[1]]

                    synapse_segment_synapse = self.cell_synapses.reshape(-1)[synapse_segment_cell] + synapse_segment_synapse_index
                    self.synapse_segment.reshape(self.columns * self.cells, -1)[(synapse_segment_cell, synapse_segment_synapse)] = learning_segment[candidate_cell_new_synapse[1]]
                    self.cell_synapses.reshape(-1)[self.prev_winner_cell] = cell_synapses
                    self.synapses = synapses

        active_cell = self.get_active_column_cell_index(active_column, np.nonzero(cell_active))
        active_synapse_segment = self.synapse_segment.reshape(self.columns * self.cells, -1)[active_cell].reshape(-1)
        active_synapse_weight = self.synapse_permanence.reshape(self.columns * self.cells, -1)[active_cell].reshape(-1) > self.permanence_threshold
        affected_segment = np.unique(active_synapse_segment[active_synapse_segment >= 0])
        segment_synapse_active = affected_segment[:, None] == active_synapse_segment[None, :]
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
            cell_segment_matching = affected_cell[:, None] == matching_segment_cell[None, :]
            cell_segment_activation = self.segment_activation[matching_segment][None, :] * cell_segment_matching
            cell_segment_potential = self.segment_potential[matching_segment][None, :] * cell_segment_matching
            cell_best_matching_segment = self.segment_index[matching_segment][np.argmax(cell_segment_potential, axis=1)]
            self.cell_predictive.reshape(-1)[affected_cell] = np.any(cell_segment_activation >= self.segment_activation_threshold, axis=1)
            self.cell_best_matching_segment.reshape(-1)[affected_cell] = cell_best_matching_segment

        self.prev_active_cell = active_cell

    def get_active_column_cell_index(self, active_column, active_column_cell):
        return active_column[active_column_cell[0]] * self.cells + active_column_cell[1]

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
