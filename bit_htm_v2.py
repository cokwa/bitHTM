import numpy as np


class DenseProjection:
    def __init__(
        self, input_dim, output_dim,
        perm_mean=0.0, perm_std=0.1,
        perm_threshold=0.0, perm_increment=0.1, perm_decrement=0.3
    ):
        self.perm_threshold = perm_threshold
        self.perm_increment = perm_increment
        self.perm_decrement = perm_decrement

        self.permanence = np.random.randn(output_dim, input_dim) * perm_std + perm_mean

    def process(self, input):
        weight = self.permanence >= self.perm_threshold
        overlaps = np.sum(weight & input, axis=1)
        return overlaps

    def update(self, input, target):
        self.permanence[target] += input * (self.perm_increment + self.perm_decrement) - self.perm_decrement


class ExponentialBoosting:
    def __init__(
        self, output_dim, actives,
        intensity=0.3, momentum=0.99
    ):
        self.sparsity = actives / output_dim
        self.intensity = intensity
        self.momentum = momentum

        self.duty_cycle = np.zeros(output_dim, dtype=np.float32)

    def process(self, input):
        factor = np.exp(self.intensity * -self.duty_cycle / self.sparsity)
        return factor * input

    def update(self, active):
        self.duty_cycle *= self.momentum
        self.duty_cycle[active] += 1.0 - self.momentum


def bincount(x, weights=None, minLength=0):
    out = np.bincount(x, weights=weights)
    out = np.concatenate([out, np.zeros(minLength - len(out), dtype=out.dtype)])
    return out


class GlobalInhibition:
    def __init__(self, actives):
        self.actives = actives

    def process(self, input):
        return np.argpartition(input, -self.actives)[-self.actives:]


class DynamicArray:
    def __init__(self, size=tuple(), dtypes=[np.float32], capacity=0, capacity_exponential=True):
        self.length = 0
        self.capacity = capacity
        
        self.size = size
        self.dtypes = dtypes
        self.capacity_exponential = capacity_exponential
        
        self.arrays = self.initialize_arrays(self.capacity)

    def initialize_arrays(self, capacity):
        return tuple(np.empty(self.size + (capacity, ), dtype=dtype) for dtype in self.dtypes)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return tuple(array[..., :self.length][index] for array in self.arrays)

    def __setitem__(self, index, values):
        for array, value in zip(self.arrays, values):
            array[..., :self.length][index] = value

    def add(self, *added_arrays):
        new_length = self.length + added_arrays[0].shape[-1]
        if new_length > self.capacity:
            new_capacity = 2 ** int(np.ceil(np.log2(new_length))) if self.capacity_exponential else new_length
            new_arrays = self.initialize_arrays(new_capacity)
            for old_array, new_array in zip(self.arrays, new_arrays):
                new_array[..., :self.length] = old_array[..., :self.length]
            self.arrays = new_arrays
            self.capacity = new_capacity
        for array, added_array in zip(self.arrays, added_arrays):
            array[..., self.length:new_length] = added_array
        self.length = new_length


class SpatialPooler:
    class State:
        def __init__(self, overlaps, boosted_overlaps, active_column):
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps
            self.active_column = active_column

    def __init__(self, input_dim, column_dim, active_columns):
        self.input_dim = input_dim
        self.column_dim = column_dim
        self.active_columns = active_columns

        self.proximal_projection = DenseProjection(input_dim, column_dim)
        self.boosting = ExponentialBoosting(column_dim, active_columns)
        self.inhibition = GlobalInhibition(active_columns)

    def process(self, input, learning=True):
        overlaps = self.proximal_projection.process(input)
        boosted_overlaps = self.boosting.process(overlaps)
        active_column = self.inhibition.process(boosted_overlaps)

        if learning:
            self.proximal_projection.update(input, active_column)
        self.boosting.update(active_column)

        return SpatialPooler.State(overlaps, boosted_overlaps, active_column)


class TemporalMemory:
    class State:
        def __init__(self, column_bursting, cell_prediction, active_cell, winner_cell, segment_activation, segment_potential):
            self.column_bursting = column_bursting
            self.cell_prediction = cell_prediction
            self.active_cell = active_cell
            self.winner_cell = winner_cell
            self.segment_activation = segment_activation
            self.segment_potential = segment_potential

    # TODO: active_columns is temp
    def __init__(self, column_dim, cell_dim):
        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.segm_matching_threshold = 10
        self.segm_activation_threshold = 10
        self.syn_sample_size = 20

        self.perm_initial = 0.01
        self.perm_threshold = 0.5
        self.perm_increment = 0.3
        self.perm_decrement = 0.05
        self.perm_punishment = 0.01

        self.eps = 1e-8

        self.cell_segments = np.zeros((self.column_dim, self.cell_dim), dtype=np.int32)
        # self.cell_synapse = DynamicArray(size=(self.column_dim, self.cell_dim), dtypes=[np.int32, np.float32], capacity_exponential=False)
        self.cell_synapse = DynamicArray(size=(self.column_dim, self.cell_dim), dtypes=[np.int32, np.float32])
        self.segment_cell = DynamicArray(dtypes=[np.int32])

        self.last_state = TemporalMemory.State(
            np.empty(0, dtype=np.bool_),
            np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_),
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)),
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)),
            np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        )

    # TODO: implement return_winner_cell for real
    def process(self, sp_state, prev_state=None, learning=True, return_winner_cell=True):
        if prev_state is None:
            prev_state = self.last_state

        prev_cell_prediction = prev_state.cell_prediction[sp_state.active_column]
        column_bursting = ~np.any(prev_cell_prediction, axis=1)
        
        active_cell = np.where(prev_cell_prediction | np.expand_dims(column_bursting, 1))
        active_cell = (sp_state.active_column[active_cell[0]], active_cell[1])

        if learning:
            column_activation = np.zeros(self.column_dim, dtype=np.bool_)
            column_activation[sp_state.active_column] = True
            column_punishment = np.any(prev_state.cell_prediction, axis=1)
            column_punishment &= ~column_activation

            prev_active_cell = prev_state.active_cell
            target_segment, permanence = self.cell_synapse[:]
            target_segment = target_segment[prev_active_cell]
            target_column = self.segment_cell[target_segment][0] // self.cell_dim
            target_column_activation = column_activation[target_column]
            target_column_punishment = column_punishment[target_column]

            target_segment_activation = prev_state.segment_activation[target_segment]
            target_segment_potential = prev_state.segment_potential[target_segment]
            target_segment_potential_jittered = (target_segment_potential + np.random.rand(*target_segment_potential.shape)).astype(np.float32)

            target_segment_active = target_segment_activation >= self.segm_activation_threshold
            target_segment_matching = target_segment_potential >= self.segm_matching_threshold

            column_max_potential = np.zeros(self.column_dim, dtype=np.float32)
            np.maximum.at(column_max_potential, target_column.flatten(), target_segment_potential_jittered.flatten())
            column_max_potential *= column_activation
            # column_max_potential = column_activation * bincount(target_column.flatten(), weights=target_segment_potential_jittered.flatten(), minLength=self.column_dim).astype(np.int32)
            
            target_segment_best_matching = target_segment_matching & (np.abs(target_segment_potential_jittered - column_max_potential[target_column]) < self.eps)
            target_segment_learning = target_segment_active | target_segment_best_matching

            permanence[prev_active_cell] += target_segment_learning * (target_column_activation * (self.perm_increment + self.perm_decrement) - self.perm_decrement)
            permanence[prev_active_cell] -= (target_segment_potential & target_column_punishment) * self.perm_punishment

            segment_growing_active_column = np.where(column_bursting & (column_max_potential[sp_state.active_column] < self.segm_matching_threshold))[0]
            segment_growing_column = sp_state.active_column[segment_growing_active_column]
            cell_segments_jittered = self.cell_segments[segment_growing_column]
            cell_segments_jittered = (cell_segments_jittered + np.random.rand(*cell_segments_jittered.shape)).astype(np.float32)
            least_used_cell = np.argmin(cell_segments_jittered, axis=1)

            winner_cell = prev_cell_prediction.copy()
            winner_cell[segment_growing_active_column, least_used_cell] = True
            winner_cell = np.where(winner_cell)
            winner_cell = (sp_state.active_column[winner_cell[0]], winner_cell[1])

            # segment_learning = np.zeros(len(self.segment_cell), dtype=np.bool_)
            # np.maximum.at(segment_learning, target_segment.flatten(), target_segment_learning.flatten())
            segment_learning = bincount(target_segment.flatten(), weights=target_segment_learning.flatten(), minLength=len(self.segment_cell)) > 0
            learning_segment = np.concatenate([np.where(segment_learning)[0], np.arange(len(least_used_cell)) + len(self.segment_cell)])
            segment_new_synapses = np.zeros(len(learning_segment), dtype=np.int32)
            segment_new_synapses[:-len(least_used_cell)] = self.syn_sample_size - prev_state.segment_potential[learning_segment[:-len(least_used_cell)]]
            segment_new_synapses[-len(least_used_cell):] = self.syn_sample_size

            self.segment_cell.add(segment_growing_column * self.cell_dim + least_used_cell)
            self.cell_segments[segment_growing_column, least_used_cell] += 1

            prev_winner_cell = prev_state.winner_cell
            target_segment, permanence = self.cell_synapse[prev_winner_cell]

            # cell_segment_candidate = np.empty((target_segment.shape[0], target_segment.shape[1] + len(learning_segment)), dtype=np.int32)
            # cell_segment_candidate[:, :-len(learning_segment)] = (permanence > 0.0) * (target_segment + 1) - 1
            # cell_segment_candidate[:, -len(learning_segment):] = np.expand_dims(learning_segment, axis=0)
            # cell_segment_candidate.sort(axis=1)

            # cell_segment_already_targeted = cell_segment_candidate[:, :-1] == cell_segment_candidate[:, 1:]
            # cell_segment_candidate[:, :-1][cell_segment_already_targeted] = -1
            # cell_segment_candidate[:, 1:][cell_segment_already_targeted] = -1

            # cell_segment_priority = np.random.rand(*cell_segment_candidate.shape)
            # cell_segment_priority[cell_segment_candidate < 0] = np.inf
            # # cell_segment_sample = np.argpartition(cell_segment_priority, self.syn_sample_size, axis=1)[:, :self.syn_sample_size]
            # # cell_segment_candidate = np.take_along_axis(cell_segment_candidate, cell_segment_sample, axis=1)

            # segment_new_synapses = np.zeros(len(self.segment_cell), dtype=np.int32)
            # np.add.at(segment_new_synapses, cell_segment_candidate.flatten(), cell_segment_candidate.flatten() >= 0)

            # # import matplotlib.pyplot as plt
            # # vis = np.zeros((int(np.ceil(np.sqrt(len(segment_new_synapses)))), ) * 2)
            # # vis.reshape(-1)[:len(segment_new_synapses)] = segment_new_synapses
            # # plt.imshow(vis)
            # # plt.show()

            ###########################################

            segment_full_to_learning = np.random.randint(0, len(learning_segment), len(self.segment_cell), dtype=np.int32)
            segment_full_to_learning_valid = np.zeros_like(segment_full_to_learning, dtype=np.bool_)
            segment_full_to_learning[learning_segment] = np.arange(len(learning_segment))
            segment_full_to_learning_valid[learning_segment] = True

            segment_cell_priority = np.random.rand(len(prev_winner_cell[0]), len(learning_segment)).astype(np.float32)
            np.add.at(
                segment_cell_priority,
                (np.repeat(np.arange(target_segment.shape[0]), target_segment.shape[1]), segment_full_to_learning[target_segment.flatten()]),
                (segment_full_to_learning_valid[target_segment.flatten()] & (permanence.flatten() > 0.0)) * (1 + self.eps)
            )
            priority_argsort = np.argsort(segment_cell_priority, axis=1)

            new_target_segment = np.random.randint(0, len(self.segment_cell), (self.column_dim, self.cell_dim, len(learning_segment)), dtype=np.int32)
            new_permanence = np.full(new_target_segment.shape, -1.0, dtype=np.float32)
            new_target_segment[prev_winner_cell] = learning_segment[priority_argsort]
            new_permanence[prev_winner_cell] = (np.take_along_axis(segment_cell_priority, priority_argsort, axis=1) < 1) * (self.perm_initial + 1.0) - 1.0
            self.cell_synapse.add(new_target_segment, new_permanence)

        target_segment, permanence = map(np.ndarray.flatten, self.cell_synapse[active_cell])
        # segment_activation = np.zeros(len(self.segment_cell), dtype=np.int32)
        # segment_potential = np.zeros_like(segment_activation)
        # np.add.at(segment_activation, target_segment, permanence >= self.perm_threshold)
        # np.add.at(segment_potential, target_segment, permanence > 0.0)
        segment_activation = bincount(target_segment, weights=permanence >= self.perm_threshold, minLength=len(self.segment_cell)).astype(np.int32)
        segment_potential = bincount(target_segment, weights=permanence > 0.0, minLength=len(self.segment_cell)).astype(np.int32)

        # cell_prediction = np.zeros(self.column_dim * self.cell_dim, dtype=np.int32)
        # np.add.at(cell_prediction, self.segment_cell[segment_activation >= self.segm_activation_threshold][0], 1)
        cell_prediction = bincount(self.segment_cell[segment_activation >= self.segm_activation_threshold][0], minLength=self.column_dim * self.cell_dim)
        cell_prediction = (cell_prediction > 0).reshape(self.column_dim, self.cell_dim)

        curr_state = TemporalMemory.State(column_bursting, cell_prediction, active_cell, winner_cell, segment_activation, segment_potential)
        self.last_state = curr_state
        return curr_state


class HierarchicalTemporalMemory:
    def __init__(self, input_dim, column_dim, cell_dim, active_columns=None):
        if active_columns is None:
            active_columns = round(column_dim * 0.02)

        self.spatial_pooler = SpatialPooler(input_dim, column_dim, active_columns)
        self.temporal_memory = TemporalMemory(column_dim, cell_dim)

    def process(self, input, learning=True):
        sp_state = self.spatial_pooler.process(input, learning=learning)
        tm_state = self.temporal_memory.process(sp_state, learning=learning)
        return sp_state, tm_state


if __name__ == '__main__':
    inputs = np.random.randn(10, 1000) > 1.0
    htm = HierarchicalTemporalMemory(inputs.shape[1], 2048, 32)

    import time

    start_time = time.time()

    for epoch in range(10):
        for input in inputs:
            sp_state, tm_state = htm.process(input)
            print(tm_state.column_bursting.sum(), tm_state.cell_prediction.sum(), tm_state.segment_potential.sum())
            # print(htm.spatial_pooler.boosting.duty_cycle.mean(), htm.spatial_pooler.boosting.duty_cycle.std())

    print(f'{time.time() - start_time}s')