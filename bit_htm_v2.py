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


class GlobalInhibition:
    def __init__(self, actives):
        self.actives = actives

    def process(self, input):
        return np.argpartition(input, -self.actives)[-self.actives:]


class DynamicArray:
    def __init__(self, sizes=[tuple()], dtypes=[np.float32], default_values=[0.0], capacity=0):
        self.length = 0
        self.capacity = capacity
        
        self.sizes = sizes
        self.dtypes = dtypes
        self.default_values = default_values

        self.arrays = self.initialize_arrays(self.capacity)

    def initialize_arrays(self, capacity):
        return [np.full((capacity, ) + size, default_value, dtype=dtype) for size, dtype, default_value in zip(self.sizes, self.dtypes, self.default_values)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return [array[:self.length][index] for array in self.arrays]

    def __setitem__(self, index, values):
        for array, value in zip(self.arrays, values):
            array[:self.length][index] = value

    def add(self, added_arrays):
        new_length = self.length + len(added_arrays[0])
        if new_length > self.capacity:
            new_capacity = 2 ** int(np.ceil(np.log2(new_length)))
            new_arrays = self.initialize_arrays(new_capacity)
            for old_array, new_array in zip(self.arrays, new_arrays):
                new_array[:self.length] = old_array[:self.length]
            self.arrays = new_arrays
            self.capacity = new_capacity
        for array, added_array in zip(self.arrays, added_arrays):
            array[self.length:new_length] = added_array
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
        def __init__(self, bursting_column, predictive_cell, active_cell, winner_cell, active_segment, matching_segment):
            self.bursting_column = bursting_column
            self.predictive_cell = predictive_cell
            self.active_cell = active_cell
            self.winner_cell = winner_cell
            self.active_segment = active_segment
            self.matching_segment = matching_segment

    # TODO: active_columns is temp
    def __init__(self, column_dim, cell_dim):
        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.segm_matching_threshold = 10
        self.segm_active_threshold = 10

        self.perm_initial = 0.01
        self.perm_threshold = 0.5
        self.perm_increment = 0.3
        self.perm_decrement = 0.05
        self.perm_punishment = 0.01

        self.cell_synapse = DynamicArray(sizes=[(self.column_dim, self.cell_dim)] * 2, dtypes=[np.int32, np.float32], default_values=[-1, -1.0])
        self.segment_cell = DynamicArray(dtypes=[np.int32])

        self.last_state = TemporalMemory.State(
            np.empty(0, dtype=np.bool_),
            np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_),
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)),
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)),
            np.empty(0, dtype=np.bool_), np.empty(0, dtype=np.bool_)
        )

    def process(self, sp_state, prev_state=None, learning=True):
        if prev_state is None:
            prev_state = self.last_state

        prev_predictive_cell = prev_state.predictive_cell[sp_state.active_column]
        bursting_column = ~np.any(prev_predictive_cell, axis=1, keepdims=True)
        
        random_cell = np.zeros((len(sp_state.active_column), self.cell_dim), dtype=np.bool_)
        random_cell[np.arange(len(sp_state.active_column)), np.random.randint(0, self.cell_dim, len(sp_state.active_column))] = True

        active_cell = np.where(prev_predictive_cell | bursting_column)
        winner_cell = np.where(prev_predictive_cell | (bursting_column & random_cell))

        ######

        # TODO: bad name. come up with an adjective
        # TODO: maybe noun_adjective: boolean mask, adjective_noun: indices like before?
        cell_active = np.zeros(self.column_dim * self.cell_dim, dtype=np.bool_)
        cell_active.reshape(self.column_dim, self.cell_dim)[active_cell] = True

        prev_active_cell = prev_state.active_cell
        target_segment, permanence = self.cell_synapse[:]
        target_segment = target_segment[:, prev_active_cell[0], prev_active_cell[1]]
        target_active_cell = cell_active[self.segment_cell[target_segment][0]]

        permanence[:, prev_active_cell[0], prev_active_cell[1]] += prev_state.matching_segment[target_segment] * (target_active_cell * (self.perm_increment + self.perm_decrement) - self.perm_decrement)
        
        # TODO: validation needed
        permanence[:, prev_active_cell[0], prev_active_cell[1]] -= (prev_state.active_segment[target_segment] & target_active_cell) * self.perm_punishment

        ######

        # TODO: using the word potential is probably bad
        target_segment, permanence = map(np.ndarray.flatten, self.cell_synapse[:, active_cell[0], active_cell[1]])
        segment_activation = np.zeros(len(self.segment_cell), dtype=np.int32)
        segment_potential = np.zeros_like(segment_activation)
        np.add.at(segment_activation, target_segment, permanence >= self.perm_threshold)
        np.add.at(segment_potential, target_segment, 1)
        active_segment = segment_activation >= self.segm_active_threshold
        matching_segment = segment_potential >= self.segm_active_threshold

        predictive_cell = np.zeros(self.column_dim * self.cell_dim, dtype=np.int32)
        np.add.at(predictive_cell, self.segment_cell[active_segment][0], 1)
        predictive_cell = (predictive_cell > 0).reshape(self.column_dim, self.cell_dim)

        curr_state = TemporalMemory.State(bursting_column.squeeze(1), predictive_cell, active_cell, winner_cell, active_segment, matching_segment)
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
    inputs = np.random.randn(100, 1000) > 1.0
    htm = HierarchicalTemporalMemory(inputs.shape[1], 2048, 32)

    import time

    start_time = time.time()

    for epoch in range(10):
        for input in inputs:
            sp_state, tm_state = htm.process(input)
            # print(htm.spatial_pooler.boosting.duty_cycle.mean(), htm.spatial_pooler.boosting.duty_cycle.std())

    print(f'{time.time() - start_time}s')