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

    def evaluate_weight(self):
        return self.permanence >= self.perm_threshold

    def process(self, input):
        weight = self.evaluate_weight()
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


class DynamicArray:
    def __init__(self, sizes=[(1, )], dtypes=[np.float32], capacity=0):
        self.length = 0
        self.capacity = capacity
        self.sizes = sizes
        self.dtypes = dtypes

        self.arrays = self.initialize_arrays(self.capacity)

    def initialize_arrays(self, capacity):
        return [np.empty((capacity, ) + size, dtype=dtype) for size, dtype in zip(self.sizes, self.dtypes)]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return [array[:self.length][index] for array in self.arrays]
        
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

    def evaluate_active_column(self, overlaps):
        return np.argpartition(overlaps, -self.active_columns)[-self.active_columns:]

    def process(self, input, learning=True):
        overlaps = self.proximal_projection.process(input)
        boosted_overlaps = self.boosting.process(overlaps)
        active_column = self.evaluate_active_column(boosted_overlaps)

        if learning:
            self.proximal_projection.update(input, active_column)
        self.boosting.update(active_column)

        return SpatialPooler.State(overlaps, boosted_overlaps, active_column)


class TemporalMemory:
    class State:
        def __init__(self, bursting_column=None, predictive_cell=None, active_cell=None, winner_cell=None):
            self.bursting_column = bursting_column
            self.predictive_cell = predictive_cell
            self.active_cell = active_cell
            self.winner_cell = winner_cell

    # TODO: active_columns is temp
    def __init__(self, column_dim, cell_dim, active_columns):
        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.segm_matching_threshold = 10
        self.segm_active_threshold = 10

        self.perm_threshold = 0.5

        # TODO: every bit of code referring to these is wrong.
        # distal_postsynaptic is the distal segments and the permanence matrix should also grow in the other(synapses) direction.
        self.distal_presynaptic = DynamicArray(sizes=[(self.column_dim, self.cell_dim)], dtypes=[np.int32])
        self.distal_postsynaptic = DynamicArray(sizes=[(self.column_dim * self.cell_dim, )] * 2, dtypes=[np.int32, np.float32])

        self.last_state = TemporalMemory.State(
            bursting_column=np.zeros((active_columns, ), dtype=np.bool_),
            predictive_cell=np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_),
            active_cell=np.zeros((active_columns, self.cell_dim), dtype=np.bool_),
            winner_cell=np.zeros((active_columns, self.cell_dim), dtype=np.bool_),
        )

    def process(self, sp_state, prev_state=None, learning=True):
        if prev_state is None:
            prev_state = self.last_state

        col_prev_predictive_cell = prev_state.predictive_cell[sp_state.active_column]
        col_bursting_column = ~np.any(col_prev_predictive_cell, axis=1)
        
        col_active_cell = col_prev_predictive_cell | np.expand_dims(col_bursting_column, 1)
        active_cell = np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_)
        active_cell[sp_state.active_column] = col_active_cell

        postsynaptic_cell = np.unique(self.distal_presynaptic[:, sp_state.active_column][0][:, col_active_cell])
        presynaptic_cell, presynaptic_permanence = self.distal_postsynaptic[:, postsynaptic_cell]
        presynaptic_weight = presynaptic_permanence >= self.perm_threshold

        active_presynaptic_cell = active_cell.reshape(-1)[presynaptic_cell]
        segment_activation = np.sum(active_presynaptic_cell & presynaptic_weight, axis=1)
        segment_potential = np.sum(active_presynaptic_cell, axis=1)
        active_segment = segment_activation >= self.segm_active_threshold
        matching_segment = segment_potential >= self.segm_matching_threshold

        predictive_cell = np.zeros((self.column_dim * self.cell_dim), dtype=np.bool_)
        predictive_cell[postsynaptic_cell] = np.any(active_segment, axis=0)

        winner_cell = active_cell.copy()

        curr_state = TemporalMemory.State(col_bursting_column, predictive_cell.reshape(self.column_dim, self.cell_dim), active_cell, winner_cell)
        self.last_state = curr_state
        return curr_state


class HierarchicalTemporalMemory:
    def __init__(self, input_dim, column_dim, cell_dim, active_columns=None):
        if active_columns is None:
            active_columns = round(column_dim * 0.02)

        self.spatial_pooler = SpatialPooler(input_dim, column_dim, active_columns)
        self.temporal_memory = TemporalMemory(column_dim, cell_dim, active_columns)

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