import numpy as np


class DenseProjection:
    def __init__(
        self, input_dim, output_dim,
        perm_mean=0.0, perm_std=1.0,
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
        self.permanence[target] = input * (self.perm_increment + self.perm_decrement) - self.perm_decrement


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
        def __init__(self, overlaps, boosted_overlaps, column_active):
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps
            self.column_active = column_active

    def __init__(self, input_dim, column_dim, active_columns):
        self.input_dim = input_dim
        self.column_dim = column_dim
        self.active_columns = active_columns

        self.proximal_projection = DenseProjection(input_dim, column_dim)
        self.boosting = ExponentialBoosting(column_dim, active_columns)

    def evaluate_column_active(self, overlaps):
        return np.argpartition(overlaps, -self.active_columns)[-self.active_columns:]

    def process(self, input, learning=True):
        overlaps = self.proximal_projection.process(input)
        boosted_overlaps = self.boosting.process(overlaps)
        column_active = self.evaluate_column_active(boosted_overlaps)

        if learning:
            self.proximal_projection.update(input, column_active)
        self.boosting.update(column_active)

        return SpatialPooler.State(overlaps, boosted_overlaps, column_active)


class TemporalMemory:
    class State:
        def __init__(self, column_bursting=None, cell_predictive=None, cell_active=None, cell_winner=None):
            self.column_bursting = column_bursting
            self.cell_predictive = cell_predictive
            self.cell_active = cell_active
            self.cell_winner = cell_winner

    # TODO: active_columns is temp
    def __init__(self, column_dim, cell_dim, active_columns):
        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.distal_synapses = DynamicArray(sizes=[tuple(), tuple()], dtypes=[np.int32, np.float32])

        self.last_state = TemporalMemory.State(
            column_bursting=np.zeros((active_columns, ), dtype=np.bool_),
            cell_predictive=np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_),
            cell_active=np.zeros((active_columns, self.cell_dim), dtype=np.bool_),
            cell_winner=np.zeros((active_columns, self.cell_dim), dtype=np.bool_),
        )

    def process(self, sp_state, prev_state=None, learning=True):
        if prev_state is None:
            prev_state = self.last_state

        prev_cell_predictive = prev_state.cell_predictive[sp_state.column_active]
        column_bursting = ~np.any(prev_cell_predictive, axis=1)
        cell_active = prev_cell_predictive | np.expand_dims(column_bursting, 1)

        cell_predictive = np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_)
        cell_winner = cell_active.copy()

        curr_state = TemporalMemory.State(column_bursting, cell_predictive, cell_active, cell_winner)
        self.last_state = curr_state
        return curr_state


class HierarchicalTemporalMemory:
    def __init__(self, input_dim, column_dim, cell_dim, active_columns=None):
        if active_columns is None:
            active_columns = int(round(column_dim * 0.02))

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
            print(htm.spatial_pooler.boosting.duty_cycle.mean(), htm.spatial_pooler.boosting.duty_cycle.std())

    print(f'{time.time() - start_time}s')