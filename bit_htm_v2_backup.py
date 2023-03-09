import numpy as np
from collections import namedtuple


try:
    np.bincount([0], minLength=1)
    bincount = np.bincount
except:
    def bincount(x, weights=None, minLength=0):
        out = np.bincount(x, weights=weights)
        out = np.concatenate([out, np.zeros(max(minLength - len(out), 0), dtype=out.dtype)])
        return out

def construct_block_mapping(block_lengths):
    total_block_length = block_lengths.sum()
    nonempty_blocks, = np.nonzero(block_lengths)
    block_borders = block_lengths.cumsum() - block_lengths
    row_index = np.zeros(total_block_length, dtype=np.int32)
    row_index[block_borders[nonempty_blocks]] = np.arange(len(block_lengths))[nonempty_blocks]
    row_index = np.maximum.accumulate(row_index)
    col_index = np.arange(total_block_length, dtype=np.int32)
    col_index -= block_borders[row_index]
    return (row_index, col_index)

def nonzero_bounded_2d(value, bounds, offset=None, lengths=None):
    if offset is None:
        if lengths is None:
            lengths = (value != 0).sum(axis=1)
        _, offset = construct_block_mapping(lengths)
    index = np.nonzero(value)
    if type(bounds) == np.ndarray:
        bounds = bounds[index[0]]
    bounded = np.nonzero(offset < bounds)
    return (index[0][bounded], index[1][bounded])


class DenseProjection:
    def __init__(
        self, input_dim, output_dim,
        perm_mean=0.0, perm_std=0.1,
        permanence_threshold=0.0, permanence_increment=0.1, permanence_decrement=0.3
    ):
        self.permanence_threshold = permanence_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement

        self.permanence = np.random.randn(output_dim, input_dim) * perm_std + perm_mean

    def process(self, input):
        weight = self.permanence >= self.permanence_threshold
        overlaps = (weight & input).sum(axis=1)
        return overlaps

    def update(self, input, target):
        self.permanence[target] += input * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement


class ExponentialBoosting:
    def __init__(
        self, output_dim, actives,
        intensity=0.2, momentum=0.99
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
    def __init__(self, size=tuple(), dtypes=[np.float32], names=None, capacity=0, capacity_exponential=True):
        if names is not None:
            self.ReturnedArrays = namedtuple('ReturnedArray', names)
            self.name_to_index = dict(zip(names, range(len(names))))
        else:
            self.ReturnedArrays = tuple
            self.name_to_index = None

        self.length = 0
        self.capacity = capacity
        
        self.size = size
        self.dtypes = dtypes
        self.capacity_exponential = capacity_exponential
        
        self.arrays = self.initialize_arrays(self.capacity)

    def initialize_arrays(self, capacity):
        return tuple(np.empty(self.size + (capacity, ), dtype=dtype) for dtype in self.dtypes)

    def evaluate_capacity(self, length):
        if length == 0:
            return 0
        return 2 ** int(np.ceil(np.log2(length))) if self.capacity_exponential else length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.ReturnedArrays(*[array[..., :self.length][index] for array in self.arrays])

    def __setitem__(self, index, values):
        for array, value in zip(self.arrays, values):
            array[..., :self.length][index] = value

    def __getattr__(self, name):
        return self.arrays[self.name_to_index[name]][..., :self.length]

    def set_capcity(self, new_capacity):
        if new_capacity <= self.capacity:
            self.arrays = tuple(array[..., :new_capacity] for array in self.arrays)
            self.length = min(self.length, new_capacity)
            self.capacity = new_capacity
            return

        new_arrays = self.initialize_arrays(new_capacity)
        for old_array, new_array in zip(self.arrays, new_arrays):
            new_array[..., :self.length] = old_array[..., :self.length]
        self.arrays = new_arrays
        self.capacity = new_capacity

    def add(self, *added_arrays):
        new_length = self.length + added_arrays[0].shape[-1]
        if new_length > self.capacity:
            self.set_capcity(self.evaluate_capacity(new_length))
        for array, added_array in zip(self.arrays, added_arrays):
            array[..., self.length:new_length] = added_array
        self.length = new_length


class CompactDynamicArray(DynamicArray):
    def __init__(self, size=tuple(), dtypes=[np.float32], names=None, capacity=0, capacity_exponential=True, is_stale=None, initialize_override=None):
        self.is_stale = is_stale
        self.initialize_override = initialize_override

        self.lengths = np.zeros(size, dtype=np.int32)

        super().__init__(size=size, dtypes=dtypes, names=names, capacity=capacity, capacity_exponential=capacity_exponential)

    def initialize_arrays(self, capacity):
        if self.initialize_override is None:
            return super().initialize_arrays(capacity)
        return self.initialize_override(self.size + (capacity, ))

    def fill_blocks(self, index, *added_arrays, added_lengths=None):
        if added_lengths is None:
            added_lengths = added_arrays[0].shape[-1]

        block_offset = self.lengths[index]
        block_lengths = np.clip(self.capacity - block_offset, 0, added_lengths)
        block_index = construct_block_mapping(block_lengths)
        array_index = tuple(axis_index[block_index[0]] for axis_index in index) + (block_offset[block_index[0]] + block_index[1], )
        self.lengths[index] += block_lengths
        self.length = max(self.length, (block_offset + block_lengths).max(initial=0))
        self[array_index] = tuple(added_array[block_index] for added_array in added_arrays)
        return block_lengths

    def replace_stale(self, index, *added_arrays, added_lengths=None, prior_block_lengths=None, is_stale=None):
        if added_lengths is None:
            added_lengths = added_arrays[0].shape[-1]
        if is_stale is None:
            is_stale = self.is_stale

        entry_stale = is_stale(*self[:], index).reshape(-1, self.capacity)
        max_virtual_block_lengths = entry_stale.sum(axis=1)
        virtual_block_lengths = np.minimum(max_virtual_block_lengths, added_lengths)
        virtual_block_index = nonzero_bounded_2d(entry_stale, virtual_block_lengths, lengths=max_virtual_block_lengths)
        block_index = construct_block_mapping(virtual_block_lengths)

        array_index = tuple(axis_index[virtual_block_index[0]] for axis_index in index) + (virtual_block_index[1], )
        added_array_index = block_index if prior_block_lengths is None else (block_index[0], prior_block_lengths[block_index[0]] + block_index[1])
        self[array_index] = tuple(added_array[added_array_index] for added_array in added_arrays)
        return virtual_block_lengths

    def add(self, index, *added_arrays, added_lengths=None):
        if added_lengths is None:
            added_lengths = np.full(index[0].shape, added_arrays[0].shape[-1], dtype=np.int32)

        if len(index[0].shape) != 1:
            index = tuple(axis_index.reshape(-1) for axis_index in index)
            added_arrays = tuple(added_array.reshape(-1, added_array.shape[-1]) for added_array in added_arrays)
            added_lengths = tuple(added_length.reshape(-1) for added_length in added_lengths)

        if self.capacity > 0:
            block_lengths = self.fill_blocks(index, *added_arrays, added_lengths=added_lengths)
            added_lengths -= block_lengths
            if added_lengths.sum() <= 0:
                return

            if self.is_stale is not None:
                added_lengths -= self.replace_stale(index, *added_arrays, added_lengths=added_lengths, prior_block_lengths=block_lengths)
                if added_lengths.sum() <= 0:
                    return

        self.set_capcity(self.evaluate_capacity(self.capacity + added_lengths.max(initial=0)))
        self.fill_blocks(index, *added_arrays, added_lengths=added_lengths)


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
    def __init__(
        self, column_dim, cell_dim,
        segment_matching_threshold=10, segment_activation_threshold=10, segment_sampling_synapses=20,
        permanence_initial=0.01, permanence_threshold=0.5, permanence_increment=0.3, permanence_decrement=0.05, permanence_punishment=0.01,
        eps = 1e-8
    ):
        if segment_matching_threshold > segment_activation_threshold:
            raise NotImplementedError()

        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.segment_matching_threshold = segment_matching_threshold
        self.segment_activation_threshold = segment_activation_threshold
        self.segment_sampling_synapses = segment_sampling_synapses

        self.permanence_initial = permanence_initial
        self.permanence_threshold = permanence_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.permanence_punishment = permanence_punishment

        self.eps = eps

        self.segment = DynamicArray(dtypes=[np.int32], names=['cell'])
        self.cell_segments = np.zeros((self.column_dim, self.cell_dim), dtype=np.int32)
        self.cell_synapse = CompactDynamicArray(
            size=(self.column_dim, self.cell_dim),
            dtypes=[np.int32, np.float32],
            names=['segment', 'permanence'],
            is_stale=lambda _, permanence, index: permanence[index] <= 0.0,
            initialize_override=lambda shape: (np.random.randint(0, len(self.segment), shape, dtype=np.int32), np.full(shape, -1.0, dtype=np.float32)),
            capacity_exponential=False
        )

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
        column_bursting = ~prev_cell_prediction.any(axis=1)
        
        active_cell = np.where(prev_cell_prediction | np.expand_dims(column_bursting, 1))
        active_cell = (sp_state.active_column[active_cell[0]], active_cell[1])

        if learning or return_winner_cell:
            column_activation = np.zeros(self.column_dim, dtype=np.bool_)
            column_activation[sp_state.active_column] = True

            matching_segment, = np.where(prev_state.segment_potential >= self.segment_matching_threshold)
            segment_cell = self.segment.cell[matching_segment]
            segment_column = segment_cell // self.cell_dim
            segment_potential = prev_state.segment_potential[matching_segment]

            # consider optimizing
            segment_potential_jittered = (segment_potential + np.random.rand(*segment_potential.shape)).astype(np.float32)
            column_max_potential = np.zeros(self.column_dim, dtype=np.float32)
            np.maximum.at(column_max_potential, segment_column, segment_potential_jittered)
            column_max_potential *= column_activation

            segment_growing_active_column, = np.where(column_max_potential[sp_state.active_column] < self.segment_matching_threshold)
            segment_growing_column = sp_state.active_column[segment_growing_active_column]
            cell_segments_jittered = self.cell_segments[segment_growing_column].astype(np.float32)
            cell_segments_jittered += np.random.rand(*cell_segments_jittered.shape)
            least_used_cell = np.argmin(cell_segments_jittered, axis=1)

            winner_cell = prev_cell_prediction.copy()
            winner_cell[segment_growing_active_column, least_used_cell] = True
            winner_cell = np.where(winner_cell)
            winner_cell = (sp_state.active_column[winner_cell[0]], winner_cell[1])
            
        if learning:
            segment_active = prev_state.segment_activation[matching_segment] >= self.segment_activation_threshold
            segment_best_matching = np.abs(segment_potential_jittered - column_max_potential[segment_column]) < self.eps
            segment_learning = segment_active | segment_best_matching

            # TODO: factor out
            target_segment, permanence = self.cell_synapse[prev_state.active_cell]
            matching_segment_synapse = np.where((prev_state.segment_potential[target_segment] >= self.segment_matching_threshold) & (permanence > 0.0))
            matching_segment_synapse = (prev_state.active_cell[0][matching_segment_synapse[0]], prev_state.active_cell[1][matching_segment_synapse[0]], matching_segment_synapse[1])
            total_segment_to_matching = np.empty(len(self.segment), dtype=np.int32)
            total_segment_to_matching[matching_segment] = np.arange(len(matching_segment))

            target_segment = total_segment_to_matching[self.cell_synapse.segment[matching_segment_synapse]]
            target_column = segment_column[target_segment]
            column_punishment = prev_state.cell_prediction.any(axis=1) & (~column_activation)
            permanence = self.cell_synapse.permanence
            permanence[matching_segment_synapse] += segment_learning[target_segment] * (column_activation[target_column] * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement)
            permanence[matching_segment_synapse] -= column_punishment[target_column] * self.permanence_punishment

            prev_segments = len(self.segment)
            self.segment.add(segment_growing_column * self.cell_dim + least_used_cell)
            self.cell_segments[segment_growing_column, least_used_cell] += 1

            # learning_segment = np.concatenate([matching_segment[segment_learning], np.arange(prev_segments, len(self.segment))])
            # target_segment = np.tile(learning_segment, (len(prev_state.winner_cell[0]), 1))
            # permanence = np.full(target_segment.shape, self.permanence_initial, dtype=np.float32)
            # self.cell_synapse.add(prev_state.winner_cell, target_segment, permanence)

            learning_segment = matching_segment[segment_learning]
            segment_learning = np.zeros(prev_segments, dtype=np.bool_)
            segment_learning[learning_segment] = True

            # TODO: factor out
            target_segment, permanence = self.cell_synapse[prev_state.winner_cell]
            learning_segment_synapse = np.where(segment_learning[target_segment] & (permanence > 0.0))
            learning_segment_synapse = (prev_state.winner_cell[0][learning_segment_synapse[0]], prev_state.winner_cell[1][learning_segment_synapse[0]], learning_segment_synapse[1])
            total_segment_to_learning = np.empty(prev_segments, dtype=np.int32)
            total_segment_to_learning[learning_segment] = np.arange(len(learning_segment))

            target_segment, permanence = self.cell_synapse[learning_segment_synapse]
            target_segment = total_segment_to_learning[target_segment]
            segment_cell_already_connected = np.zeros((len(learning_segment), len(prev_state.winner_cell[0])), dtype=np.bool_)
            segment_cell_already_connected[target_segment, learning_segment_synapse[0]]


            

        target_segment, permanence = map(np.ndarray.flatten, self.cell_synapse[active_cell])
        segment_activation = bincount(target_segment, weights=permanence >= self.permanence_threshold, minLength=len(self.segment)).astype(np.int32)
        segment_potential = bincount(target_segment, weights=permanence > 0.0, minLength=len(self.segment)).astype(np.int32)

        cell_prediction = bincount(self.segment.cell, weights=segment_activation >= self.segment_activation_threshold, minLength=self.column_dim * self.cell_dim)
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
    inputs = np.random.rand(10, 1000) < 0.2
    htm = HierarchicalTemporalMemory(inputs.shape[1], 2048, 32)

    import time

    start_time = time.time()

    for epoch in range(100):
        for curr_input in inputs:
            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.any(axis=1)
            
            noisy_input = curr_input # ^ (np.random.rand(*curr_input.shape) < 0.05)
            sp_state, tm_state = htm.process(noisy_input)

            burstings = tm_state.column_bursting.sum()
            corrects = prev_column_prediction[sp_state.active_column].sum()
            incorrects = prev_column_prediction.sum() - corrects
            print(burstings, corrects, incorrects)

    print(f'{time.time() - start_time}s')