import numpy as np


try:
    np.bincount([0], minLength=1)
    bincount = np.bincount
except:
    def bincount(x, weights=None, minLength=0):
        out = np.bincount(x, weights=weights)
        out = np.concatenate([out, np.zeros(max(minLength - len(out), 0), dtype=out.dtype)])
        return out

def arange_concatenated(lengths, border_offset=None):
    total_length = lengths.sum()
    nonempty_row, = np.nonzero(lengths)
    row_border = lengths.cumsum() - lengths
    row_index = np.zeros(total_length, dtype=np.int32)
    row_index[row_border[nonempty_row]] = np.arange(len(lengths), dtype=np.int32)[nonempty_row]
    row_index = np.maximum.accumulate(row_index)
    if border_offset is not None:
        row_border -= border_offset
    col_index = np.arange(total_length, dtype=np.int32) - row_border[row_index]
    return row_index, col_index

def nonzero_bounded_2d(value, bounds, lengths=None):
    assert len(value.shape) == 2
    if lengths is None:
        lengths = (value != 0).sum(axis=1)
    bounded_lengths = np.minimum(lengths, bounds)
    row_index, col_index = arange_concatenated(bounded_lengths, border_offset=lengths.cumsum() - lengths)
    _, col_nonzero = np.nonzero(value)
    return row_index, col_nonzero[col_index]

def replace_free(dests, srcs, free, dest_index=None, free_lengths=None, src_valid=None, src_lengths=None):
    assert len(dests[0].shape) == len(free.shape) == len(srcs[0].shape) == 2
    if free_lengths is None:
        free_lengths = free.sum(axis=1)
    if src_lengths is None:
        src_lengths = src_valid.sum(axis=1) if src_valid is not None else srcs[0].shape[1]
    mutually_bounded_lengths = np.maximum(free_lengths, src_lengths)
    free_index = nonzero_bounded_2d(free, mutually_bounded_lengths, lengths=free_lengths)
    if dest_index is not None:
        assert dest_index.shape[0] == free.shape[0]
        free_index = (dest_index[free_index[0]], free_index[1])
    if src_valid is None:
        src_index = arange_concatenated(mutually_bounded_lengths)
    else:
        src_index = nonzero_bounded_2d(src_valid, mutually_bounded_lengths, lengths=src_lengths)
    for dest, src in zip(dests, srcs):
        if np.ndim(src) == 0:
            dest[free_index] = src
            continue
        dest[free_index] = src[src_index]


class DynamicArray2D:
    def __init__(self, dtype, size=(0, 0), capacities=None, exponential_growths=(True, True), on_grow=None):
        if capacities is None:
            capacities = tuple(size)
        assert len(size) == len(capacities) == len(exponential_growths) == 2

        self.dtype = dtype
        self.capacities = tuple(capacities)
        self.exponential_growths = tuple(exponential_growths)
        self.on_grow = on_grow

        self.size = tuple(size)
        self.values = self.initialize_values(capacities=self.capacities)

    def initialize_values(self, capacities=None):
        if capacities is None:
            capacities = self.capacities
        return np.empty(capacities, dtype=self.dtype)

    def evaluate_capacity(self, length, axis):
        if not self.exponential_growths[axis]:
            return length
        return 2 ** int(np.ceil(np.log2(length)))

    def __len__(self):
        return self.size[0]

    def __getitem__(self, index):
        return self.values[:self.size[0], :self.size[1]][index]

    def __setitem__(self, index, new_values):
        self.values[:self.size[0], :self.size[1]][index] = new_values

    def add(self, added_values, axis):
        assert len(added_values.shape) == 2 and added_values.shape[1 - axis] == self.size[1 - axis]
        new_length = self.size[axis] + added_values.shape[axis]
        new_size = list(self.size)
        new_size[axis] = new_length
        if new_length > self.capacities[axis]:
            new_capacities = list(self.capacities)
            new_capacities[axis] = self.evaluate_capacity(new_length, axis)
            new_values = self.initialize_values(capacities=new_capacities)
            new_values[:self.size[0], :self.size[1]] = self.values[:self.size[0], :self.size[1]]
            if self.on_grow is not None:
                self.on_grow(new_values, new_size, new_capacities)
            self.capacities = new_capacities
            self.values = new_values
        if axis == 0:
            self.values[self.size[0]:new_length, :self.size[1]] = added_values
        else:
            self.values[:self.size[0], self.size[1]:new_length] = added_values
        self.size = tuple(new_size)

    def add_rows(self, added_values):
        return self.add(added_values, 0)

    def add_cols(self, added_values):
        return self.add(added_values, 1)


class DenseProjection:
    def __init__(
        self, input_dim, output_dim,
        permanence_mean=0.0, permanence_std=0.1,
        permanence_threshold=0.0, permanence_increment=0.1, permanence_decrement=0.3
    ):
        self.permanence_threshold = permanence_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement

        self.permanence = np.random.randn(output_dim, input_dim) * permanence_std + permanence_mean

    def process(self, dense_input):
        weight = self.permanence >= self.permanence_threshold
        overlaps = (weight & dense_input).sum(axis=1)
        return overlaps

    def update(self, dense_input, learning_output):
        self.permanence[learning_output] += dense_input * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement


class SparseProjection:
    projection_invalid_flag = 0x1 << 31

    def __init__(
        self, input_dim, output_dim=0,
        projection_exponential_growth=False, output_exponential_growth=True
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        def on_projection_grow(new_values, new_size, new_capacities):
            assert np.log2(self.input_dim * new_capacities[1]) <= 31
            macro, micro = divmod(self.backprojection[:], self.projection.capacities[1])
            self.backprojection[:] = macro * new_capacities[1] + micro

        self.projection = DynamicArray2D(np.int32, size=(self.input_dim, 0), exponential_growths=(False, projection_exponential_growth), on_grow=on_projection_grow)
        self.backprojection = DynamicArray2D(np.int32, size=(self.output_dim, 0), exponential_growths=[output_exponential_growth, projection_exponential_growth])
        self.backprojection_permanence = DynamicArray2D(np.float32, size=(self.output_dim, 0), exponential_growths=[output_exponential_growth, projection_exponential_growth])

    def get_pure_projection(self, flaggable_projection):
        return flaggable_projection & (~SparseProjection.projection_invalid_flag)

    def get_projection_validity(self, flaggable_projection):
        return (flaggable_projection & SparseProjection.projection_invalid_flag) == 0x0

    def get_projection_invalidity(self, flaggable_projection):
        return (flaggable_projection & SparseProjection.projection_invalid_flag) != 0x0

    def decompose_flaggable_projection(self, flaggable_projection):
        return self.get_pure_projection(flaggable_projection), self.get_projection_validity(flaggable_projection)

    def add_output_dim(self, added_output_dim):
        assert added_output_dim >= 0
        self.output_dim += added_output_dim
        assert np.log2(self.output_dim) <= 31
        self.backprojection.add_rows(np.random.randint(0, self.input_dim * self.projection.capacities[1], (added_output_dim, self.backprojection.size[1]), dtype=self.backprojection.dtype))
        self.backprojection_permanence.add_rows(np.full((added_output_dim, 1), -1.0, dtype=self.backprojection_permanence.dtype))

    def update_permanence(self, input_activation, learning_output, permanence_chage_active, permanence_change_inactive, return_projection_info=False):
        learning_backprojection_macro, learning_backprojection_micro = divmod(self.backprojection[learning_output], self.projection.capacities[1])
        learning_backprojection_permanence = self.backprojection_permanence[learning_output]
        learning_backprojection_valid = learning_backprojection_permanence > 0.0
        backprojected_activation = input_activation[learning_backprojection_macro]
        naive_permanence_change = backprojected_activation * (permanence_chage_active - permanence_change_inactive) + permanence_change_inactive
        updated_permanence = learning_backprojection_permanence + learning_backprojection_valid * naive_permanence_change
        self.backprojection_permanence[learning_output] = updated_permanence
        self.projection[
            learning_backprojection_macro,
            learning_backprojection_micro
        ] &= ((updated_permanence <= 0.0) * SparseProjection.projection_invalid_flag) | (~SparseProjection.projection_invalid_flag)

        if return_projection_info:
            return learning_backprojection_macro, learning_backprojection_micro, learning_backprojection_permanence, learning_backprojection_valid

    def add_projections(
        self, input_activation, learning_output, winner_input, permanence_initial, min_active_projections,
        learning_backprojection_macro=None, learning_backprojection_valid=None
    ):
        assert permanence_initial >= 0.0
        if learning_backprojection_macro is None:
            learning_backprojection_macro = self.backprojection[learning_output] // self.projection.capacities[1]
        if learning_backprojection_valid is None:
            learning_backprojection_valid = self.backprojection_permanence[learning_output] > 0.0

        whole_input_to_winner = np.random.randint(0, len(winner_input), self.projection.size[0], dtype=np.int32) | SparseProjection.projection_invalid_flag
        whole_input_to_winner[winner_input] = np.arange(len(winner_input))
        backprojection_winner_macro, backprojection_winner_macro_valid = self.decompose_flaggable_projection(whole_input_to_winner[learning_backprojection_macro])
        candidate_priority = np.random.rand(len(learning_output), len(winner_input)).astype(np.float32)
        candidate_priority += bincount(
            np.repeat(np.arange(len(learning_output)), self.backprojection.size[1]) * len(winner_input) + backprojection_winner_macro.flatten(),
            weights=learning_backprojection_valid.flatten() & backprojection_winner_macro_valid.flatten(),
            minLength=len(learning_output) * len(winner_input)
        ).reshape(candidate_priority.shape)
        
        added_projections = np.clip(min_active_projections - input_activation[learning_backprojection_macro].sum(axis=1), 0, min(min_active_projections, len(winner_input)))
        max_added_projections = added_projections.max()
        priority_argsort = np.argpartition(candidate_priority, min(max_added_projections, candidate_priority.shape[1] - 1), axis=1)[:, :max_added_projections]

        candidate = winner_input[priority_argsort]
        unconnected = np.take_along_axis(candidate_priority, priority_argsort, axis=1) < 1.0
        candidate_prioritized = np.tile(np.arange(max_added_projections), (len(learning_output), 1)) < np.expand_dims(added_projections, 1)
        candidate_valid = unconnected & candidate_prioritized
        
        # replace_free(
        #     [self.projection],
        #     [],
        #     winner_input, self.get_projection_invalidity(self.projection[winner_input]), added_lengths=None
        # )

        # TODO: account for micro. do the forward projections first.
        resolved_backprojections = replace_free(
            [self.backprojection, self.backprojection_permanence],
            [candidate, permanence_initial],
            ~learning_backprojection_valid, desc_index=learning_output, src_valid=candidate_valid, src_lengths=added_projections
        )
        remaining_backprojections = added_projections - resolved_backprojections
        max_remaining_backprojections = remaining_backprojections.max()
        if max_remaining_backprojections > 0:
            added_macro, added_micro = arange_concatenated(remaining_backprojections)
            learning_added_macro = learning_output[added_macro]
            added_backprojection = np.random.randint(0, self.input_dim * self.projection.capacities[1], (self.backprojection.size[0], max_remaining_backprojections), dtype=np.int32)
            added_backprojection_permanence = np.full((self.backprojection.size[0], max_remaining_backprojections), -1.0, dtype=np.float32)
            added_backprojection[learning_added_macro, added_micro] = candidate[added_macro, added_micro + resolved_backprojections[added_macro]]
            added_backprojection_permanence[learning_added_macro, added_micro] = permanence_initial
            self.backprojection.add_cols(added_backprojection)
            self.backprojection_permanence.add_cols(added_backprojection_permanence)

    def process(self, active_input=None, dense_input=None):
        assert (active_input is None) ^ (dense_input is None)
        if dense_input is not None:
            active_input = np.nonzero(dense_input)

        active_projection, active_weight = self.decompose_flaggable_projection(self.projection[active_input].flatten())
        projected = bincount(active_projection, weights=active_weight, minLength=self.output_dim)        
        
        assert projected.shape[0] == self.output_dim
        return projected

    def update(
        self, input_activation, learning_output, winner_input=None, added_output_dim=0,
        permanence_initial=0.01, permanence_chage_active=0.3, permanence_change_inactive=-0.05,
        min_active_projections=20
    ):
        if added_output_dim > 0:
            self.add_output_dim(added_output_dim)
        learning_backprojection_macro, _, _, learning_backprojection_valid = self.update_permanence(
            input_activation, learning_output, permanence_chage_active, permanence_change_inactive, return_projection_info=True
        )
        if winner_input is not None:
            self.add_projections(
                input_activation, learning_output, winner_input, permanence_initial, min_active_projections,
                learning_backprojection_macro=learning_backprojection_macro, learning_backprojection_valid=learning_backprojection_valid
            )


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


class SpatialPooler:
    class State:
        def __init__(self, overlaps, boosted_overlaps, active_column):
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps
            self.active_column = active_column

    def __init__(
        self, input_dim, column_dim, active_columns,
        proximal_projection=None, boosting=None, inhibition=None
    ):
        self.input_dim = input_dim
        self.column_dim = column_dim
        self.active_columns = active_columns

        self.proximal_projection = proximal_projection or DenseProjection(input_dim, column_dim)
        self.boosting = boosting or ExponentialBoosting(column_dim, active_columns)
        self.inhibition = inhibition or GlobalInhibition(active_columns)

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
        def __init__(self):
            pass

    # TODO: active_columns is temp
    def __init__(
        self, column_dim, cell_dim,
        segment_matching_threshold=10, segment_activation_threshold=10, segment_sampling_synapses=20,
        permanence_initial=0.01, permanence_threshold=0.5, permanence_increment=0.3, permanence_decrement=0.05, permanence_punishment=0.01
    ):
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
        
        

        self.last_state = TemporalMemory.State()

    # TODO: implement return_winner_cell for real
    def process(self, sp_state, prev_state=None, learning=True, return_winner_cell=True):
        if prev_state is None:
            prev_state = self.last_state

        

        curr_state = TemporalMemory.State()
        self.last_state = curr_state
        return curr_state


class HierarchicalTemporalMemory:
    def __init__(
        self, input_dim, column_dim, cell_dim, active_columns=None,
        spatial_pooler=None, temporal_memory=None
    ):
        active_columns = active_columns or round(column_dim * 0.02)

        self.spatial_pooler = spatial_pooler or SpatialPooler(input_dim, column_dim, active_columns)
        self.temporal_memory = temporal_memory or TemporalMemory(column_dim, cell_dim)

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