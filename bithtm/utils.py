import numpy as np


try:
    np.bincount([0], minLength=1)
    bincount = np.bincount
except:
    def bincount(x, weights=None, minLength=0):
        out = np.bincount(x, weights=weights)
        out = np.concatenate([out, np.zeros(max(minLength - len(out), 0), dtype=out.dtype)])
        return out

def arange_concatenated(lengths, border_offsets=None, lengths_cumsum=None):
    if lengths_cumsum is None:
        lengths_cumsum = lengths.cumsum()
    total_length = lengths_cumsum[-1]
    nonempty_row, = np.nonzero(lengths)
    row_borders = lengths_cumsum - lengths
    row_index = np.zeros(total_length, dtype=np.int32)
    row_index[row_borders[nonempty_row]] = np.arange(len(lengths), dtype=np.int32)[nonempty_row]
    row_index = np.maximum.accumulate(row_index)
    if border_offsets is not None:
        row_borders -= border_offsets
    col_index = np.arange(total_length, dtype=np.int32) - row_borders[row_index]
    return row_index, col_index

def nonzero_bounded_2d(value, bounds, lengths=None, return_out_of_bounds=False):
    assert len(value.shape) == 2
    if lengths is None:
        lengths = (value != 0).sum(axis=1)
    bounded_lengths = np.minimum(lengths, bounds)
    lengths_cumsum = lengths.cumsum()
    bounded_lengths_cumsum = bounded_lengths.cumsum()
    border_offsets = lengths_cumsum - lengths
    row_index, col_index = arange_concatenated(bounded_lengths, border_offsets=border_offsets, lengths_cumsum=bounded_lengths_cumsum)
    _, col_nonzero = np.nonzero(value)
    nonzero_bounded = (row_index, col_nonzero[col_index])
    if return_out_of_bounds:
        oob_row_index, oob_col_index = arange_concatenated(lengths - bounded_lengths, border_offsets=bounded_lengths + border_offsets, lengths_cumsum=lengths_cumsum - bounded_lengths_cumsum)
        nonzero_oob = (oob_row_index, col_nonzero[oob_col_index])
        return nonzero_bounded, nonzero_oob
    return nonzero_bounded

def replace_free(free, dests, srcs, dest_index=None, nonempty_lengths=None, free_lengths=None, src_valid=None, src_lengths=None, return_indices=False, return_residue_info=False):
    assert len(dests[0].shape) == len(free.shape) == len(srcs[0].shape) == 2
    if free_lengths is None:
        free_lengths = free.sum(axis=1)
    if src_lengths is None:
        src_lengths = src_valid.sum(axis=1) if src_valid is not None else srcs[0].shape[1]
    if nonempty_lengths is not None:
        bounded_empty_lengths = np.minimum(dests.shape[1] - nonempty_lengths, src_lengths)
        local_empty_index = arange_concatenated(bounded_empty_lengths)
        local_empty_index = (local_empty_index[0], nonempty_lengths + local_empty_index[1])
        empty_index = local_empty_index
        if dest_index is not None:
            assert dest_index.shape[0] == free.shape[0]
            empty_index = (dest_index[local_empty_index[0]], local_empty_index[1])
        if src_valid is None:
            src_index = arange_concatenated(bounded_empty_lengths)
        else:
            src_index = nonzero_bounded_2d(src_valid, bounded_empty_lengths, lengths=src_lengths)
        for dest, src in zip(dests, srcs):
            if np.ndim(src) == 0:
                dest[empty_index] = src
                continue
            dest[empty_index] = src[src_index]
        src_lengths = src_lengths - bounded_empty_lengths
        if src_lengths.max() == 0:
            returned_values = [bounded_empty_lengths]
            if return_indices:
                returned_values += [empty_index, src_index]
            if return_residue_info:
                returned_values += [
                    np.zeros(src_lengths.shape, dtype=np.int32),
                    (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)),
                    (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
                ]
            return tuple(returned_values)
        if src_valid is not None:
            src_valid = src_valid.copy()
            src_valid[src_index] = False
        free_lengths = free_lengths - bounded_empty_lengths
        free = free.copy()
        free[local_empty_index] = False
    mutually_bounded_lengths = np.minimum(free_lengths, src_lengths)
    free_index = nonzero_bounded_2d(free, mutually_bounded_lengths, lengths=free_lengths)
    if return_residue_info:
        residue_lengths = src_lengths - mutually_bounded_lengths
        residue_index = arange_concatenated(residue_lengths)
    if dest_index is not None:
        assert dest_index.shape[0] == free.shape[0]
        free_index = (dest_index[free_index[0]], free_index[1])
    if src_valid is None:
        src_index = arange_concatenated(mutually_bounded_lengths)
        if return_residue_info:
            src_residue_index = (residue_index[0], residue_index[1] + mutually_bounded_lengths[residue_index[0]])
    else:
        src_index = nonzero_bounded_2d(src_valid, mutually_bounded_lengths, lengths=src_lengths, return_out_of_bounds=return_residue_info)
        if return_residue_info:
            src_index, src_residue_index = src_index
    for dest, src in zip(dests, srcs):
        if np.ndim(src) == 0:
            dest[free_index] = src
            continue
        dest[free_index] = src[src_index]
    returned_values = [mutually_bounded_lengths]
    if return_indices:
        returned_values += [free_index, src_index]
    if return_residue_info:
        returned_values += [residue_lengths, residue_index, src_residue_index]
    return tuple(returned_values)


class DynamicArray2D:
    def __init__(self, dtype, size=(0, 0), capacity=None, growth_exponential=(True, True), on_grow=None):
        if capacity is None:
            capacity = (0, 0)
        assert len(size) == len(capacity) == len(growth_exponential) == 2
        capacity = tuple(np.maximum(capacity, size))

        self.dtype = dtype
        self.capacity = tuple(capacity)
        self.growth_exponential = tuple(growth_exponential)
        self.on_grow = on_grow

        self.size = tuple(size)
        self.values = self.initialize_values(capacity=self.capacity)

    def initialize_values(self, capacity=None):
        if capacity is None:
            capacity = self.capacity
        return np.empty(capacity, dtype=self.dtype)

    def evaluate_capacity(self, length, axis):
        if not self.growth_exponential[axis]:
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
        new_size = list(self.size)
        new_size[axis] += added_values.shape[axis]
        if new_size[axis] > self.capacity[axis]:
            new_capacity = list(self.capacity)
            new_capacity[axis] = self.evaluate_capacity(new_size[axis], axis)
            new_values = self.initialize_values(capacity=new_capacity)
            new_values[:self.capacity[0], :self.capacity[1]] = self.values
            if self.on_grow is not None:
                self.on_grow(new_values, tuple(new_size), tuple(new_capacity), axis)
            self.capacity = tuple(new_capacity)
            self.values = new_values
        index = [slice(None, self.size[1 - axis])]
        index.insert(axis, slice(self.size[axis], new_size[axis]))
        self.values[tuple(index)] = added_values
        self.size = tuple(new_size)

    def add_rows(self, added_values):
        return self.add(added_values, 0)

    def add_cols(self, added_values):
        return self.add(added_values, 1)