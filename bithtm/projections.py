from .utils import DynamicArray2D, bincount, replace_free

import numpy as np


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

    def process(self, input_activation):
        weight = self.permanence >= self.permanence_threshold
        overlaps = (weight & input_activation).sum(axis=1)
        return overlaps

    def update(self, input_activation, learning_output):
        self.permanence[learning_output] += input_activation * (self.permanence_increment + self.permanence_decrement) - self.permanence_decrement


# class SparseProjection:
#     projection_dtype = np.int32
#     backprojection_dtype = np.int32
#     projection_invalid_flag = projection_dtype(0x1) << 31

#     def __init__(
#         self, input_dim, output_dim=0,
#         projection_growth_exponential=True, output_growth_exponential=True
#     ):
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         # TODO: macro, micro for projection, not backprojection (use winner_input for permanence update) <- inactive cells' validity should change as well. work around it.
#         # maybe instead of the flag, use separate binary weight matrix
#         # differentiate between projection and connection

#         def on_projection_grow(new_values, new_size, new_capacity, axis):
#             assert axis == 1
#             assert np.log2(self.input_dim * new_capacity[1]) <= 31
#             new_values[:, new_size[1]:new_capacity[1]] = np.random.randint(
#                 0, self.output_dim,
#                 (self.projection.capacity[0], new_capacity[1] - new_size[1]), dtype=self.projection_dtype
#             ) | self.projection_invalid_flag
#             macro, micro = divmod(self.backprojection[:], self.projection.capacity[1])
#             self.backprojection[:] = macro * new_capacity[1] + micro

#         def on_backprojection_grow(new_values, new_size, new_capacity, axis):
#             index = [slice(None)]
#             index.insert(axis, slice(new_size[axis], new_capacity[axis]))
#             empty_size = [self.backprojection.capacity[1 - axis]]
#             empty_size.insert(axis, new_capacity[axis] - new_size[axis])
#             new_values[tuple(index)] = np.random.randint(0, self.input_dim * self.projection.capacity[1], tuple(empty_size), dtype=self.backprojection_dtype)

#         def on_backprojection_permanence_grow(new_values, new_size, new_capacity, axis):
#             index = [slice(None)]
#             index.insert(axis, slice(new_size[axis], new_capacity[axis]))
#             new_values[tuple(index)] = -1.0

#         self.projection = DynamicArray2D(self.projection_dtype, size=(self.input_dim, 0), growth_exponentials=(False, projection_growth_exponential), on_grow=on_projection_grow)
#         self.backprojection = DynamicArray2D(self.backprojection_dtype, size=(self.output_dim, 0), growth_exponentials=[output_growth_exponential, projection_growth_exponential], on_grow=on_backprojection_grow)
#         self.backprojection_permanence = DynamicArray2D(np.float32, size=(self.output_dim, 0), growth_exponentials=[output_growth_exponential, projection_growth_exponential], on_grow=on_backprojection_permanence_grow)
#         self.backprojection_valids = DynamicArray2D(self.backprojection_dtype, size=(self.output_dim, 1), growth_exponentials=[output_growth_exponential, False]) # TODO

#     def get_pure_projection(self, flaggable_projection):
#         return flaggable_projection & (~self.projection_invalid_flag)

#     def get_projection_validity(self, flaggable_projection):
#         return (flaggable_projection & self.projection_invalid_flag) == 0x0

#     def get_projection_invalidity(self, flaggable_projection):
#         return (flaggable_projection & self.projection_invalid_flag) != 0x0

#     def decompose_flaggable_projection(self, flaggable_projection):
#         return self.get_pure_projection(flaggable_projection), self.get_projection_validity(flaggable_projection)

#     def add_output(self, added_outputs, valids_threshold):
#         assert added_outputs >= 0
        
#         replaced_output, = np.where(self.backprojection_valids[:].squeeze(1) < valids_threshold)
#         # replaced_output, = np.where((self.backprojection_permanence[:] > 0.0).sum(axis=1) < valids_threshold)
#         replaced_output = replaced_output[:added_outputs]
#         replaced_backprojection = self.backprojection[replaced_output][self.backprojection_permanence[replaced_output] > 0.0]
#         self.projection[divmod(replaced_backprojection, self.projection.capacity[1])] |= self.projection_invalid_flag
#         self.backprojection_permanence[replaced_output] = -1.0
#         self.backprojection_valids[replaced_output] = 0.0
#         added_outputs -= len(replaced_output)
#         if added_outputs == 0:
#             return replaced_output, np.empty(0, dtype=np.int32)

#         self.output_dim += added_outputs
#         assert np.log2(self.output_dim) <= 31
#         self.backprojection.add_rows(np.random.randint(0, self.input_dim * self.projection.capacity[1], (added_outputs, self.backprojection.size[1]), dtype=self.backprojection.dtype))
#         self.backprojection_permanence.add_rows(np.full((added_outputs, self.backprojection.size[1]), -1.0, dtype=self.backprojection_permanence.dtype))
#         self.backprojection_valids.add_rows(np.zeros((added_outputs, 1), dtype=self.backprojection_dtype))
#         # return np.concatenate([replaced_output, np.arange(self.output_dim - added_outputs, self.output_dim)])
#         return replaced_output, np.arange(self.output_dim - added_outputs, self.output_dim)

#     def update_permanence(self, input_activation, learning_output, permanence_change_active, permanence_change_inactive, return_projection_info=False):
#         learning_backprojection_macro, learning_backprojection_micro = divmod(self.backprojection[learning_output], self.projection.capacity[1])
#         learning_backprojection_permanence = self.backprojection_permanence[learning_output]
#         learning_backprojection_valid = learning_backprojection_permanence > 0.0
#         backprojected_activation = input_activation[learning_backprojection_macro]
#         naive_permanence_change = backprojected_activation * (permanence_change_active - permanence_change_inactive) + permanence_change_inactive
#         updated_permanence = learning_backprojection_permanence + learning_backprojection_valid * naive_permanence_change
#         self.backprojection_permanence[learning_output] = updated_permanence

#         invalidated_learning_backprojection = np.where(learning_backprojection_valid & (updated_permanence <= 0.0))
#         self.projection.values[
#             learning_backprojection_macro[invalidated_learning_backprojection],
#             learning_backprojection_micro[invalidated_learning_backprojection]
#         ] |= self.projection_invalid_flag
#         self.backprojection_valids[learning_output] -= (learning_backprojection_valid & (updated_permanence <= 0.0)).sum(axis=1, keepdims=True)
        
#         if return_projection_info:
#             return learning_backprojection_macro, learning_backprojection_micro, updated_permanence

#     def add_projections(
#         self, input_activation, learning_output, winner_input, permanence_initial, min_active_projections,
#         learning_backprojection_macro=None, learning_backprojection_permanence=None
#     ):
#         if min(len(learning_output), len(winner_input)) == 0:
#             return

#         assert permanence_initial >= 0.0
#         if learning_backprojection_macro is None:
#             learning_backprojection_macro = self.backprojection[learning_output] // self.projection.capacity[1]
#         if learning_backprojection_permanence is None:
#             learning_backprojection_permanence = self.backprojection_permanence[learning_output]
#         learning_backprojection_valid = learning_backprojection_permanence > 0.0

#         learning_output_index = np.arange(len(learning_output))
#         whole_input_to_winner = np.random.randint(0, len(winner_input), self.projection.size[0], dtype=self.projection_dtype) | self.projection_invalid_flag
#         whole_input_to_winner[winner_input] = np.arange(len(winner_input))
#         backprojection_winner_macro, backprojection_winner_macro_valid = self.decompose_flaggable_projection(whole_input_to_winner[learning_backprojection_macro])
#         candidate_priority = np.random.rand(len(learning_output), len(winner_input)).astype(np.float32)
#         candidate_priority += bincount(
#             (np.expand_dims(learning_output_index, 1) * len(winner_input) + backprojection_winner_macro).flatten(),
#             weights=(learning_backprojection_valid & backprojection_winner_macro_valid).flatten(),
#             minLength=len(learning_output) * len(winner_input)
#         ).reshape(candidate_priority.shape)
        
#         active_learning_backprojections = (input_activation[learning_backprojection_macro] & learning_backprojection_valid).sum(axis=1)
#         added_backprojections = np.clip(min_active_projections - active_learning_backprojections, 0, min(min_active_projections, len(winner_input)))
#         # candidate_unconnected = candidate_priority < 1.0
#         # candidate_prioritized = np.argpartition(candidate_priority, added_backprojections.max(initial=0), axis=1) < np.expand_dims(added_backprojections, 1)
#         # candidate_picked = candidate_unconnected & candidate_prioritized
#         max_added_backprojections = added_backprojections.max(initial=0)
#         candidate_index_prioritized = np.argpartition(candidate_priority, max_added_backprojections, axis=1)[:, :max_added_backprojections]
#         candidate_unconnected = candidate_priority < 1.0
#         candidate_prioritized = np.zeros((len(learning_output), len(winner_input)), dtype=np.bool_)
#         np.put_along_axis(candidate_prioritized, candidate_index_prioritized, np.expand_dims(np.arange(max_added_backprojections), 0) < np.expand_dims(added_backprojections, 1), axis=1)
#         candidate_picked = candidate_unconnected & candidate_prioritized

#         projection_added = np.tile(learning_output, (len(winner_input), 1))
#         resovled_projections, free_index, src_index, residue_projections, residue_index, src_residue_index = replace_free(
#             [self.projection[:]], [projection_added], self.get_projection_invalidity(self.projection[winner_input]),
#             dest_index=winner_input, src_valid=candidate_picked.T, return_indices=True, return_residue_info=True
#         )
#         max_new_projections = residue_projections.max(initial=0)
#         if max_new_projections > 0:
#             prev_max_projection_micros = self.projection.size[1]
#             new_projection = (winner_input[residue_index[0]], residue_index[1])
#             projection_new = np.random.randint(0, self.output_dim, (self.projection.size[0], max_new_projections), dtype=self.projection_dtype) | self.projection_invalid_flag
#             projection_new[new_projection] = learning_output[src_residue_index[1]]
#             self.projection.add_cols(projection_new)

#         backprojection_added = np.empty((len(learning_output), len(winner_input)), dtype=self.backprojection_dtype)
#         backprojection_added[src_index[::-1]] = free_index[0] * self.projection.capacity[1] + free_index[1]
#         if max_new_projections > 0:
#             backprojection_added[src_residue_index[::-1]] = new_projection[0] * self.projection.capacity[1] + (new_projection[1] + prev_max_projection_micros)
#         resolved_backprojections, residue_backprojections, residue_index, src_residue_index = replace_free(
#             [self.backprojection[:], self.backprojection_permanence[:]],
#             [backprojection_added, permanence_initial],
#             ~learning_backprojection_valid, dest_index=learning_output, src_valid=candidate_picked, src_lengths=added_backprojections, return_residue_info=True
#         )
#         max_new_backprojections = residue_backprojections.max(initial=0)
#         if max_new_backprojections > 0:
#             new_backprojection = (learning_output[residue_index[0]], residue_index[1])
#             backprojection_new = np.random.randint(0, self.input_dim * self.projection.capacity[1], (self.backprojection.size[0], max_new_backprojections), dtype=self.projection_dtype)
#             backprojection_permanence_new = np.full(backprojection_new.shape, -1.0, dtype=np.float32)
#             backprojection_new[new_backprojection] = backprojection_added[src_residue_index]
#             backprojection_permanence_new[new_backprojection] = permanence_initial
#             self.backprojection.add_cols(backprojection_new)
#             self.backprojection_permanence.add_cols(backprojection_permanence_new)
#         self.backprojection_valids[learning_output] += np.expand_dims(added_backprojections, 1)

#     def process(self, active_input=None, dense_input=None, invoked_output=None, permanence_threshold=None):
#         assert (active_input is None) ^ (dense_input is None)
#         if (invoked_output is not None and dense_input is None) or (permanence_threshold is not None and invoked_output is None):
#             raise NotImplementedError()

#         if invoked_output is not None:
#             backprojection_weight = self.backprojection_permanence[invoked_output] >= permanence_threshold if permanence_threshold is not None else np.bool_(True)
#             backprojection_macro = self.backprojection[invoked_output] // self.projection.capacity[1]
#             projected = (dense_input[backprojection_macro] & backprojection_weight).sum(axis=1)
#             return projected

#         if dense_input is not None:
#             active_input, = np.where(dense_input)
#         active_projection, active_weight = self.decompose_flaggable_projection(self.projection[active_input].flatten())
#         projected = bincount(active_projection, weights=active_weight, minLength=self.output_dim)        
#         assert projected.shape[0] == self.output_dim
#         return projected

#     def update(
#         self, input_activation, learning_output, winner_input=None,
#         permanence_initial=0.01, permanence_change_active=0.3, permanence_change_inactive=-0.05,
#         min_active_projections=20
#     ):
#         learning_backprojection_macro, _, learning_backprojection_permanence = self.update_permanence(
#             input_activation, learning_output, permanence_change_active, permanence_change_inactive, return_projection_info=True
#         )
#         if winner_input is not None:
#             self.add_projections(
#                 input_activation, learning_output, winner_input, permanence_initial, min_active_projections,
#                 learning_backprojection_macro=learning_backprojection_macro, learning_backprojection_permanence=learning_backprojection_permanence
#             )


class SparseProjection:
    def __init__(
        self, input_dim, output_dim=0,
        output_growth_exponential=True, edge_growth_exponential=True
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.invalid_input_edge = 0
        self.invalid_output_edge = self.input_dim

        self.input_edge = DynamicArray2D(np.int32, size=(self.input_dim + 1, 0), growth_exponential=(False, edge_growth_exponential), on_grow=self.on_input_edge_grow)
        
        self.output_edges = DynamicArray2D(np.int32, size=(self.output_dim, 1), growth_exponential=(output_growth_exponential, False))
        self.output_edge = DynamicArray2D(np.int32, size=(self.output_dim, 0), growth_exponential=(output_growth_exponential, edge_growth_exponential), on_grow=self.on_output_edge_grow)
        self.output_permanence = DynamicArray2D(np.float32, size=(self.output_dim, 0), growth_exponential=(output_growth_exponential, edge_growth_exponential), on_grow=self.on_output_permanence_grow)
        
    def on_input_edge_grow(self, new_values, new_size, new_capacity, axis):
        assert axis == 1
        new_values[:, self.input_edge.capacity[1]:new_capacity[1]] = self.invalid_input_edge

    def on_output_edge_grow(self, new_values, new_size, new_capacity, axis):
        index = [slice(None)]
        index.insert(axis, slice(self.output_edge.capacity[axis], new_capacity[axis]))
        new_values[tuple(index)] = self.invalid_output_edge

    def on_output_permanence_grow(self, new_values, new_size, new_capacity, axis):
        index = [slice(None)]
        index.insert(axis, slice(self.output_edge.capacity[axis], new_capacity[axis]))
        new_values[tuple(index)] = -1.0

    def get_output_edge_target(self, output_edge):
        return output_edge % self.input_dim

    def pack_output_edge(self, target_input, input_edge):
        return input_edge * self.input_dim + target_input

    def unpack_output_edge(self, output_edge):
        input_edge, target_input = np.divmod(output_edge, self.input_dim)
        return target_input, input_edge

    def pad_input_activation(self, active_input=None, input_activation=None):
        assert active_input is not None or input_activation is not None
        padded_input_activation = np.zeros(self.input_dim + 1, dtype=np.bool_)
        if active_input is not None:
            padded_input_activation[:-1][active_input] = True
        else:
            padded_input_activation[:-1] = input_activation
        return padded_input_activation

    def add_output(self, added_outputs, edges_threshold):
        replaced_output, = np.where(self.output_edges[:].squeeze(1) < edges_threshold)
        replaced_output = replaced_output[:added_outputs]
        self.output_edges[replaced_output] = 0
        self.output_edge[replaced_output] = self.invalid_output_edge
        self.output_permanence[replaced_output] = -1.0
        added_outputs -= len(replaced_output)
        if added_outputs == 0:
            return replaced_output, np.empty(0, dtype=np.int32)

        self.output_edges.add_rows(np.zeros((added_outputs, 1), dtype=self.output_edges.dtype))
        self.output_edge.add_rows(np.full((added_outputs, self.output_edge.size[1]), self.invalid_output_edge, dtype=self.output_edge.dtype))
        self.output_permanence.add_rows(np.full((added_outputs, self.output_permanence.size[1]), -1.0, dtype=self.output_permanence.dtype))
        new_output = np.arange(self.output_dim, self.output_dim + added_outputs)
        self.output_dim += added_outputs
        return replaced_output, new_output

    def update_permanence(self, padded_input_activation, learning_output, active_edge_permanence_change, inactive_edge_permanence_change):
        learning_output_edge = self.output_edge[learning_output]
        learning_target_input, learning_input_edge = self.unpack_output_edge(learning_output_edge)
        learning_permanence = self.output_permanence[learning_output]
        edge_activation = padded_input_activation[learning_target_input]
        learning_permanence_valid = learning_permanence > 0.0
        naive_permanence_change = edge_activation * (active_edge_permanence_change - inactive_edge_permanence_change) + inactive_edge_permanence_change
        updated_permanence = learning_permanence + learning_permanence_valid * naive_permanence_change
        updated_permanence_invalid = updated_permanence <= 0.0
        self.output_edges[learning_output] -= (learning_permanence_valid * updated_permanence_invalid).sum(axis=1, keepdims=True)
        self.output_edge[learning_output] = np.where(updated_permanence_invalid, self.invalid_output_edge, learning_output_edge)
        self.input_edge[learning_target_input, learning_input_edge] = np.where(updated_permanence_invalid, self.invalid_input_edge, self.input_edge[learning_target_input, learning_input_edge])
        self.output_permanence[learning_output] = updated_permanence

    def add_edge(self, padded_input_activation, winner_input, learning_output, permanence_initial, min_active_edges):
        assert permanence_initial > 0.0

        learning_edge = self.output_edge[learning_output]
        learning_edge_target = self.get_output_edge_target(learning_edge)
        output_edge_activation = padded_input_activation[learning_edge_target]
        added_output_edges = np.clip(min_active_edges - output_edge_activation.sum(axis=1), 0, min(min_active_edges, len(winner_input)))
        max_added_output_edges = added_output_edges.max()

        whole_input_to_winner = np.full(self.input_dim + 1, len(winner_input), dtype=np.int32)
        whole_input_to_winner[winner_input] = np.arange(len(winner_input))

        edge_priority = np.random.rand(len(learning_output), len(winner_input) + 1).astype(np.float32)
        np.put_along_axis(edge_priority, whole_input_to_winner[learning_edge_target], np.inf, axis=1)
        edge_priority = edge_priority[:, :-1]
        edge_absent = edge_priority < 1.0
        # edge_prioritized = np.argpartition(edge_priority, max_added_output_edges - 1, axis=1) < np.expand_dims(added_output_edges, 1)
        edge_prioritized = np.zeros((len(learning_output), len(winner_input)), dtype=np.bool_)
        np.put_along_axis(edge_prioritized, np.argpartition(edge_priority, max_added_output_edges - 1, axis=1)[:, :max_added_output_edges], np.expand_dims(np.arange(max_added_output_edges), 0) < np.expand_dims(added_output_edges, 1), axis=1)
        edge_added = edge_absent & edge_prioritized

        # TODO: tmp
        added_output_edges = np.minimum(added_output_edges, edge_added.sum(axis=1))
        # print(*edge_prioritized.astype(int).tolist(), sep='\n')
        # if not np.allclose(added_output_edges, edge_added.sum(axis=1)):
        #     print(added_output_edges)
        #     print(edge_added.sum(axis=1))
        #     quit()
        if (edge_priority[edge_prioritized] > 1.0).any():
            print(edge_priority)
            print(edge_prioritized)
            quit()

        added_input_edge_target = np.tile(learning_output + 1, (len(winner_input), 1))
        replaced_edges, free_index, src_index, residue_edges, residue_index, src_residue_index = replace_free(
            self.input_edge[winner_input] == self.invalid_input_edge, [self.input_edge[:]], [added_input_edge_target],
            dest_index=winner_input, src_valid=edge_added.T, return_indices=True, return_residue_info=True
        )
        max_new_input_edges = residue_edges.max(initial=0)
        if max_new_input_edges > 0:
            prev_max_input_edges = self.input_edge.size[1]
            new_input_edge = (winner_input[residue_index[0]], residue_index[1])
            new_input_edge_target = np.full((1 + self.input_dim, max_new_input_edges), self.invalid_input_edge, dtype=self.input_edge.dtype)
            new_input_edge_target[new_input_edge] = learning_output[src_residue_index[1]]
            self.input_edge.add_cols(new_input_edge_target)

        added_output_edge_target = np.full((len(learning_output), len(winner_input)), self.invalid_output_edge, dtype=self.output_edge.dtype)
        added_output_edge_target[src_index[::-1]] = self.pack_output_edge(*free_index)
        if max_new_input_edges > 0:
            added_output_edge_target[src_residue_index[::-1]] = self.pack_output_edge(new_input_edge[0], new_input_edge[1] + prev_max_input_edges)

        replaced_edges, residue_edges, residue_index, src_residue_index = replace_free(
            learning_edge == self.invalid_output_edge, [self.output_edge[:], self.output_permanence[:]], [added_output_edge_target, permanence_initial],
            dest_index=learning_output, src_valid=edge_added, return_residue_info=True
        )
        max_new_output_edges = residue_edges.max(initial=0)
        if max_new_output_edges > 0:
            new_output_edge = (learning_output[residue_index[0]], residue_index[1])
            new_output_edge_target = np.full((self.output_dim, max_new_output_edges), self.invalid_output_edge, dtype=self.output_edge.dtype)
            new_output_edge_permanence = np.full(new_output_edge_target.shape, -1.0, dtype=self.output_permanence.dtype)
            new_output_edge_target[new_output_edge] = added_output_edge_target[src_residue_index]
            new_output_edge_permanence[new_output_edge] = permanence_initial
            self.output_edge.add_cols(new_output_edge_target)
            self.output_permanence.add_cols(new_output_edge_permanence)
        self.output_edges[learning_output] += np.expand_dims(added_output_edges, 1)
        
        # print('already connected:\n', (~edge_absent).sum(axis=1))
        # print('input -> output:\n', self.process(winner_input))
        # print('output <- input:\n', self.process(winner_input, invoked_output=learning_output))

    def process(self, active_input=None, padded_input_activation=None, invoked_output=None, permanence_threshold=None):
        if invoked_output is None and permanence_threshold is not None:
            raise NotImplementedError()

        if invoked_output is not None:
            if padded_input_activation is None:
                padded_input_activation = self.pad_input_activation(active_input=active_input)
            edge_target = self.get_output_edge_target(self.output_edge[invoked_output])
            edge_weight = self.output_permanence[invoked_output] >= permanence_threshold if permanence_threshold is not None else 1
            projected = (edge_weight & padded_input_activation[edge_target]).sum(axis=1)
            return projected

        edge_target = self.input_edge[active_input]
        projected = bincount(edge_target.flatten(), minLength=1 + self.output_dim)
        # if not np.allclose(projected[1:], self.process(active_input, padded_input_activation, np.arange(self.output_dim))):
        #     print(self.process(active_input, padded_input_activation, np.arange(self.output_dim)) - projected[1:])
        assert len(projected) == 1 + self.output_dim
        return projected[1:]

        # return self.process(active_input, padded_input_activation, np.arange(self.output_dim))

    def update(
        self, learning_output,
        input_activation=None, padded_input_activation=None,
        winner_input=None, min_active_edges=20,
        permanence_initial=0.01, active_edge_permanence_change=0.3, inactive_edge_permanence_change=0.05
    ):
        assert input_activation is not None or padded_input_activation is not None
        if padded_input_activation is None:
            padded_input_activation = self.pad_input_activation(input_activation=input_activation)

        self.update_permanence(padded_input_activation, learning_output, active_edge_permanence_change, inactive_edge_permanence_change)
        if winner_input is not None:
            self.add_edge(padded_input_activation, winner_input, learning_output, permanence_initial, min_active_edges)

class PredictiveProjection:
    class State:
        def __init__(self, prediction, segment_potential, matching_segment, matching_segment_activation):
            self.prediction = prediction
            self.segment_potential = segment_potential
            self.matching_segment = matching_segment
            self.matching_segment_activation = matching_segment_activation
            self.max_jittered_potential = None
            self.matching_segment_jittered_potential = None

    # TODO: group is a terrible name
    segment_group_dtype = np.int32
    group_segment_dtype = np.int32

    def __init__(
        self, output_dim,
        permanence_initial=0.01, permanence_threshold=0.5, permanence_increment=0.3, permanence_decrement=0.05, permanence_punishment=0.01,
        segment_activation_threshold=10, segment_matching_threshold=10, segment_sampling_synapses=20,
        group_segment_growth_exponential=True
    ):
        assert segment_activation_threshold >= segment_matching_threshold

        self.output_dim = output_dim

        self.permanence_initial = permanence_initial
        self.permanence_threshold = permanence_threshold
        self.permanence_increment = permanence_increment
        self.permanence_decrement = permanence_decrement
        self.permanence_punishment = permanence_punishment

        self.segment_activation_threshold = segment_activation_threshold
        self.segment_matching_threshold = segment_matching_threshold
        self.segment_sampling_synapses = segment_sampling_synapses

        self.segment_projection = SparseProjection(self.output_dim)
        self.segment_group = DynamicArray2D(self.segment_group_dtype, size=(0, 1), growth_exponential=(group_segment_growth_exponential, False))
        self.group_segments = np.zeros(self.output_dim, dtype=self.group_segment_dtype)

    def ensure_jittered_potential_info(self, state, matching_segment_group=None):
        if state.max_jittered_potential is not None and state.matching_segment_jittered_potential is not None:
            return
        if matching_segment_group is None:
            matching_segment_group = self.segment_group[state.matching_segment].squeeze(1)
        matching_segment_jittered_potential = state.segment_potential[state.matching_segment].astype(np.float32)
        matching_segment_jittered_potential += np.random.rand(*state.matching_segment.shape)
        max_jittered_potential = np.zeros(self.output_dim, dtype=np.float32)
        np.maximum.at(max_jittered_potential, matching_segment_group, matching_segment_jittered_potential)
        state.max_jittered_potential = max_jittered_potential
        state.matching_segment_jittered_potential = matching_segment_jittered_potential

    def process(self, active_input, return_jittered_potential_info=True):
        segment_potential = self.segment_projection.process(active_input=active_input)
        matching_segment, = np.where(segment_potential >= self.segment_matching_threshold)
        matching_segment_group = self.segment_group[matching_segment].squeeze(1)
        matching_segment_activation = self.segment_projection.process(active_input=active_input, invoked_output=matching_segment, permanence_threshold=self.permanence_threshold)
        prediction = bincount(matching_segment_group, weights=matching_segment_activation >= self.segment_activation_threshold, minLength=self.output_dim)
        state = self.State(prediction, segment_potential, matching_segment, matching_segment_activation)
        if return_jittered_potential_info:
            self.ensure_jittered_potential_info(state, matching_segment_group=matching_segment_group)
        return state

    def update(self, prev_state, input_activation, learning_output, output_punishment, winner_input=None, output_learning=None, epsilon=1e-8):
        if prev_state is None:
            return
        if output_learning is None:
            output_learning = np.zeros(self.output_dim, dtype=np.bool_)
            output_learning[learning_output] = True

        matching_segment_group = self.segment_group[prev_state.matching_segment].squeeze(1)
        self.ensure_jittered_potential_info(prev_state, matching_segment_group=matching_segment_group)
        matching_segment_group_unpredicted = prev_state.prediction[matching_segment_group] < epsilon
        matching_segment_best_matching = np.abs(prev_state.matching_segment_jittered_potential - prev_state.max_jittered_potential[matching_segment_group]) < epsilon
        learning_segment, = np.where(output_learning[matching_segment_group] & ((prev_state.matching_segment_activation > 0) | (matching_segment_group_unpredicted & matching_segment_best_matching)))
        punished_segment, = np.where(output_punishment[matching_segment_group])
        learning_segment = prev_state.matching_segment[learning_segment]
        punished_segment = prev_state.matching_segment[punished_segment]

        unaccounted_output, = np.where(prev_state.max_jittered_potential[learning_output] < epsilon)
        if len(unaccounted_output) > 0:
            unaccounted_output = learning_output[unaccounted_output]
            replaced_segment, new_segment = self.segment_projection.add_output(len(unaccounted_output), self.segment_matching_threshold)
            learning_segment = np.concatenate([learning_segment, replaced_segment, new_segment])
            self.segment_group[replaced_segment] = np.expand_dims(unaccounted_output[:len(replaced_segment)], 1)
            self.segment_group.add_rows(np.expand_dims(unaccounted_output[len(replaced_segment):], 1))
            self.group_segments[replaced_segment] -= 1
            self.group_segments[unaccounted_output] += 1

        padded_input_activation = self.segment_projection.pad_input_activation(input_activation)
        self.segment_projection.update(
            learning_segment, padded_input_activation=padded_input_activation, winner_input=winner_input,
            permanence_initial=self.permanence_initial,
            active_edge_permanence_change=self.permanence_increment, inactive_edge_permanence_change=(-self.permanence_decrement),
            min_active_edges=self.segment_sampling_synapses
        )
        self.segment_projection.update(
            punished_segment, padded_input_activation=padded_input_activation,
            active_edge_permanence_change=(-self.permanence_punishment), inactive_edge_permanence_change=0.0
        )