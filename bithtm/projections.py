from .utils import DynamicArray2D, bincount, replace_free

import numpy as np


class DenseProjection:
    def __init__(
        self, input_dim, output_dim,
        permanence_mean=0.0, permanence_std=0.1,
        permanence_threshold=0.0, permanence_increment=0.1, permanence_decrement=0.1
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


class SparseProjection:
    def __init__(
        self, input_dim, output_dim=0,
        output_growth_exponential=True, edge_growth_exponential=True
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.invalid_input_edge = 0
        self.invalid_output_edge = self.input_dim

        # TODO: optimize by separately tracking completely empty space of edges per input/output. -> use for adding edges and projection. (max cut-off)

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
        return output_edge % (self.input_dim + 1)

    def pack_output_edge(self, target_input, input_edge):
        return input_edge * (self.input_dim + 1) + target_input

    def unpack_output_edge(self, output_edge):
        input_edge, target_input = np.divmod(output_edge, self.input_dim + 1)
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
        self.input_edge[self.unpack_output_edge(self.output_edge[replaced_output])] = self.invalid_input_edge
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
        learning_output_edge_valid = learning_output_edge != self.invalid_output_edge
        learning_target_input, learning_input_edge = self.unpack_output_edge(learning_output_edge)
        edge_activation = padded_input_activation[learning_target_input]
        naive_permanence_change = edge_activation * (active_edge_permanence_change - inactive_edge_permanence_change) + inactive_edge_permanence_change
        updated_permanence = self.output_permanence[learning_output] + learning_output_edge_valid * naive_permanence_change
        updated_permanence_invalid = updated_permanence <= 0.0
        self.output_edges[learning_output] -= (learning_output_edge_valid & updated_permanence_invalid).sum(axis=1, keepdims=True)
        self.output_edge[learning_output] = np.where(updated_permanence_invalid, self.invalid_output_edge, learning_output_edge)
        self.input_edge[learning_target_input, learning_input_edge] = np.where(updated_permanence_invalid, self.invalid_input_edge, np.expand_dims(1 + learning_output, 1))
        self.output_permanence[learning_output] = updated_permanence

        # np.subtract.at(self.output_edges[:], learning_output, (learning_output_edge_valid & updated_permanence_invalid).sum(axis=1, keepdims=True))
        # updated_permanence_invalid = np.where(updated_permanence_invalid)
        # np.maximum.at(self.output_edge[:], (learning_output[updated_permanence_invalid[0]], updated_permanence_invalid[1]), self.invalid_output_edge)
        # np.minimum.at(self.input_edge[:], (learning_target_input[updated_permanence_invalid], learning_input_edge[updated_permanence_invalid]), self.invalid_input_edge)

        # for output in learning_output:
        #     for edge_index, edge in enumerate(self.output_edge[output]):
        #         if edge == self.invalid_output_edge:
        #             continue
        #         target_input, input_edge = self.unpack_output_edge(edge)
        #         if padded_input_activation[target_input]:
        #             self.output_permanence[output, edge_index] += active_edge_permanence_change
        #         else:
        #             self.output_permanence[output, edge_index] += inactive_edge_permanence_change
        #         if self.output_permanence[output, edge_index] <= 0.0:
        #             self.output_edges[output] -= 1
        #             self.output_edge[output, edge_index] = self.invalid_output_edge
        #             self.input_edge[target_input, input_edge] = self.invalid_input_edge

    def add_edge(self, padded_input_activation, winner_input, learning_output, permanence_initial, min_active_edges):
        assert permanence_initial > 0.0

        learning_edge = self.output_edge[learning_output]
        learning_edge_target = self.get_output_edge_target(learning_edge)
        output_active_edges = padded_input_activation[learning_edge_target].sum(axis=1)
        added_output_edges = np.clip(min_active_edges - output_active_edges, 0, min(min_active_edges, len(winner_input)))
        
        whole_input_to_winner = np.full(self.input_dim + 1, len(winner_input), dtype=np.int32)
        whole_input_to_winner[winner_input] = np.arange(len(winner_input))

        edge_priority = np.random.rand(len(learning_output), len(winner_input) + 1).astype(np.float32)
        np.put_along_axis(edge_priority, whole_input_to_winner[learning_edge_target], np.inf, axis=1)
        edge_priority = edge_priority[:, :-1]
        edge_absent = edge_priority < 1.0
        edge_prioritized = np.zeros(edge_priority.shape, dtype=np.bool_)
        np.put_along_axis(edge_prioritized, np.argsort(edge_priority, axis=1), np.expand_dims(np.arange(len(winner_input)), 0) < np.expand_dims(added_output_edges, 1), axis=1)
        edge_added = edge_absent & edge_prioritized
        added_output_edges = edge_added.sum(axis=1)

        added_input_edge_target = np.tile(1 + learning_output, (len(winner_input), 1))
        replaced_edges, free_index, src_index, residue_edges, residue_index, src_residue_index = replace_free(
            self.input_edge[winner_input] == self.invalid_input_edge, [self.input_edge[:]], [added_input_edge_target],
            dest_index=winner_input, src_valid=edge_added.T, return_indices=True, return_residue_info=True
        )
        max_new_input_edges = residue_edges.max(initial=0)
        if max_new_input_edges > 0:
            prev_max_input_edges = self.input_edge.size[1]
            new_input_edge = (winner_input[residue_index[0]], residue_index[1])
            new_input_edge_target = np.full((self.input_dim + 1, max_new_input_edges), self.invalid_input_edge, dtype=self.input_edge.dtype)
            new_input_edge_target[new_input_edge] = 1 + learning_output[src_residue_index[1]]
            self.input_edge.add_cols(new_input_edge_target)
            new_input_edge = (new_input_edge[0], prev_max_input_edges + new_input_edge[1])

        added_output_edge_target = np.full((len(learning_output), len(winner_input)), self.invalid_output_edge, dtype=self.output_edge.dtype)
        added_output_edge_target[src_index[::-1]] = self.pack_output_edge(*free_index)
        if max_new_input_edges > 0:
            added_output_edge_target[src_residue_index[::-1]] = self.pack_output_edge(*new_input_edge)

        replaced_edges, residue_edges, residue_index, src_residue_index = replace_free(
            learning_edge == self.invalid_output_edge, [self.output_edge[:], self.output_permanence[:]], [added_output_edge_target, permanence_initial],
            dest_index=learning_output, src_valid=edge_added, src_lengths=added_output_edges, return_residue_info=True
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
        assert len(projected) == 1 + self.output_dim
        return projected[1:]

    def update(
        self, learning_output,
        input_activation=None, padded_input_activation=None,
        winner_input=None, min_active_edges=32,
        permanence_initial=0.21, active_edge_permanence_change=0.1, inactive_edge_permanence_change=0.1
    ):
        assert input_activation is not None or padded_input_activation is not None
        if padded_input_activation is None:
            padded_input_activation = self.pad_input_activation(input_activation=input_activation)

        self.update_permanence(padded_input_activation, learning_output, active_edge_permanence_change, inactive_edge_permanence_change)
        if winner_input is not None:
            self.add_edge(padded_input_activation, winner_input, learning_output, permanence_initial, min_active_edges)

class PredictiveProjection:
    class State:
        def __init__(self, prediction, segment_potential, matching_segment, matching_segment_activation, matching_segment_active):
            self.prediction = prediction
            self.segment_potential = segment_potential
            self.matching_segment = matching_segment
            self.matching_segment_activation = matching_segment_activation
            self.matching_segment_active = matching_segment_active
            self.max_jittered_potential = None
            self.matching_segment_jittered_potential = None

    def __init__(
        self, output_dim,
        permanence_initial=0.21, permanence_threshold=0.5, permanence_increment=0.1, permanence_decrement=0.1, permanence_punishment=0.01,
        segment_activation_threshold=15, segment_matching_threshold=15, segment_sampling_synapses=32,
        segment_bundle_growth_exponential=True
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
        self.segment_bundle = DynamicArray2D(np.int32, size=(0, 1), growth_exponential=(segment_bundle_growth_exponential, False))
        self.bundle_segments = np.zeros(self.output_dim, dtype=np.int32)

    def ensure_jittered_potential_info(self, state, matching_segment_bundle=None):
        if state.max_jittered_potential is not None and state.matching_segment_jittered_potential is not None:
            return
        if matching_segment_bundle is None:
            matching_segment_bundle = self.segment_bundle[state.matching_segment].squeeze(1)
        matching_segment_jittered_potential = state.segment_potential[state.matching_segment].astype(np.float32)
        matching_segment_jittered_potential += np.random.rand(*state.matching_segment.shape)
        max_jittered_potential = np.zeros(self.output_dim, dtype=np.float32)
        np.maximum.at(max_jittered_potential, matching_segment_bundle, matching_segment_jittered_potential)
        state.max_jittered_potential = max_jittered_potential
        state.matching_segment_jittered_potential = matching_segment_jittered_potential

    def process(self, active_input, return_jittered_potential_info=True):
        segment_potential = self.segment_projection.process(active_input=active_input)
        matching_segment, = np.where(segment_potential >= self.segment_matching_threshold)
        matching_segment_bundle = self.segment_bundle[matching_segment].squeeze(1)
        matching_segment_activation = self.segment_projection.process(active_input=active_input, invoked_output=matching_segment, permanence_threshold=self.permanence_threshold)
        matching_segment_active = matching_segment_activation >= self.segment_activation_threshold
        prediction = bincount(matching_segment_bundle, weights=matching_segment_active, minLength=self.output_dim)
        state = self.State(prediction, segment_potential, matching_segment, matching_segment_activation, matching_segment_active)
        if return_jittered_potential_info:
            self.ensure_jittered_potential_info(state, matching_segment_bundle=matching_segment_bundle)
        return state

    def update(self, prev_state, input_activation, learning_output, output_punishment, winner_input=None, output_learning=None, epsilon=1e-8):
        if prev_state is None:
            return
        if output_learning is None:
            output_learning = np.zeros(self.output_dim, dtype=np.bool_)
            output_learning[learning_output] = True

        matching_segment_bundle = self.segment_bundle[prev_state.matching_segment].squeeze(1)
        self.ensure_jittered_potential_info(prev_state, matching_segment_bundle=matching_segment_bundle)
        matching_segment_bundle_unpredicted = prev_state.prediction[matching_segment_bundle] < epsilon
        matching_segment_best_matching = np.abs(prev_state.matching_segment_jittered_potential - prev_state.max_jittered_potential[matching_segment_bundle]) < epsilon
        learning_segment = prev_state.matching_segment[output_learning[matching_segment_bundle] & (prev_state.matching_segment_active | (matching_segment_bundle_unpredicted & matching_segment_best_matching))]
        punished_segment = prev_state.matching_segment[output_punishment[matching_segment_bundle]]

        unaccounted_output, = np.where(prev_state.max_jittered_potential[learning_output] < epsilon)
        if len(unaccounted_output) > 0:
            unaccounted_output = learning_output[unaccounted_output]
            replaced_segment, new_segment = self.segment_projection.add_output(len(unaccounted_output), self.segment_matching_threshold)
            learning_segment = np.concatenate([learning_segment, replaced_segment, new_segment])
            replaced_segment_bundle, bundle_replaced_segments = np.unique(self.segment_bundle[replaced_segment], return_counts=True)
            self.bundle_segments[replaced_segment_bundle] -= bundle_replaced_segments
            self.bundle_segments[unaccounted_output] += 1
            self.segment_bundle[replaced_segment] = np.expand_dims(unaccounted_output[:len(replaced_segment)], 1)
            if len(new_segment) > 0:
                self.segment_bundle.add_rows(np.expand_dims(unaccounted_output[-len(new_segment):], 1))

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