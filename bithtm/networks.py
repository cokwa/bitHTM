from .projections import DenseProjection, PredictiveProjection
from .regularizations import ExponentialBoosting, GlobalInhibition

import numpy as np


class SpatialPooler:
    class State:
        def __init__(self, active_column, overlaps=None, boosted_overlaps=None):
            self.active_column = active_column
            self.overlaps = overlaps
            self.boosted_overlaps = boosted_overlaps

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

        return self.State(active_column, overlaps=overlaps, boosted_overlaps=boosted_overlaps)


class TemporalMemory:
    class State:
        def __init__(self, active_cell, winner_cell=None, cell_activation=None, cell_prediction=None, active_column_bursting=None, distal_state=None):
            self.active_cell = active_cell
            self.winner_cell = winner_cell
            self.cell_activation = cell_activation
            self.cell_prediction = cell_prediction
            self.active_column_bursting = active_column_bursting
            self.distal_state = distal_state

    def __init__(
        self, column_dim, cell_dim,
        distal_projection=None
    ):
        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.distal_projection = distal_projection or PredictiveProjection(self.column_dim * self.cell_dim)

        self.last_state = self.get_empty_state()

    def get_empty_state(self):
        return self.State(
            (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)),
            cell_activation=np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_),
            cell_prediction=np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_),
            active_column_bursting=np.empty(0, dtype=np.bool_)
        )

    def flatten_cell(self, cell):
        if cell is None:
            return None
        assert len(cell) == 2 and len(cell[0].shape) == 1
        return cell[0] * self.cell_dim + cell[1]

    def evaluate_cell_best_matching(self, predictive_projection, prev_projection_state, relevant_column, epsilon=1e-8):
        if prev_projection_state is None:
            return np.zeros((len(relevant_column), 1), dtype=np.bool_), np.zeros((len(relevant_column), self.cell_dim), dtype=np.bool_)
        predictive_projection.fill_jittered_potential_info(prev_projection_state)
        cell_max_jittered_potential = prev_projection_state.max_jittered_potential.reshape(self.column_dim, self.cell_dim)
        cell_max_jittered_potential = cell_max_jittered_potential[relevant_column]
        column_max_jittered_potential = cell_max_jittered_potential.max(axis=1, keepdims=True)
        column_matching = column_max_jittered_potential >= predictive_projection.segment_matching_threshold
        cell_best_matching = np.abs(cell_max_jittered_potential - column_max_jittered_potential) < epsilon
        return column_matching, cell_best_matching

    def evaluate_cell_least_used(self, predictive_projection, relevant_column, epsilon=1e-8):
        cell_segments = predictive_projection.bundle_segments.reshape(self.column_dim, self.cell_dim)
        cell_segments_jittered = cell_segments[relevant_column].astype(np.float32)
        cell_segments_jittered += np.random.rand(*cell_segments_jittered.shape)
        cell_least_used = np.abs(cell_segments_jittered - cell_segments_jittered.min(axis=1, keepdims=True)) < epsilon
        return cell_least_used

    def process(self, sp_state, prev_state=None, learning=True, return_winner_cell=True, epsilon=1e-8):
        # valid_output_edge = np.where(self.distal_projection.segment_projection.output_edge != self.distal_projection.segment_projection.invalid_output_edge)
        # print(f'synapse tgt: {self.distal_projection.segment_projection.output_edge[valid_output_edge].squeeze(0).tolist()}')
        # print(f'synapse prm: {self.distal_projection.segment_projection.output_permanence[valid_output_edge].squeeze(0).tolist()}')

        if prev_state is None:
            prev_state = self.last_state

        active_column = sp_state.active_column
        active_column_cell_prediction = prev_state.cell_prediction[active_column]
        active_column_bursting = ~active_column_cell_prediction.max(axis=1, keepdims=True)

        if learning or return_winner_cell:
            column_matching, cell_best_matching = self.evaluate_cell_best_matching(self.distal_projection, prev_state.distal_state, active_column, epsilon=epsilon)
            least_used_cell = self.evaluate_cell_least_used(self.distal_projection, active_column, epsilon=epsilon)
            active_column_cell_winner = active_column_cell_prediction | (active_column_bursting & np.where(column_matching, cell_best_matching, least_used_cell))
            winner_cell = np.where(active_column_cell_winner)
            winner_cell = (active_column[winner_cell[0]], winner_cell[1])
            
            # TODO: TMP
            argsort = np.argsort(self.flatten_cell(winner_cell))
            winner_cell = (winner_cell[0][argsort], winner_cell[1][argsort])

        if learning:
            column_punishment = np.ones(self.column_dim, dtype=np.bool_)
            column_punishment[active_column] = False
            self.distal_projection.update(
                prev_state.distal_state,
                prev_state.cell_activation.flatten(), self.flatten_cell(winner_cell), np.repeat(column_punishment, self.cell_dim),
                winner_input=self.flatten_cell(prev_state.winner_cell), epsilon=epsilon
            )

        active_column_cell_activation = active_column_cell_prediction | active_column_bursting
        active_cell = np.where(active_column_cell_activation)
        active_cell = (active_column[active_cell[0]], active_cell[1])
        cell_activation = np.zeros((self.column_dim, self.cell_dim), dtype=np.bool_)
        cell_activation[active_column] = active_column_cell_activation

        # TODO: TMP
        argsort = np.argsort(self.flatten_cell(active_cell))
        active_cell = (active_cell[0][argsort], active_cell[1][argsort])

        distal_state = self.distal_projection.process(self.flatten_cell(active_cell), return_jittered_potential_info=return_winner_cell)
        cell_prediction = distal_state.prediction.reshape(self.column_dim, self.cell_dim) > epsilon

        curr_state = self.State(active_cell, cell_activation=cell_activation, cell_prediction=cell_prediction, active_column_bursting=active_column_bursting, distal_state=distal_state)
        if learning or return_winner_cell:
            curr_state.winner_cell = winner_cell
        self.last_state = curr_state
        return curr_state


class HierarchicalTemporalMemory:
    def __init__(
        self, input_dim, column_dim, cell_dim, active_columns=None,
        spatial_pooler=None, temporal_memory=None
    ):
        if active_columns is None:
            active_columns = round(column_dim * 0.02)

        self.column_dim = column_dim
        self.cell_dim = cell_dim
        self.active_columns = active_columns

        self.spatial_pooler = spatial_pooler or SpatialPooler(input_dim, column_dim, active_columns)
        self.temporal_memory = temporal_memory or TemporalMemory(column_dim, cell_dim)
        # from .reference_implementations import TemporalMemory as ReferenceTemporalMemory
        # self.temporal_memory = temporal_memory or ReferenceTemporalMemory(column_dim, cell_dim)

    def process(self, input, learning=True):
        sp_state = self.spatial_pooler.process(input, learning=learning)
        tm_state = self.temporal_memory.process(sp_state, learning=learning)
        return sp_state, tm_state