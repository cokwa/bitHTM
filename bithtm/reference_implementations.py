import numpy as np


class TemporalMemory:
    class State:
        def __init__(self, column_dim, cell_dim):
            self.active_cells = set()
            self.winner_cells = set()
            self.active_segments = set()
            self.matching_segments = set()
            self.segment_num_active_potential_synapses = {}

            self.active_cell = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
            self.winner_cell = (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
            self.cell_activation = np.zeros((column_dim, cell_dim), dtype=np.bool_)
            self.cell_prediction = np.zeros((column_dim, cell_dim), dtype=np.bool_)
            self.active_column_bursting = np.empty(0, dtype=np.bool_)
            self.distal_state = None

    def __init__(self, column_dim, cell_dim):
        self.column_dim = column_dim
        self.cell_dim = cell_dim

        self.permanence_initial = 0.21
        self.permanence_threshold = 0.5
        self.permanence_increment = 0.1
        self.permanence_decrement = 0.1
        self.permanence_punishment = 0.01

        self.segment_activation_threshold = 15
        self.segment_matching_threshold = 15
        self.segment_sampling_synapses = 32

        self.columns = np.arange(self.column_dim)
        self.column_cells = np.arange(self.column_dim * self.cell_dim).reshape(self.column_dim, self.cell_dim)
        self.cell_segments = [set() for _ in range(self.column_dim * self.cell_dim)]
        self.segment_cell = []
        self.segment_synapses = []
        self.synapse_presynaptic_cell = {}
        self.synapse_permanence = {}
        self.next_synapse = 0

        self.last_state = self.get_empty_state()

    def get_empty_state(self):
        return self.State(self.column_dim, self.cell_dim)

    def copy_custom(self, custom_tm):
        assert self.column_dim == custom_tm.column_dim and self.cell_dim == custom_tm.cell_dim

        self.segment_cell = custom_tm.distal_projection.segment_bundle[:].squeeze(1).tolist()
        self.cell_segments = [set() for _ in range(self.column_dim * self.cell_dim)]
        for segment, cell in enumerate(self.segment_cell):
            self.cell_segments[cell].add(segment)

        custom_segment_projection = custom_tm.distal_projection.segment_projection
        self.segment_synapses = []
        self.synapse_presynaptic_cell = {}
        self.synapse_permanence = {}
        self.next_synapse = 0
        for synapses, permanences in zip(custom_segment_projection.output_edge[:], custom_segment_projection.output_permanence[:]):
            self.segment_synapses.append([])
            for synapse, permanence in zip(synapses, permanences):
                if synapse == custom_segment_projection.invalid_output_edge:
                    continue
                presynaptic_cell = custom_segment_projection.get_output_edge_target(synapse)
                self.segment_synapses[-1].append(self.next_synapse)
                self.synapse_presynaptic_cell[self.next_synapse] = presynaptic_cell
                self.synapse_permanence[self.next_synapse] = permanence
                self.next_synapse += 1

        empty_state = self.get_empty_state()
        self.last_state.winner_cells = empty_state.winner_cells
        self.last_state.active_segments = empty_state.active_segments
        self.last_state.matching_segments = empty_state.matching_segments
        self.last_state.segment_num_active_potential_synapses = empty_state.segment_num_active_potential_synapses

        self.last_state.active_cells = custom_tm.flatten_cell(custom_tm.last_state.active_cell).tolist()
        if custom_tm.last_state.winner_cell is not None:
            self.last_state.winner_cells = custom_tm.flatten_cell(custom_tm.last_state.winner_cell).tolist()

        custom_distal_state = custom_tm.last_state.distal_state
        if custom_distal_state is not None:
            matching_segment = custom_distal_state.matching_segment
            self.last_state.active_segments = set(matching_segment[custom_distal_state.matching_segment_active].tolist())
            self.last_state.matching_segments = set(matching_segment.tolist())
            for segment, num_active_potential_synapses in zip(matching_segment, custom_distal_state.segment_potential[matching_segment]):
                self.last_state.segment_num_active_potential_synapses[segment] = num_active_potential_synapses

    def activate_predicted_column(self, column, prev_state, curr_state, learning):
        for segment in self.segments_for_column(column, prev_state.active_segments):
            curr_state.active_cells.add(self.segment_cell[segment])
            curr_state.winner_cells.add(self.segment_cell[segment])

            if learning:
                for synapse in self.segment_synapses[segment]:
                    if self.synapse_presynaptic_cell[synapse] in prev_state.active_cells:
                        self.synapse_permanence[synapse] += self.permanence_increment
                    else:
                        self.synapse_permanence[synapse] -= self.permanence_decrement

                new_synapse_count = self.segment_sampling_synapses - self.num_active_potential_synapses(prev_state, segment)
                self.grow_synapses(segment, new_synapse_count, prev_state)

    def burst_column(self, column, prev_state, curr_state, learning):
        for cell in self.column_cells[column]:
            curr_state.active_cells.add(cell)

        if len(self.segments_for_column(column, prev_state.matching_segments)) > 0:
            learning_segment = self.best_matching_segment(column, prev_state)
            winner_cell = self.segment_cell[learning_segment]
        else:
            winner_cell = self.least_used_cell(column)
            if learning:
                learning_segment = self.grow_new_segment(winner_cell) if len(prev_state.winner_cells) > 0 else None

        curr_state.winner_cells.add(winner_cell)

        if learning and learning_segment is not None:
            for synapse in self.segment_synapses[learning_segment]:
                if self.synapse_presynaptic_cell[synapse] in prev_state.active_cells:
                    self.synapse_permanence[synapse] += self.permanence_increment
                else:
                    self.synapse_permanence[synapse] -= self.permanence_decrement

            new_synapse_count = self.segment_sampling_synapses - self.num_active_potential_synapses(prev_state, learning_segment)
            self.grow_synapses(learning_segment, new_synapse_count, prev_state)

    def punish_predicted_column(self, column, prev_state, learning):
        if learning:
            for segment in self.segments_for_column(column, prev_state.matching_segments):
                for synapse in self.segment_synapses[segment]:
                    if self.synapse_presynaptic_cell[synapse] in prev_state.active_cells:
                        self.synapse_permanence[synapse] -= self.permanence_punishment

    def grow_new_segment(self, cell):
        new_segment = len(self.segment_cell)
        self.cell_segments[cell].add(new_segment)
        self.segment_cell.append(cell)
        self.segment_synapses.append([])
        return new_segment

    def grow_synapses(self, segment, new_synapse_count, prev_state):
        candidates = list(prev_state.winner_cells)
        while len(candidates) > 0 and new_synapse_count > 0:
            presynaptic_cell = np.random.choice(candidates)
            candidates.remove(presynaptic_cell)

            already_connected = False
            for synapse in self.segment_synapses[segment]:
                if self.synapse_presynaptic_cell[synapse] == presynaptic_cell:
                    already_connected = True
                    break

            if not already_connected:
                new_synapse = self.create_new_synapse(segment, presynaptic_cell, self.permanence_initial)
                new_synapse_count -= 1

    def create_new_synapse(self, segment, presynaptic_cell, permanence):
        new_synapse = self.next_synapse
        self.next_synapse += 1
        self.segment_synapses[segment].append(new_synapse)
        self.synapse_presynaptic_cell[new_synapse] = presynaptic_cell
        self.synapse_permanence[new_synapse] = permanence
        return new_synapse

    def least_used_cell(self, column):
        fewest_segments = np.inf
        for cell in self.column_cells[column]:
            fewest_segments = min(fewest_segments, len(self.cell_segments[cell]))

        least_used_cells = []
        for cell in self.column_cells[column]:
            if len(self.cell_segments[cell]) == fewest_segments:
                least_used_cells.append(cell)

        return np.random.choice(least_used_cells)

    def best_matching_segment(self, column, prev_state):
        best_matching_segment = None
        best_score = -1
        for segment in self.segments_for_column(column, prev_state.matching_segments):
            if self.num_active_potential_synapses(prev_state, segment) > best_score:
                best_matching_segment = segment
                best_score = self.num_active_potential_synapses(prev_state, segment)

        return best_matching_segment

    def segments_for_column(self, column, segments):
        owning_segments = set()
        for cell in self.column_cells[column]:
            owning_segments.update(segments.intersection(self.cell_segments[cell]))
        return owning_segments
    
    def num_active_potential_synapses(self, state, segment):
        if segment not in state.segment_num_active_potential_synapses:
            return 0
        return state.segment_num_active_potential_synapses[segment]

    def cleanup_synapses(self, segment):
        synapses = self.segment_synapses[segment]
        synapse = 0
        while synapse < len(synapses):
            if self.synapse_permanence[synapses[synapse]] >= 0:
                synapse += 1
                continue
            del self.synapse_presynaptic_cell[synapses[synapse]]
            del self.synapse_permanence[synapses[synapse]]
            del synapses[synapse]

    def process(self, sp_state, prev_state=None, learning=True):
        if prev_state is None:
            prev_state = self.last_state
        curr_state = self.get_empty_state()

        for column in self.columns:
            if column in sp_state.active_column:
                if len(self.segments_for_column(column, prev_state.active_segments)) > 0:
                    self.activate_predicted_column(column, prev_state, curr_state, learning)
                else:
                    self.burst_column(column, prev_state, curr_state, learning)
            else:
                if len(self.segments_for_column(column, prev_state.matching_segments)) > 0:
                    self.punish_predicted_column(column, prev_state, learning)

        for segment in prev_state.matching_segments:
            self.cleanup_synapses(segment)

        for segment, synapses in enumerate(self.segment_synapses):
            num_active_connected = 0
            num_active_potential = 0
            for synapse in synapses:
                if self.synapse_presynaptic_cell[synapse] in curr_state.active_cells:
                    if self.synapse_permanence[synapse] >= self.permanence_threshold:
                        num_active_connected += 1

                    if self.synapse_permanence[synapse] >= 0:
                        num_active_potential += 1

            if num_active_connected >= self.segment_activation_threshold:
                curr_state.active_segments.add(segment)

            if num_active_potential >= self.segment_matching_threshold:
                curr_state.matching_segments.add(segment)

            curr_state.segment_num_active_potential_synapses[segment] = num_active_potential

        curr_state.active_cell = divmod(np.array(list(curr_state.active_cells)), self.cell_dim)
        curr_state.winner_cell = divmod(np.array(list(curr_state.winner_cells)), self.cell_dim)
        curr_state.cell_activation[curr_state.active_cell] = True
        for segment in curr_state.active_segments:
            curr_state.cell_prediction[divmod(self.segment_cell[segment], self.cell_dim)] = True
        curr_state.active_column_bursting = ~prev_state.cell_prediction[sp_state.active_column].max(axis=1, keepdims=True)

        self.last_state = curr_state
        return curr_state


# legacy: it doesn't work anymore.
class RNGSyncedTemporalMemory(TemporalMemory):
    def __init__(self, column_dim, cell_dim):
        super().__init__(column_dim, cell_dim)

        self.cell_segments_jitter = None
        self.matching_segment_potential_jitter = None
        self.synapse_priority_jitter = None
        self.next_synapse_priority_jitter = 0

    def grow_synapses(self, segment, new_synapse_count, prev_state):
        for candidate in np.argsort(self.synapse_priority_jitter[self.next_synapse_priority_jitter]):
            presynaptic_cell = prev_state.winner_cells[candidate]

            already_connected = False
            for synapse in self.segment_synapses[segment]:
                if self.synapse_presynaptic_cell[synapse] == presynaptic_cell:
                    already_connected = True
                    break

            if not already_connected:
                new_synapse = self.create_new_synapse(segment, presynaptic_cell, self.permanence_initial)
                new_synapse_count -= 1
                if new_synapse_count <= 0:
                    break

        self.next_synapse_priority_jitter += 1

    def least_used_cell(self, column):
        cell_segments_jittered = [len(self.cell_segments[cell]) for cell in self.column_cells[column]] + self.cell_segments_jitter[column]
        return column * self.cell_dim + cell_segments_jittered.argmin()

    def best_matching_segment(self, column, prev_state):
        segments = self.segments_for_column(column, prev_state.matching_segments)
        segment_matching_index = [prev_state.matching_segments.index(segment) for segment in segments]
        matching_segment_potential_jittered = [self.num_active_potential_synapses(prev_state, segment) for segment in segments] + self.matching_segment_potential_jitter[segment_matching_index]
        return segments[matching_segment_potential_jittered.argmax()]

    def process(self, sp_state, prev_state=None, learning=True):
        if prev_state is None:
            prev_state = self.last_state

        self.cell_segments_jitter = np.zeros((self.column_dim, self.cell_dim), dtype=np.float32)
        self.cell_segments_jitter[sp_state.active_column] = np.random.rand(len(sp_state.active_column), self.cell_dim).astype(np.float32)

        if len(prev_state.winner_cells) > 0:
            num_winner_segments = 0
            for active_column in sp_state.active_column:
                num_winner_segments += max(len(self.segments_for_column(active_column, prev_state.active_segments)), 1)
    
            self.synapse_priority_jitter = np.random.rand(num_winner_segments, len(prev_state.winner_cells) + 1).astype(np.float32)
            self.synapse_priority_jitter = self.synapse_priority_jitter[:, :-1]
            self.next_synapse_priority_jitter = 0

        curr_state = super().process(sp_state, prev_state=prev_state, learning=learning)
        self.matching_segment_potential_jitter = np.random.rand(len(curr_state.matching_segments)).astype(np.float32)

        return curr_state