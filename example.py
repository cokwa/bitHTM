from bithtm import HierarchicalTemporalMemory
from bithtm.reference_implementations import TemporalMemory as ReferenceTemporalMemory

from bithtm.networks import SpatialPooler
from bithtm.projections import LocalProjection
from bithtm.regularizations import LocalInhibition

import numpy as np


class LocallyConnectedSpatialPooler(SpatialPooler):
    def __init__(self, input_dim, column_dim, active_columns, window_size):
        super().__init__(
            np.prod(input_dim), np.prod(column_dim), active_columns,
            proximal_projection=LocalProjection(input_dim, column_dim, window_size),
            inhibition=LocalInhibition(input_dim, column_dim, window_size, active_outputs_per_neighborhood=2)
        )


class LocallyDrivedHierarchicalTemporalMemory(HierarchicalTemporalMemory):
    def __init__(self, input_dim, column_dim, cell_dim, window_size, active_columns=None):
        if active_columns is None:
            active_columns = round(np.prod(column_dim) * 0.02)
        super().__init__(
            np.prod(input_dim), np.prod(column_dim), cell_dim, active_columns=active_columns,
            spatial_pooler=LocallyConnectedSpatialPooler(input_dim, column_dim, active_columns, window_size)
        )


class ReferenceHierarchicalTemporalMemory(HierarchicalTemporalMemory):
    def __init__(self, input_dim, column_dim, cell_dim, active_columns=None):
        super().__init__(
            input_dim, column_dim, cell_dim, active_columns=active_columns,
            temporal_memory=ReferenceTemporalMemory(column_dim, cell_dim)
        )


if __name__ == '__main__':
    import argparse
    import time


    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--input_patterns', type=int, default=100)
    parser.add_argument('--input_dim', type=int, default=1000)
    parser.add_argument('--input_density', type=float, default=0.2)
    parser.add_argument('--input_noise_probability', type=float, default=0.05)

    parser.add_argument('--column_dim', type=int, default=2048)
    parser.add_argument('--cell_dim', type=int, default=32)
    parser.add_argument('--use_reference_implementation', action='store_true')

    args = parser.parse_args()

    inputs = np.random.rand(args.input_patterns, args.input_dim) < args.input_density

    if args.use_reference_implementation:
        htm = ReferenceHierarchicalTemporalMemory(args.input_dim, args.column_dim, args.cell_dim)
    else:
        # htm = HierarchicalTemporalMemory(args.input_dim, args.column_dim, args.cell_dim)
        htm = LocallyDrivedHierarchicalTemporalMemory((20, 50), (32, 64), args.cell_dim, (8, 20))

    epoch_string_length = int(np.ceil(np.log10(args.epochs - 1)))
    pattern_string_length = int(np.ceil(np.log10(args.input_patterns - 1)))
    column_string_length = int(np.ceil(np.log10(args.column_dim - 1)))
    active_column_string_length = int(np.ceil(np.log10(htm.spatial_pooler.active_columns - 1)))

    start_time = time.time()

    for epoch in range(args.epochs):
        for input_index, curr_input in enumerate(inputs):
            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.max(axis=1)

            noisy_input = curr_input ^ (np.random.rand(args.input_dim) < args.input_noise_probability)
            sp_state, tm_state = htm.process(noisy_input)

            burstings = tm_state.active_column_bursting.sum()
            corrects = prev_column_prediction[sp_state.active_column].sum()
            incorrects = prev_column_prediction.sum() - corrects

            print(
                f'epoch {epoch:{epoch_string_length}d}, '
                f'pattern {input_index:{pattern_string_length}d}: '
                f'bursting columns: {burstings:{active_column_string_length}d}, '
                f'correct columns: {corrects:{active_column_string_length}d}, '
                f'incorrect columns: {incorrects:{column_string_length}d}'
            )

    print(f'{time.time() - start_time} seconds.')