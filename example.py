from bithtm import HierarchicalTemporalMemory
from bithtm.reference_implementations import TemporalMemory as ReferenceTemporalMemory

import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--reference', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    epochs = 100
    input_patterns = 10
    input_dim = 1000
    input_density = 0.2
    input_noise_probability = 0.05

    column_dim = 2048
    cell_dim = 32
    use_reference_implementation = args.reference

    np.random.seed(3407)
    inputs = np.random.rand(input_patterns, input_dim) < input_density
    noisy_inputs = inputs ^ (np.random.rand(epochs, input_patterns, input_dim) < input_noise_probability)
    htm = HierarchicalTemporalMemory(input_dim, column_dim, cell_dim)

    if use_reference_implementation:
        htm.temporal_memory = ReferenceTemporalMemory(column_dim, cell_dim)

    import time
    start_time = time.time()

    epoch_string_length = int(np.ceil(np.log10(epochs - 1)))
    pattern_string_length = int(np.ceil(np.log10(input_patterns - 1)))
    column_string_length = int(np.ceil(np.log10(column_dim - 1)))
    active_column_string_length = int(np.ceil(np.log10(htm.spatial_pooler.active_columns - 1)))

    for epoch in range(epochs):
        # for input_index, curr_input in enumerate(inputs):
        for input_index, noisy_input in enumerate(noisy_inputs[epoch]):
            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.max(axis=1)

            # noisy_input = curr_input ^ (np.random.rand(input_dim) < input_noise_probability)
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