from bithtm import HierarchicalTemporalMemory

import numpy as np


if __name__ == '__main__':
    epochs = 100
    input_patterns = 100
    input_dim = 1000
    input_density = 0.2
    input_noise_probability = 0.05

    column_dim = 2048
    cell_dim = 32

    inputs = np.random.rand(input_patterns, input_dim) < input_density
    htm = HierarchicalTemporalMemory(input_dim, column_dim, cell_dim)

    import time
    start_time = time.time()

    for epoch in range(epochs):
        for input_index, curr_input in enumerate(inputs):
            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.max(axis=1)

            noisy_input = curr_input ^ (np.random.rand(input_dim) < input_noise_probability)
            sp_state, tm_state = htm.process(noisy_input)

            burstings = tm_state.active_column_bursting.sum()
            corrects = prev_column_prediction[sp_state.active_column].sum()
            incorrects = prev_column_prediction.sum() - corrects
            print(f'epoch {epoch:2d}, pattern {input_index:2d}: bursting columns: {burstings:2d}, correct columns: {corrects:2d}, incorrect columns: {incorrects:2d}')

    print(f'{time.time() - start_time} seconds.')