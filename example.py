from bithtm import HierarchicalTemporalMemory

import numpy as np


if __name__ == '__main__':
    inputs = np.random.rand(100, 1000) < 0.2
    htm = HierarchicalTemporalMemory(inputs.shape[1], 2048, 32)

    import time
    start_time = time.time()

    for epoch in range(100):
        for i, curr_input in enumerate(inputs):
            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.max(axis=1)

            noisy_input = curr_input ^ (np.random.rand(*curr_input.shape) < 0.05)
            sp_state, tm_state = htm.process(noisy_input)

            burstings = tm_state.active_column_bursting.sum()
            corrects = prev_column_prediction[sp_state.active_column].sum()
            incorrects = prev_column_prediction.sum() - corrects
            print(f'epoch {epoch}, pattern {i}: burstings columns: {burstings}, correct columns: {corrects}, incorrect columns: {incorrects}')

    print(f'{time.time() - start_time} seconds.')