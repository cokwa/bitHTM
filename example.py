from bithtm import HierarchicalTemporalMemory

import numpy as np


if __name__ == '__main__':
    inputs = np.random.rand(10, 1000) < 0.2
    htm = HierarchicalTemporalMemory(inputs.shape[1], 2048, 32)

    import time

    start_time = time.time()

    for epoch in range(100):
        for curr_input in inputs:
            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.max(axis=1)
            
            noisy_input = curr_input ^ (np.random.rand(*curr_input.shape) < 0.05)
            sp_state, tm_state = htm.process(noisy_input)

            print(htm.temporal_memory.distal_projection.segment_projection.projection[:].shape, htm.temporal_memory.distal_projection.segment_projection.backprojection[:].shape)
            # print(np.unique(htm.temporal_memory.distal_projection.group_segments, return_counts=True))
            print(((htm.temporal_memory.distal_projection.segment_projection.backprojection_permanence[:] > 0.0).sum(axis=1) < 10).sum())

            burstings = tm_state.active_column_bursting.sum()
            corrects = prev_column_prediction[sp_state.active_column].sum()
            incorrects = prev_column_prediction.sum() - corrects
            print(burstings, corrects, incorrects)

    print(f'{time.time() - start_time}s')