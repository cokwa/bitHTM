from bithtm import HierarchicalTemporalMemory

import numpy as np


if __name__ == '__main__':
    inputs = np.random.rand(10, 1000) < 0.2
    htm = HierarchicalTemporalMemory(inputs.shape[1], 2048, 32)

    import time

    start_time = time.time()

    hist_burstings = []
    hist_corrects = []
    hist_incorrects = []

    for epoch in range(100):
        # htm.temporal_memory.last_state = htm.temporal_memory.get_empty_state()

        for i, curr_input in enumerate(inputs):
            print(f'epoch {epoch}, step {i}.')

            prev_column_prediction = htm.temporal_memory.last_state.cell_prediction.max(axis=1)
            
            prev_column_matching = htm.temporal_memory.last_state.distal_state
            if prev_column_matching is not None:
                prev_column_matching = prev_column_matching.max_jittered_potential.reshape(2048, 32).max(axis=1) > 0
            
            prev_tm_state = htm.temporal_memory.last_state

            noisy_input = curr_input ^ (np.random.rand(*curr_input.shape) < 0.05)
            sp_state, tm_state = htm.process(noisy_input)

            print(htm.temporal_memory.distal_projection.segment_projection.input_edge[:].shape, htm.temporal_memory.distal_projection.segment_projection.output_edge[:].shape)
            # print(np.unique(htm.temporal_memory.distal_projection.set_segments, return_counts=True))
            # print(((htm.temporal_memory.distal_projection.segment_projection.output_edge[:] > 0.0).sum(axis=1) < 10).sum())

            burstings = tm_state.active_column_bursting.sum()
            corrects = prev_column_prediction[sp_state.active_column].sum()
            incorrects = prev_column_prediction.sum() - corrects
            print(burstings, corrects, incorrects)

            hist_burstings.append(burstings)
            hist_corrects.append(corrects)
            hist_incorrects.append(incorrects)
            
            if prev_column_matching is not None:
                print((~prev_column_matching[sp_state.active_column]).sum(), prev_column_matching[sp_state.active_column].sum(), prev_column_matching.sum() - prev_column_matching[sp_state.active_column].sum())

            def transform(x, output_resolution=800):
                return tuple(np.round((np.array([x[0] % 256, x[1] + (x[0] // 256) * 32]) + 0.5) * (output_resolution / 256)).astype(int))

            column_activation = np.zeros((2048, 1), dtype=np.bool_)
            column_activation[sp_state.active_column] = True

            # import cv2
            # img = np.zeros((2048, 32, 3), dtype=np.uint8)
            # img.fill(127)
            # img[:] = np.where(np.expand_dims(column_activation, -1), [0, 0, 0], img)
            # img[:] = np.where(np.expand_dims(prev_tm_state.cell_prediction & column_activation, -1), [255, 0, 0], img)
            # img[:] = np.where(np.expand_dims(prev_tm_state.cell_prediction & (~column_activation), -1), [0, 0, 255], img)
            # img[:] = np.where(np.expand_dims(tm_state.cell_prediction, -1), [255, 0, 255], img)
            
            # img2 = np.zeros((256, 256, 3), dtype=np.uint8)
            # for i in range(2048 // 256):
            #     img2[i*32:(i+1)*32] = img[i*256:(i+1)*256].transpose(1, 0, 2)
            # img2 = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)

            # import cv2
            # img = np.zeros((800, 800, 3), dtype=np.uint8)
            # img.fill(127)

            # for column in sp_state.active_column:
            #     cv2.line(img, transform([column, 0]), transform([column, 31]), (0, 0, 0), 3)

            # for postsynaptic_cell, segment, segment_permanence in zip(
            #     htm.temporal_memory.distal_projection.segment_set[:].squeeze(1),
            #     htm.temporal_memory.distal_projection.segment_projection.output_edge[:],
            #     htm.temporal_memory.distal_projection.segment_projection.output_permanence[:]
            # ):
            #     end = transform(divmod(postsynaptic_cell, 32))
            #     for presynaptic_cell, permanence in zip(segment, segment_permanence):
            #         if permanence <= 0.0:
            #             continue
            #         if presynaptic_cell == htm.temporal_memory.distal_projection.segment_projection.invalid_output_edge:
            #             continue
            #         presynaptic_cell = presynaptic_cell % (2048 * 32)
            #         if not tm_state.cell_activation[divmod(presynaptic_cell, 32)]:
            #             continue
            #         start = transform(divmod(presynaptic_cell, 32))
            #         cv2.line(img, start, end, (0, 0, 0) if permanence < 0.5 else (255, 255, 255))

            # for column, cell in np.array(np.where(prev_tm_state.cell_prediction & column_activation)).T:
            #     cv2.circle(img, transform([column, cell]), 3, (255, 0, 0), -1)
            # for column, cell in np.array(np.where(prev_tm_state.cell_prediction & (~column_activation))).T:
            #     cv2.circle(img, transform([column, cell]), 3, (0, 0, 255), -1)
            # for column, cell in np.array(np.where(tm_state.cell_prediction)).T:
            #     cv2.circle(img, transform([column, cell]), 3, (255, 0, 255), -1)

            # cv2.imshow('htm', img)
            # if cv2.waitKey(0) == ord('q'):
            #     quit()

    print(f'{time.time() - start_time}s')

    from matplotlib import pyplot as plt
    # from bithtm.projections import SparseProjection

    plt.figure()
    plt.plot(hist_burstings)
    plt.plot(hist_corrects)
    plt.plot(hist_incorrects)
    
    window_size = 13
    kernel = np.ones(window_size) / window_size
    plt.figure()
    plt.plot(np.convolve(hist_burstings, kernel, mode='valid'))
    plt.plot(np.convolve(hist_corrects, kernel, mode='valid'))
    plt.plot(np.convolve(hist_incorrects, kernel, mode='valid'))

    plt.figure()
    input_edges = (htm.temporal_memory.distal_projection.segment_projection.input_edge[:] != htm.temporal_memory.distal_projection.segment_projection.invalid_input_edge).sum(axis=1)
    total_input_edges = input_edges.sum()
    input_edges = input_edges[input_edges > 0]
    plt.hist(input_edges, np.logspace(np.log10(input_edges.min()), np.log10(input_edges.max()), 50))
    plt.gca().set_xscale('log')

    plt.figure()
    output_edges = (htm.temporal_memory.distal_projection.segment_projection.output_edge[:] != htm.temporal_memory.distal_projection.segment_projection.invalid_output_edge).sum(axis=1)
    total_output_edges = output_edges.sum()
    output_edges = output_edges[output_edges >= 10]
    plt.hist(output_edges, np.logspace(np.log10(output_edges.min()), np.log10(output_edges.max()), 50))
    plt.gca().set_xscale('log')
    
    print(total_input_edges, total_output_edges)

    plt.show()