def best_ctc_alignment(tf_gloss_logits, input_lengths):
    vocab_size = tf_gloss_logits.shape[-1]
    T = k2.topo(vocab_size)
    supervision_segments = torch.zeros((len(input_lengths), 3))
    supervision_segments[:,0] = torch.arange(len(input_lengths))
    supervision_segments[:,2] = input_lengths
    nnet_out_dense = k2.DenseFsaVec(tf_gloss_logits, supervision_segments)
    lattice = k2.intersect_dense_pruned(
        T, nnet_out_dense,
        frame_idx_name='frame_idx',
        search_beam=3,
        output_beam=1
    )
    best_path = k2.shortest_path(lattice)

    return best_path['frame_idx']

def get_trellis(emission, tokens):
    num_frame = emission.size(0)
    num_tokens = len(tokens)
    blank_id = num_tokens - 1

    # Trellis has extra diemsions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis