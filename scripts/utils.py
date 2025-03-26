import torch
def average_attention_heads(
    attention_matrix: torch.Tensor, head_fusion: str = "mean", dim: int = 2
) -> torch.Tensor:
    """
    Combine attention heads into a single matrix

    Args:
        attention_matrix: Attention matrix with shape [batch, num_heads, seq_len, seq_len]
        head_fusion: Method to combine attention heads ('mean', 'max', or 'min')

    Returns:
        Combined attention matrix
    """
    if head_fusion == "mean":
        return attention_matrix.mean(dim=dim)
    elif head_fusion == "max":
        return attention_matrix.max(dim=dim).values
    elif head_fusion == "min":
        return attention_matrix.min(dim=dim).values
    else:
        raise ValueError(f"Invalid head fusion method: {head_fusion}. Must be one of ['mean', 'max', 'min']")



def attention_rollout(attention_matrices) -> torch.Tensor:
    device = attention_matrices.device
    I = torch.eye(attention_matrices.shape[-1]).unsqueeze(0).to(device)

    A = 0.5 * (attention_matrices[:, 0] + I).to(device)
    rollout = A / A.sum(axis=-1, keepdims=True)
    for attention_matrix in range(1, attention_matrices.shape[1]):
        attention_matrix = attention_matrices[:, attention_matrix]
        A = 0.5 * (attention_matrix + I)
        rollout = rollout @ (A / A.sum(axis=-1, keepdims=True))

    return rollout

def combine_attention_matrices(attention_outputs):
    # Get dimensions
    num_tokens = len(attention_outputs)
    num_layers = len(attention_outputs[0])
    batch_size, num_heads = attention_outputs[0][0].shape[:2]
    # batch_size, num_heads = attention_outputs[0][0].shape[:2]
    initial_seq_len = attention_outputs[0][0].shape[-1]
    final_seq_len = initial_seq_len + num_tokens - 1  # -1 because first output is full matrix

    # Initialize the combined attention matrix
    combined_attention = torch.zeros((num_layers, batch_size, num_heads, final_seq_len, final_seq_len))

    # Copy the first full attention matrix
    combined_attention[:, :, :, :initial_seq_len, :initial_seq_len] = attention_outputs[0][0]

    # Add subsequent token attentions
    for i in range(num_tokens):
        for j in range(num_layers):
            if i == 0:
                combined_attention[j, :, :, :initial_seq_len, :initial_seq_len] = attention_outputs[i][j]
            else:
                # Current position in the sequence

                current_attention = attention_outputs[i][j][:, :, 0, :]
                current_pos = current_attention.shape[-1]
                # print("current_attention.shape: ", current_attention.shape)
                # print("combined_attention.shape: ", combined_attention[j, :, :, current_pos - 1, :current_pos].shape)
                # print("current_pos: ", current_pos - 1)
                combined_attention[j, :, :, current_pos - 1, :current_pos] = current_attention
                # Place it in the correct position in the combined matrix
                # combined_attention[:, :, current_pos, : current_pos + 1] = current_attention[:, :, 0, :]
    combined_attention = combined_attention.permute(1, 0, 2, 3, 4)

    return combined_attention

