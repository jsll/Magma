import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = plt.cm.jet(mask, cv2.COLORMAP_JET)[:, :, :3]
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def visualize_attention_over_image(
    attention_rollout,
    observation_image,
    observation_tokens,
    action_tokens,
    save_path: str = "attention_output/",
    title: str = "",
) -> tuple[plt.Figure, plt.Figure]:
    """
    Creates two visualization figures: individual attention maps and overlay visualization.

    Args:
        attention_rollout: Attention rollout tensor
        observation_image: Input image to visualize attention over
        observation_tokens: Tokens corresponding to observation features
        action_tokens: Tokens corresponding to actions
        save_path: Path to save visualization
        title: Title for visualization
    Returns:
        Tuple of (individual_maps_figure, overlay_figure)
    """
    title = title.replace(" ", "_")
    batch_size = attention_rollout.shape[0]

    # Create individual attention maps figure
    fig_individual = plt.figure(figsize=(15, 5))
    grid_individual = plt.GridSpec(batch_size, 7, width_ratios=[1, 1, 1, 1, 1, 1, 1.1])
    axs_individual = [fig_individual.add_subplot(grid_individual[i]) for i in range(7)]

    # Process and plot individual attention maps
    images_per_readout = []
    for i, action_token in enumerate(action_tokens):
        mask = attention_rollout[0, action_token, observation_tokens]
        mask = np.asarray(mask.reshape(16, 16))
        mask = mask / np.max(mask)

        # im = axs_individual[i].imshow(mask, cmap="jet")
        # axs_individual[i].axis("off")
        # axs_individual[i].set_title(f"Token {i}")

        mask_resized = cv2.resize(mask, (256, 256))
        images_per_readout.append(mask_resized.copy())

    # plt.colorbar(im, ax=axs_individual[-1], fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.savefig(save_path + title + "_individual.svg")

    # Create overlay visualization figure
    fig_overlay = plt.figure(figsize=(15, 6))
    grid_overlay = plt.GridSpec(batch_size + 1, 3, height_ratios=[0.2, 1], width_ratios=[1, 1, 1.1])

    # Add title in the middle of the top row
    title_ax = fig_overlay.add_subplot(grid_overlay[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.5, title.replace("_", " "), ha="center", va="center", fontsize=12)

    axs_overlay = [fig_overlay.add_subplot(grid_overlay[1, i]) for i in range(3)]
    # Process overlay visualization
    average_readout_image = np.asarray(images_per_readout).mean(0)

    axs_overlay[0].imshow(observation_image, cmap="jet")
    im = axs_overlay[1].imshow(average_readout_image, cmap="jet")
    overlay = show_mask_on_image(observation_image, average_readout_image)
    axs_overlay[2].imshow(overlay, cmap="jet")

    for ax in axs_overlay:
        ax.axis("off")

    axs_overlay[0].set_title("Original Image")
    axs_overlay[1].set_title("Attention Rollout")
    axs_overlay[2].set_title("Overlay")

    plt.colorbar(im, ax=axs_overlay[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path + title + "_overlay.svg")

    return fig_individual, fig_overlay


def visualize_attention_over_prompt(
    individual_token_attentions,
    averaged_token_attention,
    x_ticks,
    save_path: str = "attention_output/",
    title: str = "",
):
    #
    title = title.replace(" ", "_")
    # batch_size = averaged_token_attention.shape[0]

    x_ticks = [tick.replace("\n", "\\n") for tick in x_ticks]

    # Create individual attention maps figure
    # fig_individual = plt.figure(figsize=(15, 5))
    # grid_individual = plt.GridSpec(7, batch_size)
    # axs_individual = [fig_individual.add_subplot(grid_individual[i]) for i in range(7)]
    # fig_individual, axs_individual = plt.subplots(7, batch_size, figsize=(15, 20))
    fig_individual, axs_individual = plt.subplots(7, figsize=(20, 20))

    # Plot individual attention maps
    for i, individual_token_attention in enumerate(individual_token_attentions):
        # Create a bar plot of attention over language tokens using axs_individual[i]
        axs_individual[i].bar(range(individual_token_attention.size), individual_token_attention)
        axs_individual[i].set_xticks(range(len(x_ticks)))
        axs_individual[i].set_xticklabels(x_ticks, rotation=45)
        axs_individual[i].set_title(f"Token {i}")

    fig_individual.suptitle(title.replace("_", " "), fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path + title + "_individual_attention_over_prompt.svg")

    # Create a bar plot of attention over language tokens
    fig_average, ax = plt.subplots()
    ax.bar(
        range(averaged_token_attention.size),
        averaged_token_attention,
    )
    ax.set_xticks(range(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation=45)
    ax.set_title("Average Attention over Language Tokens and the std of the attention")
    fig_average.suptitle(title.replace("_", " "), fontsize=16)
    plt.savefig(save_path + title + "_average_attention_over_language_tokens.svg")

    return fig_individual, fig_average
