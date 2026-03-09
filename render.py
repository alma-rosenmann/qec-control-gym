"""Rendering for the QEC maintenance environment."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def render_qec(env, step=0, action=None):
    """Draw the lattice, data qubits, ancillas, and error state."""
    d = env.distance
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    red_alpha = 1.0
    blue_alpha = 1.0

    # Draw connections
    for idx, neighbors in enumerate(env.lattice["neighbors"]):
        anc_r, anc_c = env.lattice["ancilla_pos"][idx]
        for d_idx in neighbors:
            for (dr, dc), lookup_idx in env.lattice["data_lookup"].items():
                if lookup_idx == d_idx:
                    ax.plot(
                        [anc_c, dc], [anc_r, dr],
                        color="gray", alpha=0.15, zorder=0
                    )
                    break

    # Draw data qubits: left half = px (red), right half = pz (blue)
    for (r, c), idx in env.lattice["data_lookup"].items():
        px = env.noise_manager.get_px(idx)
        pz = env.noise_manager.get_pz(idx)
        intensity_x = max(0.0, min(1.0, px / 0.05))
        intensity_z = max(0.0, min(1.0, pz / 0.05))
        # Left half (red by px), right half (blue by pz); no edge on wedges to avoid midline
        left_half = mpatches.Wedge(
            (c, r), 0.15, 90, 270,
            facecolor=(1, 1 - intensity_x * 0.5, 1 - intensity_x * 0.5),
            edgecolor="none", zorder=10
        )
        ax.add_patch(left_half)
        right_half = mpatches.Wedge(
            (c, r), 0.15, 270, 90,
            facecolor=(1 - intensity_z * 0.5, 1 - intensity_z * 0.5, 1),
            edgecolor="none", zorder=10
        )
        ax.add_patch(right_half)
        # Single circle outline so we keep the border without a vertical line
        ax.add_patch(
            mpatches.Circle((c, r), 0.15, facecolor="none", edgecolor="black", zorder=10)
        )
        ax.text(c, r, str(idx), ha="center", va="center", fontsize=8, zorder=11)

        if env.qubit_errors[idx, 0]:
            ax.add_patch(
                mpatches.Circle((c - 0.2, r), 0.06, color="red", alpha=red_alpha, zorder=12)
            )
        if env.qubit_errors[idx, 1]:
            ax.add_patch(
                mpatches.Circle((c + 0.2, r), 0.06, color="blue", alpha=blue_alpha, zorder=12)
            )

    # Draw ancillas
    for idx, (anc_r, anc_c) in enumerate(env.lattice["ancilla_pos"]):
        type_ = env.lattice["ancilla_types"][idx]
        is_violated = env.latest_raw_syndrome[idx] == 1
        is_offline = env.ancilla_manager.is_offline(idx)

        if is_offline:
            fill_color = "lightgray"
        else:
            fill_color = "aliceblue" if type_ == "X" else "mistyrose"

        edge_color = "blue" if type_ == "X" else "red"
        lw = 1
        if is_violated:
            edge_color = "orange"
            lw = 3
        fix_highlight = action[1] if (hasattr(action, "__len__") and len(action) > 1) else 0
        if action is not None and fix_highlight == idx + 1:
            edge_color = "gold"
            lw = 3

        rect = mpatches.Rectangle(
            (anc_c - 0.2, anc_r - 0.2), 0.4, 0.4,
            facecolor=fill_color, edgecolor=edge_color, linewidth=lw, zorder=5
        )
        ax.add_patch(rect)
        ax.text(
            anc_c, anc_r, str(idx + env.num_data),
            ha="center", va="center", fontsize=7, color="black", zorder=6
        )

    ax.set_xlim(-0.8, d - 0.2)
    ax.set_ylim(d - 0.2, -0.8)
    ax.set_title(f"Quantum Memory (Dual Protection)\nStep {step}", fontweight="bold")

    handles = [
        mpatches.Patch(edgecolor="orange", linewidth=3, facecolor="none", label="Violated Stab"),
        mpatches.Circle((0, 0), color="red", alpha=1.0, label="Bit Flip Chain (Deadly)"),
        mpatches.Circle((0, 0), color="blue", alpha=1.0, label="Phase Flip Chain (Deadly)"),
    ]
    ax.legend(handles=handles, loc="upper right")
    plt.show()
