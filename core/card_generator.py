import io
import textwrap
import matplotlib.pyplot as plt
import numpy as np

def render_photographer_card_image(card_data: dict, radar_scores: dict) -> bytes:
    """
    Renders a high-resolution (1280x1760) Photographer Pass card
    with dark cyber-styling, embedded radar chart, and typography.
    Returns the image as PNG bytes.
    """
    fig = plt.figure(figsize=(8, 11), facecolor="#0B0E14")
    
    # Base canvas
    ax_base = fig.add_axes([0, 0, 1, 1])
    ax_base.set_facecolor("#0B0E14")
    ax_base.axis("off")

    # Card border with accent colors
    outer_box = plt.Rectangle((0.04, 0.03), 0.92, 0.94, fill=True, 
                              facecolor="#12161F", edgecolor="#00CC96", 
                              linewidth=2.2, transform=ax_base.transAxes, zorder=1)
    ax_base.add_patch(outer_box)
    
    inner_line = plt.Rectangle((0.048, 0.038), 0.904, 0.924, fill=False,
                               edgecolor="#00CC96", alpha=0.25,
                               linewidth=0.8, transform=ax_base.transAxes, zorder=2)
    ax_base.add_patch(inner_line)

    # Header row
    ax_base.text(0.08, 0.92, "PHOTOGRAPHER IDENTITY PASS", color="#00CC96", 
                 fontsize=12, fontweight="bold", family="sans-serif", transform=ax_base.transAxes)
    
    level_txt = f"LEVEL {card_data.get('overall_level', 80)}"
    ax_base.text(0.92, 0.92, level_txt, color="#0B0E14", fontsize=11, fontweight="bold",
                 family="sans-serif", ha="right", va="center", transform=ax_base.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#00CC96", edgecolor="none"))

    # Divider line
    ax_base.plot([0.08, 0.92], [0.895, 0.895], color="#232B3B", linewidth=1.2, transform=ax_base.transAxes)

    # Title & Loadout
    ax_base.text(0.08, 0.855, card_data.get("title", "Photographer"), color="#FFFFFF", 
                 fontsize=22, fontweight="bold", family="sans-serif", transform=ax_base.transAxes)
    
    loadout = f"LOADOUT: {card_data.get('camera', 'Camera')}  +  {card_data.get('lens', 'Lens')}"
    ax_base.text(0.08, 0.825, loadout, color="#8B949E", fontsize=10, family="sans-serif", transform=ax_base.transAxes)

    # Sweet Spot Banner Box
    sweet_box = plt.Rectangle((0.08, 0.74), 0.84, 0.065, fill=True,
                              facecolor="#19202D", edgecolor="#2D3748",
                              linewidth=1, transform=ax_base.transAxes, zorder=3)
    ax_base.add_patch(sweet_box)
    
    ax_base.text(0.10, 0.785, "SIGNATURE EXPOSURE SWEET-SPOT", color="#58A6FF", 
                 fontsize=8.5, fontweight="bold", transform=ax_base.transAxes, zorder=4)
    ax_base.text(0.10, 0.753, card_data.get("sweet_spot", "f/5.6 • 1/400s • ISO 100"), 
                 color="#00CC96", fontsize=14, fontweight="bold", transform=ax_base.transAxes, zorder=4)

    # Embedded Polar Radar Plot
    ax_polar = fig.add_axes([0.18, 0.40, 0.64, 0.31], polar=True)
    ax_polar.set_facecolor("#12161F")
    
    categories = list(radar_scores.keys())
    values = list(radar_scores.values())
    
    # Close the loop
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    
    ax_polar.plot(angles_closed, values_closed, color="#00CC96", linewidth=2.0)
    ax_polar.fill(angles_closed, values_closed, color="#00CC96", alpha=0.30)
    
    ax_polar.set_xticks(angles)
    ax_polar.set_xticklabels(categories, color="#E6EDF3", fontsize=9, fontweight="bold")
    ax_polar.set_ylim(0, 100)
    ax_polar.set_yticks([25, 50, 75, 100])
    ax_polar.set_yticklabels(["25", "50", "75", "100"], color="#6E7681", fontsize=7)
    ax_polar.spines["polar"].set_color("#232B3B")
    ax_polar.grid(color="#232B3B", linewidth=0.8)

    # Superpower Box
    super_box = plt.Rectangle((0.08, 0.28), 0.40, 0.08, fill=True,
                              facecolor="#12241F", edgecolor="#00CC96",
                              linewidth=1.2, transform=ax_base.transAxes)
    ax_base.add_patch(super_box)
    ax_base.text(0.10, 0.335, "★ SUPERPOWER", color="#3FB950", fontsize=8, fontweight="bold", transform=ax_base.transAxes)
    ax_base.text(0.10, 0.30, card_data.get("top_stat", "Stability (96/100)"), color="#FFFFFF", fontsize=11, fontweight="bold", transform=ax_base.transAxes)

    # Target Box
    target_box = plt.Rectangle((0.52, 0.28), 0.40, 0.08, fill=True,
                               facecolor="#281A1C", edgecolor="#F85149",
                               linewidth=1.2, transform=ax_base.transAxes)
    ax_base.add_patch(target_box)
    ax_base.text(0.54, 0.335, "TARGET GROWTH", color="#F85149", fontsize=8, fontweight="bold", transform=ax_base.transAxes)
    ax_base.text(0.54, 0.30, card_data.get("lowest_stat", "Lighting (46/100)"), color="#FFFFFF", fontsize=11, fontweight="bold", transform=ax_base.transAxes)

    # Quest Box
    quest_box = plt.Rectangle((0.08, 0.12), 0.84, 0.13, fill=True,
                              facecolor="#19202D", edgecolor="#2D3748",
                              linewidth=1, transform=ax_base.transAxes)
    ax_base.add_patch(quest_box)
    ax_base.text(0.10, 0.22, "NEXT GROWTH QUEST:", color="#E3B341", fontsize=9, fontweight="bold", transform=ax_base.transAxes)
    
    quest_str = card_data.get("growth_quest", "Keep shooting in varied conditions!")
    wrapped_quest = "\n".join(textwrap.wrap(quest_str, width=54))
    ax_base.text(0.10, 0.18, wrapped_quest, color="#C9D1D9", fontsize=9.5, linespacing=1.3, transform=ax_base.transAxes, va="top")

    # Footer
    ax_base.text(0.50, 0.06, "PHOTOIQ ENGINE • EXIF SIGNAL AUDIT • 2026", color="#484F58", 
                 fontsize=8.5, fontweight="bold", family="sans-serif", ha="center", transform=ax_base.transAxes)

    # Export to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return buf.getvalue()
