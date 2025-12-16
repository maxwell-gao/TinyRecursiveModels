import xml.etree.ElementTree as ET


def create_rounded_rect(
    parent,
    x,
    y,
    width,
    height,
    r,
    fill,
    stroke,
    stroke_width,
    label=None,
    label_size="14",
    label_y_offset=20,
):
    rect = ET.SubElement(parent, "rect")
    rect.set("x", str(x))
    rect.set("y", str(y))
    rect.set("width", str(width))
    rect.set("height", str(height))
    rect.set("rx", str(r))
    rect.set("ry", str(r))
    rect.set("fill", fill)
    rect.set("stroke", stroke)
    rect.set("stroke-width", str(stroke_width))

    if label:
        lines = label.split("\n")
        for i, line in enumerate(lines):
            text = ET.SubElement(parent, "text")
            text.set("x", str(x + width / 2))
            text.set("y", str(y + label_y_offset + (i * 18)))
            text.set("font-family", "Arial, sans-serif")
            text.set("font-size", label_size)
            text.set("text-anchor", "middle")
            text.set("fill", "black")
            text.text = line


def create_arrow(parent, x1, y1, x2, y2, color="black", width=2, dashed=False):
    line = ET.SubElement(parent, "line")
    line.set("x1", str(x1))
    line.set("y1", str(y1))
    line.set("x2", str(x2))
    line.set("y2", str(y2))
    line.set("stroke", color)
    line.set("stroke-width", str(width))
    line.set("marker-end", "url(#arrowhead)")
    if dashed:
        line.set("stroke-dasharray", "5,5")


def create_curved_arrow(parent, path_d, color="black", width=2, dashed=False):
    path = ET.SubElement(parent, "path")
    path.set("d", path_d)
    path.set("fill", "none")
    path.set("stroke", color)
    path.set("stroke-width", str(width))
    path.set("marker-end", "url(#arrowhead)")
    if dashed:
        path.set("stroke-dasharray", "5,5")


def generate_svg():
    svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width="1200",
        height="1200",
        viewBox="0 0 1200 1200",
    )

    # Definitions for markers
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(
        defs,
        "marker",
        id="arrowhead",
        markerWidth="10",
        markerHeight="7",
        refX="9",
        refY="3.5",
        orient="auto",
    )
    # Remove unused variable assignment
    ET.SubElement(marker, "polygon", points="0 0, 10 3.5, 0 7", fill="black")

    # Background
    # Remove unused variable assignment
    ET.SubElement(svg, "rect", width="100%", height="100%", fill="white")

    # Title
    title = ET.SubElement(
        svg,
        "text",
        x="50",
        y="40",
        text_anchor="start",
        font_family="Arial, sans-serif",
        font_size="24",
        font_weight="bold",
    )
    title.text = "Deep Improvement Supervision (DIS) Architecture"

    subtitle = ET.SubElement(
        svg,
        "text",
        x="50",
        y="70",
        text_anchor="start",
        font_family="Arial, sans-serif",
        font_size="16",
        fill="#555",
    )
    subtitle.text = "Config: loop_transformer_dis.yaml | 6 Supervision Steps | 2 Internal Repeats | No Halting Head"

    # --- Layout Constants ---
    start_x = 50
    step_width = 160
    step_gap = 40
    y_base = 500

    # --- Input ---
    create_rounded_rect(
        svg,
        start_x,
        y_base + 50,
        80,
        60,
        10,
        "#e1f5fe",
        "#0277bd",
        2,
        "Input\n(X)",
        "14",
        25,
    )

    # --- Initial State ---
    create_rounded_rect(
        svg,
        start_x,
        y_base - 50,
        80,
        60,
        10,
        "#fff9c4",
        "#fbc02d",
        2,
        "Init\nState (z)",
        "14",
        25,
    )

    # --- Steps Loop ---
    for i in range(6):
        x_offset = start_x + 120 + i * (step_width + step_gap)

        # Step Container
        create_rounded_rect(
            svg,
            x_offset,
            y_base - 100,
            step_width,
            300,
            15,
            "#f5f5f5",
            "#9e9e9e",
            1,
            "",
            "14",
            20,
        )

        # Step Label
        step_label = ET.SubElement(
            svg,
            "text",
            x=str(x_offset + step_width / 2),
            y=str(y_base - 70),
            text_anchor="middle",
            font_family="Arial, sans-serif",
            font_size="14",
            font_weight="bold",
        )
        step_label.text = f"Step {i + 1}"

        # Step Embedding Injection
        create_rounded_rect(
            svg,
            x_offset + 40,
            y_base + 130,
            80,
            40,
            5,
            "#e0f2f1",
            "#00695c",
            1,
            f"Step Emb\n(s={i})",
            "10",
            15,
        )

        # Transformer Block (Internal Loop)
        create_rounded_rect(
            svg,
            x_offset + 30,
            y_base,
            100,
            80,
            10,
            "#bbdefb",
            "#1976d2",
            2,
            "Transformer\n(Repeat 2x)",
            "12",
            30,
        )

        # Internal Loop Arrow
        create_curved_arrow(
            svg,
            f"M {x_offset + 130} {y_base + 40} C {x_offset + 160} {y_base + 40}, {x_offset + 160} {y_base - 20}, {x_offset + 80} {y_base}",
            color="#1565c0",
            width=1.5,
        )

        # Loss Calculation
        create_rounded_rect(
            svg,
            x_offset + 30,
            y_base - 160,
            100,
            40,
            5,
            "#ffebee",
            "#c62828",
            1,
            "Loss",
            "12",
            25,
        )

        # Target
        create_rounded_rect(
            svg,
            x_offset + 30,
            y_base - 230,
            100,
            40,
            5,
            "#f3e5f5",
            "#8e24aa",
            1,
            f"Target\n(y_{i})",
            "12",
            20,
        )

        # Arrows within step
        # Input X to Transformer
        if i == 0:
            create_curved_arrow(
                svg,
                f"M {start_x + 80} {y_base + 80} C {x_offset} {y_base + 80}, {x_offset} {y_base + 60}, {x_offset + 30} {y_base + 60}",
            )
        else:
            # X is implicitly available, but let's show state flow mainly
            pass

        # Step Emb to Transformer
        create_arrow(svg, x_offset + 80, y_base + 130, x_offset + 80, y_base + 80)

        # Transformer to Loss (Logits)
        create_arrow(
            svg, x_offset + 80, y_base, x_offset + 80, y_base - 120, dashed=True
        )

        # Target to Loss
        create_arrow(svg, x_offset + 80, y_base - 190, x_offset + 80, y_base - 160)

        # State Flow (z)
        if i == 0:
            # Init z to Step 1
            create_curved_arrow(
                svg,
                f"M {start_x + 80} {y_base - 20} C {x_offset} {y_base - 20}, {x_offset} {y_base + 20}, {x_offset + 30} {y_base + 20}",
            )
        else:
            # Previous Step to Current Step
            prev_x = x_offset - (step_width + step_gap) + 30 + 100
            create_arrow(
                svg,
                prev_x,
                y_base + 40,
                x_offset + 30,
                y_base + 40,
                color="#fbc02d",
                width=3,
            )

    # --- Final Output ---
    final_x = start_x + 120 + 5 * (step_width + step_gap) + step_width + 40
    create_rounded_rect(
        svg,
        final_x,
        y_base,
        80,
        60,
        10,
        "#e8f5e9",
        "#2e7d32",
        2,
        "Final\nOutput",
        "14",
        25,
    )

    # Last Step to Output
    last_step_x = start_x + 120 + 5 * (step_width + step_gap) + 30 + 100
    create_arrow(svg, last_step_x, y_base + 40, final_x, y_base + 40, width=3)

    # --- Legend ---
    legend_y = 850
    create_arrow(svg, 100, legend_y, 150, legend_y, color="#fbc02d", width=3)
    ET.SubElement(
        svg, "text", x="160", y=str(legend_y + 5), font_family="Arial", font_size="12"
    ).text = "State (z) Flow"

    create_arrow(svg, 300, legend_y, 350, legend_y, dashed=True)
    ET.SubElement(
        svg, "text", x="360", y=str(legend_y + 5), font_family="Arial", font_size="12"
    ).text = "Supervision Signal (Logits)"

    tree = ET.ElementTree(svg)
    tree.write("loop_transformer_dis_arch.svg")


if __name__ == "__main__":
    generate_svg()
