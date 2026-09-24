"""Room-specific furniture built from collidable parts, shared by simulator and viewer."""


def furnish(scene, room, origin):
    wood, fabric, pale, metal = (
        (0.43, 0.32, 0.23, 1),
        (0.32, 0.43, 0.37, 1),
        (0.79, 0.76, 0.69, 1),
        (0.25, 0.28, 0.27, 1),
    )

    def part(name, center, half, color=wood):
        return scene.box(
            room + "_" + name,
            (origin[0] + center[0], origin[1] + center[1], center[2]),
            half,
            color,
        )

    def table(name, x, y, sx=0.8, sy=0.5, z=0.85):
        part(name + "_top", (x, y, z), (sx, sy, 0.045))
        for dx in (-sx + 0.08, sx - 0.08):
            for dy in (-sy + 0.08, sy - 0.08):
                part(name + "_leg", (x + dx, y + dy, z / 2), (0.035, 0.035, z / 2), metal)

    def chair(x, y):
        part("chair_seat", (x, y, 0.48), (0.24, 0.25, 0.06), fabric)
        part("chair_back", (x, y + 0.24, 0.77), (0.24, 0.045, 0.25), fabric)
        for dx in (-0.19, 0.19):
            for dy in (-0.19, 0.19):
                part("chair_leg", (x + dx, y + dy, 0.22), (0.025, 0.025, 0.22), metal)

    def shelf(x, y):
        for z in (0.18, 0.85, 1.52, 2.19):
            part("shelf", (x, y, z), (0.7, 0.28, 0.035))
        for dx in (-0.7, 0.7):
            part("shelf_side", (x + dx, y, 1.15), (0.04, 0.28, 1.15))
        for layer in (0, 1, 2):
            for book in range(5):
                h = 0.20 + 0.035 * ((book + layer) % 3)
                part(
                    "book",
                    (x - 0.47 + book * 0.19, y, 0.215 + layer * 0.67 + h),
                    (0.065, 0.18, h),
                    (0.34 + book * 0.025, 0.37 + layer * 0.025, 0.36 + book * 0.018, 1),
                )

    part("rug", (1.5, 1.5, 0.012), (1.12, 1.05, 0.012), (0.57, 0.53, 0.43, 1))
    if room in ("entry", "living"):
        part("sofa_base", (1.15, 4.9, 0.25), (1.05, 0.62, 0.16), fabric)
        part("sofa_back", (1.15, 5.47, 0.7), (1.05, 0.10, 0.5), fabric)
        for x in (0.17, 2.13):
            part("sofa_arm", (x, 4.9, 0.57), (0.10, 0.60, 0.23), fabric)
        for x in (0.55, 1.18, 1.81):
            part("cushion", (x, 4.85, 0.48), (0.29, 0.46, 0.12), (0.48, 0.56, 0.48, 1))
        table("coffee_table", 1.5, 1.5, 0.7, 0.5, 0.5)
        part("console", (5.52, 1.3, 0.45), (0.27, 0.75, 0.45))
        part("console_handle", (5.23, 1.3, 0.65), (0.025, 0.16, 0.015), metal)
    elif room == "bedroom":
        part("bed_frame", (1.35, 4.6, 0.24), (1.0, 1.0, 0.16))
        part("mattress", (1.35, 4.6, 0.5), (0.95, 0.96, 0.14), pale)
        part("bed_headboard", (1.35, 5.6, 0.75), (1.0, 0.07, 0.65))
        for x in (0.85, 1.85):
            part("pillow", (x, 5.15, 0.72), (0.36, 0.28, 0.08), (0.87, 0.85, 0.8, 1))
        part("bed_blanket", (1.35, 4.2, 0.66), (0.95, 0.53, 0.035), fabric)
        part("wardrobe", (5.5, 1.2, 1.15), (0.35, 0.7, 1.15))
        for y in (1.08, 1.32):
            part("wardrobe_handle", (5.13, y, 1.2), (0.02, 0.015, 0.2), metal)
        table("nightstand", 0.65, 2.1, 0.32, 0.32, 0.55)
    elif room == "kitchen":
        part("counter", (1.7, 0.43, 0.45), (1.25, 0.35, 0.45), pale)
        part("counter_top", (1.7, 0.43, 0.93), (1.3, 0.38, 0.035), (0.35, 0.36, 0.33, 1))
        part("fridge", (5.48, 1.0, 1.05), (0.35, 0.5, 1.05), (0.67, 0.69, 0.66, 1))
        part("fridge_handle", (5.10, 0.69, 1.25), (0.025, 0.025, 0.3), metal)
        for x in (1.0, 1.4):
            for y in (0.3, 0.57):
                part("hob", (x, y, 0.976), (0.12, 0.10, 0.012), metal)
        table("breakfast_table", 1.5, 4.8, 0.65, 0.5, 0.8)
        chair(1.5, 5.65)
    elif room == "dining":
        table("dining_table", 1.5, 1.5, 0.83, 0.62, 0.82)
        chair(1.2, 2.35)
        chair(1.8, 0.55)
        part("sideboard", (4.75, 5.55, 0.5), (0.68, 0.28, 0.5))
    elif room in ("study", "workshop"):
        table("desk" if room == "study" else "workbench", 1.5, 1.25, 0.85, 0.55, 0.84)
        chair(1.5, 2.18)
        if room == "study":
            part("monitor_stand", (1.5, 0.98, 1.01), (0.04, 0.04, 0.14), metal)
            part("monitor", (1.5, 0.98, 1.24), (0.32, 0.035, 0.20), metal)
        else:
            part("toolbox", (1.85, 1.2, 1.01), (0.26, 0.20, 0.125), (0.4, 0.43, 0.38, 1))
        shelf(4.75, 5.55)
    elif room == "library":
        shelf(4.75, 5.55)
        table("reading_table", 1.5, 1.4, 0.7, 0.5, 0.82)
        chair(1.5, 2.22)
        shelf(1.0, 4.8)

    # A planter assembled from geometry: visually distinct, genuinely collidable.
    if room in ("living", "study", "dining"):
        part("planter", (4.8, 1.05, 0.25), (0.22, 0.22, 0.25), (0.59, 0.55, 0.45, 1))
        part("plant_stem", (4.8, 1.05, 0.66), (0.03, 0.03, 0.25), (0.24, 0.34, 0.23, 1))
        for dx, dy, z in ((-0.12, 0, 0.83), (0.12, 0.06, 0.94), (0, -0.1, 1.08)):
            part(
                "plant_leaves", (4.8 + dx, 1.05 + dy, z), (0.16, 0.14, 0.12), (0.26, 0.39, 0.28, 1)
            )
