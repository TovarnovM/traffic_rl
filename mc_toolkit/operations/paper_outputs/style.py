"""Small, shared Physica A style map; color always denotes method."""

METHODS = {
    "B1": ("#4B4B4B", "o", ":", False),
    "B2": ("#927C63", "D", (0, (7, 3)), False),
    "noop": ("#879BA8", "x", "-.", False),
    "original": ("#C29A2E", "v", (0, (3, 2)), False),
    "H2": ("#80551E", "v", (0, (7, 2, 1, 2)), True),
    "Graph": ("#3850A0", "P", "-", True),
    "Node": ("#147D92", "s", "-", True),
    "Grid": ("#B4475E", "^", "-", True),
    "H1": ("#756247", "<", ":", False),
    "H4": ("#756247", ">", "--", False),
    "H8": ("#756247", "h", "-.", False),
}
ALIASES = {
    "baseline1_hdv": "B1", "baseline2_matched": "B2",
    "left_if_safe": "original", "original_left_if_safe": "original",
    "original left-if-safe": "original", "graph_ppo": "Graph",
    "node_mlp_ppo": "Node", "node_mlp": "Node", "grid_cnn_ppo": "Grid",
}
RC = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .5,
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "none",
}


def canonical_method(method):
    return ALIASES.get(method, next((m for m in METHODS if m.lower() == method.lower()), method))


def method_style(method):
    color, marker, line, filled = METHODS.get(canonical_method(method.split("−", 1)[0].strip()), ("#555555", "o", "--", False))
    return dict(color=color, marker=marker, linestyle=line,
                markerfacecolor=color if filled else "none", linewidth=1.4)
