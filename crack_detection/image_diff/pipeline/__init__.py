"""Building Change Detection — modular pipeline package."""

from .preprocessing import (
    detect_tags,
    compute_canonical_frame,
    normalize_to_canonical,
    histogram_match_l_channel,
)
from .alignment import (
    compute_homography,
    refine_ecc,
    compute_valid_overlap,
    crop_to_overlap,
)
from .diff_methods import compute_diffs
from .postprocessing import (
    build_heatmap_overlay,
    annotate,
    severity_label,
)
from .io import (
    find_pairs,
    save_debug_panel,
    save_result,
    backup_file,
)
