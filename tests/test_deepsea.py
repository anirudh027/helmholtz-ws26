import cv2
import numpy as np
import tifffile

from tracking.ctc import load_sequence
from tracking.deepsea import deepsea_to_ctc, instance_mask, read_deepsea


def _write_frame(root, name, cells, edge=None):
    """cells: [(label, x, y, half_width)] drawn as squares; edge: a column cleared through the wmap."""
    for sub in ("images", "masks", "labels", "wmaps"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    mask = np.zeros((40, 60), np.uint8)
    for _, x, y, h in cells:
        mask[y - h : y + h + 1, x - h : x + h + 1] = 255
    wmap = np.zeros_like(mask)
    if edge is not None:
        wmap[:, edge] = 255
    cv2.imwrite(str(root / "images" / f"{name}.png"), mask // 2)
    cv2.imwrite(str(root / "masks" / f"{name}_cell_area_masked.png"), mask)
    cv2.imwrite(str(root / "wmaps" / f"{name}_cell_area_masked.png"), wmap)
    rows = "".join(f"{lbl}\t{x}\t{y}\n" for lbl, x, y, _ in cells)
    (root / "labels" / f"{name}_cell_pos_labels.txt").write_text("label\tx\ty\n" + rows)


def _division(root):
    _write_frame(root, "f000", [("1", 10, 10, 3), ("2", 40, 30, 3)])
    _write_frame(root, "f001", [("1", 11, 10, 3), ("2", 40, 30, 3)])
    _write_frame(root, "f002", [("1_1", 8, 10, 2), ("1_2", 13, 10, 2), ("2", 41, 30, 3)], edge=10)  # daughters touch
    _write_frame(root, "f003", [("1_1", 6, 10, 2), ("1_2", 16, 10, 2), ("2", 41, 30, 3)])


def test_division_becomes_a_ctc_lineage(tmp_path):
    _division(tmp_path / "set_1")
    imgs, masks, tracks, report = read_deepsea(tmp_path / "set_1")
    assert imgs.shape == masks.shape == (4, 40, 60)
    assert tracks == {1: (0, 1, 0), 2: (0, 3, 0), 3: (2, 3, 1), 4: (2, 3, 1)}
    assert report["divisions"] == 1 and report["cells_without_mask"] == 0 and report["links_dropped"] == 0
    assert masks[2][10, 8] == 3 and masks[2][10, 13] == 4  # split along the wmap edge
    assert masks[3][30, 41] == 2  # same id in every frame


def test_touching_cells_without_wmap_are_split_by_nearest_centroid():
    foreground = np.zeros((20, 30), bool)
    foreground[5:15, 5:25] = True
    mask, missing, unlabelled = instance_mask(foreground, [(1, 9, 10), (2, 20, 10)])
    assert missing == [] and unlabelled == 0
    assert mask[10, 6] == 1 and mask[10, 23] == 2
    assert set(np.unique(mask)) == {0, 1, 2}


def test_unmatched_cells_and_components_are_reported():
    foreground = np.zeros((20, 30), bool)
    foreground[2:6, 2:6] = True  # nobody annotated this one
    foreground[10:14, 20:24] = True
    mask, missing, unlabelled = instance_mask(foreground, [(1, 22, 12), (2, 5, 17)])
    assert missing == [2] and unlabelled == 1
    assert set(np.unique(mask)) == {0, 1}


def test_ctc_folder_layout(tmp_path):
    _division(tmp_path / "set_1")
    out = tmp_path / "ctc"
    deepsea_to_ctc(tmp_path / "set_1", out, "01")
    assert (out / "01_GT" / "TRA" / "man_track.txt").read_text() == "1 0 1 0\n2 0 3 0\n3 2 3 1\n4 2 3 1\n"
    assert tifffile.imread(out / "01_GT" / "TRA" / "man_track002.tif")[10, 13] == 4
    assert len(list((out / "01_GT" / "SEG").glob("man_seg*.tif"))) == 4
    imgs, masks = load_sequence(str(out), "01")
    assert imgs.shape == masks.shape == (4, 40, 60)
