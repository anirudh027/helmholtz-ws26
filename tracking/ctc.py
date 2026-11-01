"""Cell Tracking Challenge (CTC) I/O.

The CTC format is the common currency between all tracking backends:
a stack of masks whose labels are global track ids, plus a table
``{track_id: (t_start, t_end, parent_id)}`` (``res_track.txt``).

``load_sequence``, ``write_ctc`` and ``tracks_from_assignment`` are adapted
from Damià Vicens Ramis' Cell-project (scripts/ctc_io.py).
"""

import glob
import os
import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy.ndimage as ndi
import tifffile

Tracks = dict[int, tuple[int, int, int]]


@dataclass
class CTCResult:
    """Tracking output: track-id masks (T, Y, X) + CTC lineage table."""

    masks: np.ndarray
    tracks: Tracks
    info: dict = field(default_factory=dict)

    # Unpacks like Nikolas' original API: `graph, masks_tracked = track(...)`
    def __iter__(self):
        return iter((self.graph, self.masks))

    @property
    def graph(self) -> nx.DiGraph:
        return ctc_to_graph(self.masks, self.tracks)


def load_sequence(ds_dir, seq="01", limit=None, mask_dir=None):
    """Return (imgs, masks) as (T, Y, X) arrays from a CTC dataset folder.

    Masks default to the silver-truth segmentation ``<seq>_ST/SEG`` so every
    tracker links the same input; pass ``mask_dir`` to use other masks.
    """
    img_files = sorted(glob.glob(os.path.join(ds_dir, seq, "*.tif")))
    mask_dir = mask_dir or os.path.join(ds_dir, f"{seq}_ST", "SEG")
    seg_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
    assert len(img_files) == len(seg_files), (len(img_files), len(seg_files))
    if limit:
        img_files, seg_files = img_files[:limit], seg_files[:limit]
    imgs = np.stack([np.squeeze(tifffile.imread(f)) for f in img_files])
    masks = np.stack([np.squeeze(tifffile.imread(f)) for f in seg_files])
    return imgs, masks


def write_ctc(out_dir, masks, tracks: Tracks):
    """Write CTC RES format (``maskTTT.tif`` + ``res_track.txt``)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*")):
        os.remove(f)
    ndigits = max(3, len(str(len(masks) - 1)))
    for t, m in enumerate(masks):
        tifffile.imwrite(
            os.path.join(out_dir, f"mask{t:0{ndigits}d}.tif"),
            m.astype(np.uint16), compression="zlib")
    with open(os.path.join(out_dir, "res_track.txt"), "w") as fh:
        for tid in sorted(tracks):
            t0, t1, parent = tracks[tid]
            fh.write(f"{tid} {t0} {t1} {parent}\n")


def _frame_number(path):
    return int(re.findall(r"\d+", os.path.basename(path))[-1])


def read_ctc(res_dir) -> CTCResult:
    """Read a CTC RES folder back into a CTCResult."""
    # Sort numerically: writers like Ultrack switch from mask999 to mask1000,
    # which a plain string sort would interleave (mask100, mask1000, mask101, ...).
    mask_files = sorted(glob.glob(os.path.join(res_dir, "mask*.tif")), key=_frame_number)
    masks = np.stack([np.squeeze(tifffile.imread(f)) for f in mask_files]).astype(np.uint16)
    tracks = {}
    with open(os.path.join(res_dir, "res_track.txt")) as fh:
        for line in fh:
            if line.strip():
                i, t0, t1, par = map(int, line.split())
                tracks[i] = (t0, t1, par)
    return CTCResult(masks, tracks)


def write_ctc_input(ds_dir, seq, imgs, masks):
    """Write raw images + segmentation as a CTC *input* folder
    (``<seq>/tTTT.tif`` and ``<seq>_ST/SEG/man_segTTT.tif``), so any dataset
    can be tracked and evaluated with the same code path."""
    img_dir = os.path.join(ds_dir, seq)
    seg_dir = os.path.join(ds_dir, f"{seq}_ST", "SEG")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    ndigits = max(3, len(str(len(imgs) - 1)))
    for t, (img, m) in enumerate(zip(imgs, masks)):
        tifffile.imwrite(os.path.join(img_dir, f"t{t:0{ndigits}d}.tif"), img)
        tifffile.imwrite(os.path.join(seg_dir, f"man_seg{t:0{ndigits}d}.tif"),
                         m.astype(np.uint16), compression="zlib")


def centroids_of(mask):
    """Return {label: (y, x)} centroids of a 2-D label image."""
    labels = np.unique(mask)
    labels = labels[labels > 0]
    cents = ndi.center_of_mass(mask > 0, mask, labels)
    return {int(lbl): c for lbl, c in zip(labels, cents)}


def tracks_from_assignment(masks, frame_label_to_track, edges):
    """Build a CTCResult from per-frame ``{orig_label: track_id}`` maps and
    parent edges ``{child_track: parent_track}``.

    Splits non-contiguous track ids into CTC-legal contiguous tracklets; a
    later tracklet of the same track gets the previous one as parent.
    """
    T = len(masks)
    frames_of = {}
    for t in range(T):
        for _, tid in frame_label_to_track[t].items():
            frames_of.setdefault(tid, []).append(t)

    new_id = 1
    runs = {}  # new_id -> (t0, t1, orig_tid, prev_run_new_id)
    orig_to_new_at = {}  # (orig_tid, t) -> new_id
    for tid in sorted(frames_of):
        fs = sorted(set(frames_of[tid]))
        run_start = prev_f = fs[0]
        prev_run = None
        for f in fs[1:] + [None]:
            if f is None or f != prev_f + 1:
                runs[new_id] = (run_start, prev_f, tid, prev_run)
                for tt in range(run_start, prev_f + 1):
                    orig_to_new_at[(tid, tt)] = new_id
                prev_run = new_id
                new_id += 1
                if f is not None:
                    run_start = f
            prev_f = f if f is not None else prev_f

    tracks = {}
    for nid, (t0, t1, tid, prev_run) in runs.items():
        if prev_run is not None:
            parent = prev_run
        else:
            parent_tid = edges.get(tid, 0)
            if parent_tid and (parent_tid, t0 - 1) in orig_to_new_at:
                parent = orig_to_new_at[(parent_tid, t0 - 1)]
            elif parent_tid:
                # parent not present exactly at t0-1: take its latest earlier frame
                cand = [tt for (ptid, tt) in orig_to_new_at if ptid == parent_tid and tt < t0]
                parent = orig_to_new_at[(parent_tid, max(cand))] if cand else 0
            else:
                parent = 0
        tracks[nid] = (t0, t1, parent)

    relabeled = np.zeros(np.shape(masks), dtype=np.uint16)
    for t in range(T):
        m = masks[t]
        for lbl, tid in frame_label_to_track[t].items():
            nid = orig_to_new_at.get((tid, t))
            if nid:
                relabeled[t][m == lbl] = nid
    return CTCResult(relabeled, tracks)


def ctc_to_graph(masks, tracks: Tracks) -> nx.DiGraph:
    """Detection-level graph: one node per (track, frame), edges between
    consecutive frames of a track and from a mother's last node to each
    daughter's first node. Node attrs: ``time``, ``label`` (track id),
    ``coords`` (centroid)."""
    g = nx.DiGraph()
    for t, m in enumerate(masks):
        for tid, c in centroids_of(m).items():
            g.add_node((t, tid), time=t, label=tid, coords=np.asarray(c))
    for tid, (t0, t1, parent) in tracks.items():
        for t in range(t0, t1):
            if (t, tid) in g and (t + 1, tid) in g:
                g.add_edge((t, tid), (t + 1, tid))
        if parent and parent in tracks:
            p_end = tracks[parent][1]
            if (p_end, parent) in g and (t0, tid) in g:
                g.add_edge((p_end, parent), (t0, tid))
    return g
