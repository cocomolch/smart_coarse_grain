"""
plot_disclination.py
====================
Visualise a single 3D disclination line/loop together with cropped H&E
XY-slice intersections, rendered as an interactive Plotly HTML file.

Usage (from your pipeline script or standalone):
------------------------------------------------
    from plot_disclination import plot_disclination

    plot_disclination(
        skel_3d      = results_3d["skel_3d"].astype(bool),
        vol          = vol,                          # raw (Z, Y, X) float32 [0,1]
        output_html  = "C:/path/to/output/disc_plot.html",
        component_id = 0,                            # 0 = largest component
        n_intersections = 5,
        crop_px      = 150,
        mum_per_px_xy = cfg["MUM_PER_PX_XY"],
        mum_per_px_z  = cfg["MUM_PER_PX_Z"],
    )

Dependencies:  pip install plotly  (numpy, scipy already in your env)
"""

import numpy as np
from scipy.ndimage import label
import plotly.graph_objects as go
import plotly.io as pio


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_component(skel_3d: np.ndarray, component_id: int = 0):
    """
    Label connected components of the skeleton and return the voxel
    coordinates of the requested component sorted by Z then Y.

    Parameters
    ----------
    skel_3d      : bool array (D, H, W)
    component_id : 0 = largest, 1 = second largest, etc.

    Returns
    -------
    coords : np.ndarray  (N, 3)  columns = [z, y, x]
    """
    labeled, n_comp = label(skel_3d)
    if n_comp == 0:
        raise ValueError("Skeleton is empty — no disclination lines found.")

    # sort components by size descending
    sizes = np.bincount(labeled.ravel())[1:]          # exclude background (0)
    order = np.argsort(sizes)[::-1]
    chosen_label = order[component_id] + 1            # +1 because bincount starts at 0

    coords = np.column_stack(np.where(labeled == chosen_label))   # (N, 3) z,y,x
    # sort by z, then y for a reasonable traversal order
    sort_idx = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[sort_idx].astype(np.float32)


def _pick_intersection_indices(coords: np.ndarray, n: int) -> np.ndarray:
    """
    Pick n indices into coords that are evenly spaced by arc-length.
    Avoids placing two intersections on the same Z-slice.
    """
    if len(coords) <= n:
        return np.arange(len(coords))

    # cumulative arc-length along the sorted coordinate list
    diffs = np.diff(coords, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg_len)])
    total = arc[-1]

    targets = np.linspace(0, total, n + 2)[1:-1]     # avoid endpoints
    indices = np.searchsorted(arc, targets)
    indices = np.clip(indices, 0, len(coords) - 1)

    # deduplicate Z — if two selected points share a Z, drop the duplicate
    seen_z = set()
    final = []
    for idx in indices:
        z = int(coords[idx, 0])
        if z not in seen_z:
            seen_z.add(z)
            final.append(idx)
    return np.array(final, dtype=int)


def _crop_slice(vol: np.ndarray, z: int, cy: float, cx: float,
                crop_px: int) -> np.ndarray:
    """
    Return a (2*crop_px, 2*crop_px) crop of vol[z] centred on (cy, cx).
    Pads with zeros if the crop extends beyond the volume boundary.
    """
    D, H, W = vol.shape
    half = crop_px
    y0, y1 = int(cy) - half, int(cy) + half
    x0, x1 = int(cx) - half, int(cx) + half

    # clamp to valid range
    vy0, vy1 = max(y0, 0), min(y1, H)
    vx0, vx1 = max(x0, 0), min(x1, W)

    patch = vol[z, vy0:vy1, vx0:vx1]

    # zero-pad if needed
    pad_top    = max(0,  -y0)
    pad_bottom = max(0, y1 - H)
    pad_left   = max(0,  -x0)
    pad_right  = max(0, x1 - W)
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)))

    return patch.astype(np.float32)


def _make_image_surface(patch: np.ndarray,
                        z_world: float,
                        cx_world: float, cy_world: float,
                        crop_um: float,
                        colorscale: str = "gray") -> go.Surface:
    """
    Build a Plotly Surface that displays `patch` as a texture on a
    horizontal plane at z_world, centred on (cx_world, cy_world).
    """
    H, W = patch.shape
    xs = np.linspace(cx_world - crop_um, cx_world + crop_um, W)
    ys = np.linspace(cy_world - crop_um, cy_world + crop_um, H)
    X, Y = np.meshgrid(xs, ys)
    Z    = np.full_like(X, z_world)

    # normalise patch to [0, 1] for colourscale
    pmin, pmax = patch.min(), patch.max()
    if pmax > pmin:
        patch_norm = (patch - pmin) / (pmax - pmin)
    else:
        patch_norm = patch

    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=patch_norm,
        colorscale=colorscale,
        showscale=False,
        opacity=0.92,
        name=f"H&E  z={z_world:.0f}µm",
        hoverinfo="name",
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        lightposition=dict(x=0, y=0, z=1e5),
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_disclination(
    skel_3d:         np.ndarray,
    vol:             np.ndarray,
    output_html:     str,
    component_id:    int   = 0,
    n_intersections: int   = 5,
    crop_px:         int   = 150,
    mum_per_px_xy:   float = 1.0,
    mum_per_px_z:    float = 1.0,
    line_color:      str   = "#e63946",
    line_width:      int   = 6,
    marker_size:     int   = 8,
    he_colorscale:   str   = "gray",
    title:           str   = "3D Disclination Line + H&E Intersections",
):
    """
    Render a single disclination component in 3D together with cropped
    H&E XY-slice intersections and save as a standalone HTML file.

    Parameters
    ----------
    skel_3d          : bool array (D, H, W) — skeletonised disclination volume
    vol              : float32 array (D, H, W) — raw intensity volume [0, 1]
    output_html      : path for the output .html file (created/overwritten)
    component_id     : which connected component to plot (0 = largest)
    n_intersections  : how many evenly-spaced XY slices to show
    crop_px          : half-width of the H&E crop in pixels (full side = 2×crop_px)
    mum_per_px_xy    : lateral pixel size [µm/px]  — used for axis labels
    mum_per_px_z     : axial pixel size   [µm/slice]
    line_color       : hex colour for the disclination line
    line_width       : line width in Plotly units
    marker_size      : size of intersection-marker spheres
    he_colorscale    : Plotly colorscale name for H&E patches ("gray" | "RdBu" | …)
    title            : figure title string
    """

    print(f"  Extracting component {component_id} from skeleton …")
    coords = _extract_component(skel_3d, component_id)
    print(f"  Component has {len(coords)} voxels")

    # convert to physical coordinates [µm]
    z_um = coords[:, 0] * mum_per_px_z
    y_um = coords[:, 1] * mum_per_px_xy
    x_um = coords[:, 2] * mum_per_px_xy

    # ── 3D line trace ─────────────────────────────────────────────────────────
    line_trace = go.Scatter3d(
        x=x_um, y=y_um, z=z_um,
        mode="lines",
        line=dict(color=line_color, width=line_width),
        name="Disclination line",
        hovertemplate="x=%{x:.1f}µm<br>y=%{y:.1f}µm<br>z=%{z:.1f}µm<extra></extra>",
    )

    # ── intersection planes ───────────────────────────────────────────────────
    inter_idx = _pick_intersection_indices(coords, n_intersections)
    print(f"  Placing {len(inter_idx)} H&E intersections at Z-slices: "
          f"{[int(coords[i, 0]) for i in inter_idx]}")

    crop_um = crop_px * mum_per_px_xy
    surfaces = []
    marker_xs, marker_ys, marker_zs = [], [], []

    for idx in inter_idx:
        zi   = int(coords[idx, 0])
        cy_v = coords[idx, 1]
        cx_v = coords[idx, 2]

        patch = _crop_slice(vol, zi, cy_v, cx_v, crop_px)

        surf = _make_image_surface(
            patch,
            z_world   = float(zi * mum_per_px_z),
            cx_world  = float(cx_v * mum_per_px_xy),
            cy_world  = float(cy_v * mum_per_px_xy),
            crop_um   = crop_um,
            colorscale= he_colorscale,
        )
        surfaces.append(surf)

        marker_xs.append(cx_v * mum_per_px_xy)
        marker_ys.append(cy_v * mum_per_px_xy)
        marker_zs.append(zi  * mum_per_px_z)

    # spheres marking intersection points on the line
    marker_trace = go.Scatter3d(
        x=marker_xs, y=marker_ys, z=marker_zs,
        mode="markers",
        marker=dict(size=marker_size, color=line_color,
                    symbol="circle", opacity=0.9,
                    line=dict(color="white", width=1)),
        name="Intersections",
        hovertemplate="Intersection<br>x=%{x:.1f}µm<br>y=%{y:.1f}µm<br>z=%{z:.1f}µm<extra></extra>",
    )

    # ── layout ────────────────────────────────────────────────────────────────
    fig = go.Figure(data=[line_trace, marker_trace] + surfaces)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis=dict(title="x [µm]", showgrid=True, gridcolor="#cccccc"),
            yaxis=dict(title="y [µm]", showgrid=True, gridcolor="#cccccc"),
            zaxis=dict(title="z [µm]", showgrid=True, gridcolor="#cccccc"),
            aspectmode="data",                       # preserves physical proportions
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
    )

    pio.write_html(fig, file=output_html, auto_open=True)
    print(f"  Saved → {output_html}  (opens in browser automatically)")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: plot all components above a minimum size
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_components(
    skel_3d:         np.ndarray,
    vol:             np.ndarray,
    output_dir:      str,
    experiment_name: str  = "sample",
    min_voxels:      int  = 50,
    max_components:  int  = 10,
    **kwargs,                          # forwarded to plot_disclination
):
    """
    Plot every connected component in skel_3d that has >= min_voxels,
    saving one HTML per component.

    Parameters
    ----------
    min_voxels     : skip components smaller than this
    max_components : hard cap to avoid generating hundreds of files
    **kwargs       : passed to plot_disclination (crop_px, n_intersections, …)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    labeled, n_comp = label(skel_3d)
    sizes = np.bincount(labeled.ravel())[1:]
    order = np.argsort(sizes)[::-1]

    plotted = 0
    for rank, lab_idx in enumerate(order):
        sz = sizes[lab_idx]
        if sz < min_voxels:
            break
        if plotted >= max_components:
            print(f"  Reached max_components={max_components}, stopping.")
            break

        out_path = os.path.join(output_dir,
                                f"{experiment_name}_disc_comp{rank:03d}_n{sz}.html")
        print(f"\n  Component {rank}  ({sz} voxels) → {out_path}")
        try:
            plot_disclination(
                skel_3d      = skel_3d,
                vol          = vol,
                output_html  = out_path,
                component_id = rank,
                **kwargs,
            )
            plotted += 1
        except Exception as e:
            print(f"  WARNING: skipped component {rank}: {e}")

    print(f"\n  Done — plotted {plotted} components.")