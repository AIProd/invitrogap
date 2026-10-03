import re
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import streamlit as st
import matplotlib.pyplot as plt

from skimage.filters import gaussian, sobel
from skimage import morphology, measure


# =====================================================
# Core image / mask utilities
# =====================================================

def to_gray(image_pil: Image.Image, gaussian_sigma: float) -> np.ndarray:
    """
    Convert PIL image to grayscale float64 in [0..1] and apply Gaussian blur.
    """
    gray = np.array(ImageOps.grayscale(image_pil)).astype(np.float32)
    blurred = gaussian(gray, sigma=gaussian_sigma)
    return blurred  # float64 after gaussian()


def build_wound_mask_from_t0(
    gray_blur: np.ndarray,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify the wound region at time 0.
    Steps:
        1. Sobel gradient
        2. Lowest percentile = smooth gap
        3. Morph cleanup
        4. Keep largest connected component
    Returns:
        wound_mask (bool HxW)
        grad0 (float HxW)
    """
    grad0 = sobel(gray_blur)

    thr = np.percentile(grad0, wound_low_grad_percentile)
    wound_candidate = grad0 < thr

    wound_candidate = morphology.remove_small_objects(
        wound_candidate, min_size=min_wound_size
    )
    wound_candidate = morphology.binary_closing(
        wound_candidate, morphology.disk(morph_kernel_radius)
    )
    wound_candidate = morphology.binary_opening(
        wound_candidate, morphology.disk(morph_kernel_radius)
    )

    labeled, _ = measure.label(wound_candidate, return_num=True)
    sizes = np.bincount(labeled.ravel())
    if sizes.size == 0:
        raise ValueError("No wound-like region detected in first frame.")
    sizes[0] = 0  # ignore background
    biggest_label = sizes.argmax()
    wound_mask = labeled == biggest_label

    return wound_mask, grad0


def make_band_mask(
    wound_mask: np.ndarray,
    band_thickness_px: int,
) -> np.ndarray:
    """
    Build a ring just outside the wound. Used as the confluent monolayer
    reference for normalization.
    """
    dilated = morphology.binary_dilation(
        wound_mask, morphology.disk(band_thickness_px)
    )
    band_mask = np.logical_and(dilated, ~wound_mask)
    return band_mask


def parse_hours_from_name(name: str) -> float:
    """
    Extract time (hours) from filename.
    Supports:
        - "01d00h00m" -> days + hours
        - "24H", "72 H" -> hours
    Fallback: 0
    """
    m = re.search(r'(\d+)\s*[dD]\s*(\d+)\s*[hH]', name)
    if m:
        days = float(m.group(1))
        hours = float(m.group(2))
        return days * 24.0 + hours

    m = re.search(r'(\d+)\s*[hH]', name)
    if m:
        return float(m.group(1))

    return 0.0


def overlay_debug_rgb(
    img_pil: Image.Image,
    wound_mask: np.ndarray,
    wound_cells_mask: np.ndarray,
    alpha_wound: float = 0.4,
    alpha_cells: float = 0.4,
) -> Image.Image:
    """
    RGB overlay for QC:
      - Wound region from t0 tinted blue (open region definition)
      - Cells inside wound tinted green (closed area / closure)
    """
    base = np.array(img_pil.convert("RGB")).astype(np.float32)
    out = base.copy()

    blue = np.array([0, 0, 255], dtype=np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)

    # First tint the entire wound band blue = open region at t0
    out[wound_mask] = (1 - alpha_wound) * out[wound_mask] + alpha_wound * blue
    # Then override cell-covered wound area with green = closure
    out[wound_cells_mask] = (
        (1 - alpha_cells) * out[wound_cells_mask] + alpha_cells * green
    )

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# =====================================================
# Metric computation
# =====================================================

def _cell_threshold(grad: np.ndarray, band_mask: np.ndarray, cell_percentile: float):
    """
    Adaptive texture threshold:
    Take a percentile of Sobel gradient in the band region.
    Lower percentile -> more sensitive to faint cells.
    """
    if band_mask.sum() == 0:
        return np.percentile(grad, cell_percentile)
    return np.percentile(grad[band_mask], cell_percentile)


def analyze_timepoint(
    gray_blur: np.ndarray,
    wound_mask: np.ndarray,
    band_mask: np.ndarray,
    w0_frac: float,
    cell_percentile: float,
    min_cell_size: int,
) -> Dict[str, float]:
    """
    Compute timepoint metrics.

    Wound Confluence (%):
        Fraction of the original wound area now classified as cells * 100.

    Relative Wound Density (%):
        RWD(t) = 100 * ( w(t) - w(0) ) / ( c(t) - w(0) )
        where
            w(t) = wound cell fraction at time t
            c(t) = band cell fraction at time t
            w(0) = wound cell fraction at t=0
    """
    grad = sobel(gray_blur)
    thr_cell = _cell_threshold(grad, band_mask, cell_percentile)

    # raw cell masks
    wound_cells_mask = np.logical_and(wound_mask, grad > thr_cell)
    band_cells_mask = np.logical_and(band_mask, grad > thr_cell)

    # --- KEY: remove tiny debris / noise (Incucyte ignores these) ---
    if min_cell_size > 0:
        wound_cells_mask = morphology.remove_small_objects(
            wound_cells_mask, min_size=min_cell_size
        )
        band_cells_mask = morphology.remove_small_objects(
            band_cells_mask, min_size=min_cell_size
        )

    wound_area = wound_mask.sum()
    band_area = band_mask.sum() if band_mask.sum() > 0 else 1

    w_frac = wound_cells_mask.sum() / max(wound_area, 1)
    c_frac = band_cells_mask.sum() / band_area

    wound_confluence_pct = 100.0 * w_frac

    denom = (c_frac - w0_frac)
    if abs(denom) < 1e-9:
        rwd_pct = 0.0
    else:
        rwd_pct = 100.0 * (w_frac - w0_frac) / denom

    rwd_pct = float(np.clip(rwd_pct, 0, 100))

    return {
        "wound_confluence_pct": float(wound_confluence_pct),
        "relative_wound_density_pct": float(rwd_pct),
        "w_frac": float(w_frac),
        "c_frac": float(c_frac),
        "wound_cells_mask": wound_cells_mask,
    }


def run_full_analysis(
    images: List[Image.Image],
    names: List[str],
    gaussian_sigma: float,
    wound_low_grad_percentile: float,
    morph_kernel_radius: int,
    min_wound_size: int,
    band_thickness_px: int,
    cell_percentile: float,
    min_cell_size: int,
) -> Tuple[pd.DataFrame, List[Image.Image]]:
    """
    Full analysis for one well / condition.
    """
    # sort by time from filename
    hours_list = [parse_hours_from_name(n) for n in names]
    order = np.argsort(hours_list)

    images_sorted = [images[i] for i in order]
    names_sorted = [names[i] for i in order]
    hours_sorted = [hours_list[i] for i in order]

    gray_series = [to_gray(im, gaussian_sigma) for im in images_sorted]

    # wound from first timepoint
    wound_mask, _grad0 = build_wound_mask_from_t0(
        gray_series[0],
        wound_low_grad_percentile,
        morph_kernel_radius,
        min_wound_size,
    )

    # reference band outside the wound
    band_mask = make_band_mask(wound_mask, band_thickness_px)

    # baseline wound cell fraction w0
    grad_first = sobel(gray_series[0])
    thr_cell_first = _cell_threshold(grad_first, band_mask, cell_percentile)
    wound_cells_first = np.logical_and(wound_mask, grad_first > thr_cell_first)

    if min_cell_size > 0:
        wound_cells_first = morphology.remove_small_objects(
            wound_cells_first, min_size=min_cell_size
        )

    w0_frac = wound_cells_first.sum() / max(wound_mask.sum(), 1)

    rows = []
    overlays = []

    for img_pil, gray_img, hr, nm in zip(
        images_sorted, gray_series, hours_sorted, names_sorted
    ):
        metrics = analyze_timepoint(
            gray_img,
            wound_mask,
            band_mask,
            w0_frac=w0_frac,
            cell_percentile=cell_percentile,
            min_cell_size=min_cell_size,
        )

        rows.append({
            "Image": nm,
            "Hours": hr,
            "Wound Confluence (%)": metrics["wound_confluence_pct"],
            "Relative Wound Density (%)": metrics["relative_wound_density_pct"],
        })

        # overlay for QC (blue = wound region, green = closed wound)
        wound_cells_now = metrics["wound_cells_mask"]
        ov = overlay_debug_rgb(img_pil, wound_mask, wound_cells_now)
        overlays.append(ov)

    df_metrics = pd.DataFrame(rows).sort_values("Hours").reset_index(drop=True)
    return df_metrics, overlays


# =====================================================
# Plotting + export helpers
# =====================================================

def plot_metric(
    hours: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    scale: float,
):
    """
    Small line+marker plot for a single metric.
    scale controls figure size.
    """
    fig_w = 4 * scale
    fig_h = 3 * scale

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    ax.plot(hours, values, marker="o", linewidth=2)
    ax.set_xlabel("Hours")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# =====================================================
# Streamlit UI
# =====================================================

st.set_page_config(page_title="Wound Healing Analysis", layout="wide")

st.title("Wound Healing Analysis")

st.write(
    "Quantifies scratch-wound closure over time for a single well / condition. "
    "Outputs Wound Confluence (%) and Relative Wound Density (%)."
)

with st.form("analysis_form"):
    uploaded_files = st.file_uploader(
        "Upload all timepoints from one well (e.g. 0h, 24h, 48h, 72h). "
        "All images should use the same magnification.",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    st.markdown("### Analysis settings (optional)")
    st.caption(
        "Tweak only if the wound mask or cell detection looks off. "
        "Defaults are tuned to approximate Incucyte for your RCC example."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gaussian_sigma = st.slider(
            "Gaussian blur σ",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Pre-smoothing before edge detection. Higher values reduce noise but can soften wound edges.",
        )
        wound_low_grad_percentile = st.slider(
            "Wound smoothness percentile",
            min_value=5,
            max_value=60,
            value=30,
            step=1,
            help="Lower = narrower wound mask; higher = wider wound mask. "
                 "Pixels with Sobel gradient below this percentile are considered wound.",
        )

    with col2:
        morph_kernel_radius = st.slider(
            "Wound edge smoothing (px)",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Radius for morphological open/close. Larger = smoother wound band.",
        )
        band_thickness_px = st.slider(
            "Reference band thickness (px)",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            help="Thickness of the ring outside the wound used as the reference monolayer.",
        )

    with col3:
        min_wound_size = st.number_input(
            "Min wound size (px area)",
            min_value=100,
            max_value=200000,
            value=500,
            step=100,
            help="Ignore wound candidates smaller than this area.",
        )
        cell_percentile = st.slider(
            "Cell texture percentile",
            min_value=1,
            max_value=50,
            value=5,   # tuned default (was 10)
            step=1,
            help="Lower = more sensitive to faint/transparent cells (higher confluence). "
                 "Higher = stricter.",
        )

    with col4:
        min_cell_size = st.slider(
            "Min cell object size (px)",
            min_value=0,
            max_value=500,
            value=80,
            step=10,
            help="Objects smaller than this (in wound/band) are treated as debris and ignored. "
                 "Increase to remove small specks; decrease if real cells are being removed.",
        )

    st.markdown("### Display settings")
    disp_col1, disp_col2 = st.columns(2)
    with disp_col1:
        plot_scale = st.slider(
            "Plot scale",
            min_value=0.5,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="Controls plot size. Lower = smaller plots.",
        )
    with disp_col2:
        overlay_cols = st.slider(
            "Overlay columns",
            min_value=2,
            max_value=4,
            value=3,
            step=1,
            help="How many overlay previews per row.",
        )

    submitted = st.form_submit_button("Analyze")

if submitted:
    if not uploaded_files:
        st.warning("Please upload at least one image series.")
    else:
        imgs = [Image.open(f).convert("RGB") for f in uploaded_files]
        names = [f.name for f in uploaded_files]

        try:
            df_metrics, overlays = run_full_analysis(
                images=imgs,
                names=names,
                gaussian_sigma=gaussian_sigma,
                wound_low_grad_percentile=wound_low_grad_percentile,
                morph_kernel_radius=morph_kernel_radius,
                min_wound_size=min_wound_size,
                band_thickness_px=band_thickness_px,
                cell_percentile=cell_percentile,
                min_cell_size=min_cell_size,
            )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        else:
            st.header("Metrics")

            styled = df_metrics.style.format({
                "Hours": "{:.2f}",
                "Wound Confluence (%)": "{:.2f}",
                "Relative Wound Density (%)": "{:.2f}",
            })
            st.dataframe(styled, use_container_width=True)

            csv_data = df_to_csv_bytes(df_metrics)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="wound_metrics.csv",
                mime="text/csv",
            )

            st.header("Time-Series Plots")

            hours_arr = df_metrics["Hours"].to_numpy(dtype=float)

            conf_arr = df_metrics["Wound Confluence (%)"].to_numpy(dtype=float)
            rwd_arr = df_metrics["Relative Wound Density (%)"].to_numpy(dtype=float)

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                fig_conf = plot_metric(
                    hours_arr,
                    conf_arr,
                    ylabel="Wound Confluence (%)",
                    title="Wound Confluence vs Time",
                    scale=plot_scale,
                )
                st.pyplot(fig_conf, clear_figure=True)

            with pcol2:
                fig_rwd = plot_metric(
                    hours_arr,
                    rwd_arr,
                    ylabel="Relative Wound Density (%)",
                    title="Relative Wound Density vs Time",
                    scale=plot_scale,
                )
                st.pyplot(fig_rwd, clear_figure=True)

            st.header("Overlay QC")
            st.caption(
                "Blue: wound region defined at first timepoint.  "
                "Green: detected cells inside that wound region at each timepoint (closure)."
            )

            cols = st.columns(overlay_cols)
            for i, (row, overlay_img) in enumerate(zip(df_metrics.itertuples(index=False), overlays)):
                col = cols[i % overlay_cols]
                with col:
                    st.caption(f"{row.Image}  ({row.Hours:.2f} h)")
                    st.image(overlay_img, use_container_width=True)
