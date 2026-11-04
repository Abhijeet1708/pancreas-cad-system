"""
Streamlit local demo application for the Pancreas CAD System.
"""

from pathlib import Path
import sys

# Ensure repository root is on sys.path for top-level module resolution
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import json  # noqa: E402
import tempfile  # noqa: E402

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

from src.pipeline.display import (  # noqa: E402
    DISCLAIMER,
    LIMITATIONS_MD,
    format_finding_messages,
    list_example_results,
    summary_rows,
    tint_overlay,
)
from src.pipeline.loaders import (  # noqa: E402
    load_cls_model,
    load_seg_model,
    preprocess_ct_volume,
)
from src.pipeline.run import (  # noqa: E402
    analyze_volume,
    clean_for_json,
    save_slice_figures,
)


@st.cache_resource
def get_seg_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Cached loader for 3D segmentation UNet model."""
    return load_seg_model(checkpoint_path, device)


@st.cache_resource
def get_cls_model(checkpoint_path: str, device: str):
    """Cached loader for 2D slice classifier model and decision threshold."""
    return load_cls_model(checkpoint_path, device)


def render_summary_cards(result_dict: dict) -> None:
    """Render summary metrics as Streamlit metric cards in columns."""
    rows = summary_rows(result_dict)
    cols = st.columns(4)
    for i, (label, val) in enumerate(rows):
        cols[i % 4].metric(label, val)


def render_findings(result_dict: dict) -> None:
    """Render segmentation and classifier findings in separate informational boxes."""
    seg_msg, cls_msg = format_finding_messages(result_dict)
    st.subheader("Findings")
    st.info(seg_msg)
    st.info(cls_msg)


def run_examples_mode(examples_root: Path) -> None:
    """Render precomputed example runs."""
    examples = list_example_results(examples_root)
    if not examples:
        st.info(
            f"No precomputed example runs found in `{examples_root.as_posix()}`. "
            "Switch to 'Analyze a CT file' to evaluate a volume."
        )
        return

    case_names = [e["name"] for e in examples]
    selected_name = st.selectbox("Select Example Case", case_names)
    selected_example = next(e for e in examples if e["name"] == selected_name)

    res = selected_example["result"]
    render_summary_cards(res)
    render_findings(res)

    figures = selected_example.get("figures", [])
    if figures:
        st.subheader("Explanation Figures")
        for fig_path in figures:
            st.image(str(fig_path), caption=fig_path.name, use_container_width=True)
        st.caption(
            "Blue = predicted pancreas, red = predicted tumor. Grad-CAM: warmer colors"
            " = regions that pushed the slice classifier toward 'tumor'"
            " (coarse 4 x 4 map, upsampled)."
        )

    with st.expander("Raw result.json", expanded=False):
        st.json(res)


def run_analyze_mode() -> None:
    """Interactive CT volume analysis mode."""
    st.sidebar.header("Model Configuration")
    seg_ckpt_str = st.sidebar.text_input(
        "Segmentation checkpoint",
        value="models/seg_best.pt",
    )
    cls_ckpt_str = st.sidebar.text_input(
        "Classifier checkpoint",
        value="models/cls_best.pt",
    )
    device_choice = st.sidebar.selectbox(
        "Compute Device",
        options=["auto", "cpu", "cuda"],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Upload a CT volume (.nii, .nii.gz, or .npz)",
        type=["nii", "gz", "npz"],
    )

    if uploaded_file is None:
        st.info("Upload a 3D CT scan (.nii, .nii.gz, or .npz) to begin analysis.")
        return

    # Suffix identification
    orig_name = uploaded_file.name.lower()
    if orig_name.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif orig_name.endswith(".nii"):
        suffix = ".nii"
    elif orig_name.endswith(".npz"):
        suffix = ".npz"
    else:
        st.error(
            f"Unsupported file type for `{uploaded_file.name}`. "
            "Please provide a .nii, .nii.gz, or .npz file."
        )
        return

    # Checkpoint existence validation
    seg_p = Path(seg_ckpt_str)
    cls_p = Path(cls_ckpt_str)
    if not seg_p.is_file():
        st.error(f"Segmentation checkpoint not found: `{seg_ckpt_str}`")
        return
    if not cls_p.is_file():
        st.error(f"Classifier checkpoint not found: `{cls_ckpt_str}`")
        return

    # Device resolution
    if device_choice == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_str = device_choice

    if dev_str == "cpu":
        st.warning("Running inference on CPU. Full-volume analysis can take several minutes.")

    # Write upload to temporary file and execute analysis
    temp_input_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_input_path = Path(tmp.name)

        with st.spinner("Analyzing CT volume (preprocessing, 3D segmentation, Grad-CAM)..."):
            image = preprocess_ct_volume(temp_input_path)
            seg_model = get_seg_model(str(seg_p), dev_str)
            cls_model, cls_thresh = get_cls_model(str(cls_p), dev_str)
            analysis = analyze_volume(
                image=image,
                seg_model=seg_model,
                cls_model=cls_model,
                threshold=cls_thresh,
                device=dev_str,
            )
    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        return
    finally:
        if temp_input_path is not None and temp_input_path.is_file():
            try:
                temp_input_path.unlink()
            except OSError:
                pass

    labels = analysis["labels"]
    result = clean_for_json(analysis["result"])
    slices = analysis.get("slices", [])

    # Slice overlay viewer
    st.subheader("Interactive Slice Overlay")
    z_max = image.shape[0] - 1
    # Default to slice with most tumor voxels or middle slice
    tumor_counts = np.count_nonzero(labels == 2, axis=(1, 2))
    if np.any(tumor_counts > 0):
        default_z = int(np.argmax(tumor_counts))
    else:
        default_z = int(z_max // 2)

    z = st.slider("Axial slice (z)", min_value=0, max_value=z_max, value=default_z)
    overlay = tint_overlay(image[z], labels[z])
    st.image(
        overlay,
        caption=f"Slice z={z} (Blue: pancreas, Red: tumor)",
        use_container_width=True,
    )
    st.caption(
        "Blue = predicted pancreas, red = predicted tumor. Grad-CAM: warmer colors"
        " = regions that pushed the slice classifier toward 'tumor'"
        " (coarse 4 x 4 map, upsampled)."
    )

    # Top-k slices figures
    if slices:
        st.subheader("Top Flagged Slices")
        with tempfile.TemporaryDirectory() as fig_dir:
            save_slice_figures(slices, fig_dir)
            fig_files = sorted(Path(fig_dir).glob("slice_*.png"))
            for fig_path in fig_files:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)

    # Summary metrics & findings
    render_summary_cards(result)
    render_findings(result)

    # Download result.json
    result_json_str = json.dumps(result, indent=2)
    st.download_button(
        label="Download result.json",
        data=result_json_str,
        file_name="result.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(
        page_title="Pancreas CAD Demo",
        layout="wide",
    )
    st.title("Pancreas CAD Demo")

    # Permanent warning banner
    st.warning(f"**Warning:** {DISCLAIMER}")

    # Mode selection
    mode = st.radio(
        "Mode",
        options=["Examples (instant, precomputed)", "Analyze a CT file (slow on CPU)"],
        horizontal=True,
    )

    if mode == "Examples (instant, precomputed)":
        repo_root = Path(__file__).resolve().parents[1]
        run_examples_mode(repo_root / "docs" / "results" / "pipeline")
    else:
        run_analyze_mode()

    # Limitations expander
    with st.expander("Limitations and performance", expanded=False):
        st.markdown(LIMITATIONS_MD)


if __name__ == "__main__":
    main()
