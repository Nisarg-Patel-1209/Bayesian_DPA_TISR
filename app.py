import os, io, json, time, zipfile, tempfile
from pathlib import Path

import numpy as np
import torch
import gradio as gr
from huggingface_hub import hf_hub_download, HfApi

import h5py
import tifffile

# Paper code imports (must exist in your Space repo)
from model_2D.models.backbones.sr_backbones import DPATISR
from src.diagram import interval_confidence

# ---------------------------
# HF repos (set in Space vars)
# ---------------------------
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "Nisargpatel1209/huh7_DPA_TISR")  # dataset repo with new_test.h5
HF_DATASET_FILENAME = os.getenv("HF_DATASET_FILENAME", "new_test.h5")

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Nisargpatel1209/huh7_DPA_TISR_weights")   # model repo with .pt weights
DEFAULT_CKPT = os.getenv("HF_DEFAULT_CKPT", "huh7_tail.pt")

# ---------------------------
# Model hyperparams (must match training)
# ---------------------------
MODEL_CFG = dict(
    mid_channels=64,
    extraction_nblocks=3,
    propagation_nblocks=3,
    reconstruction_nblocks=5,
    factor=3,
    bayesian=True,
)

DEFAULT_ENSEMBLES = 6
DEFAULT_EPSILON = 0.04
DEFAULT_FRAME_INDEX = 3  # middle frame of 7


# ---------------------------
# Helpers
# ---------------------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def enable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def robust_norm01(x: np.ndarray, p1=1.0, p99=99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, p1)
    hi = np.percentile(x, p99)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = (np.clip(x, lo, hi) - lo) / (hi - lo)
    return y.astype(np.float32)


def to_uint8_gray(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0).astype(np.uint8)


def apply_cmap(u01: np.ndarray, cmap_name="magma") -> np.ndarray:
    # returns RGB uint8
    import matplotlib
    import matplotlib.cm as cm
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(np.clip(u01, 0.0, 1.0))
    rgb = (rgba[:, :, :3] * 255.0).astype(np.uint8)
    return rgb


def ensure_7_frames(arr_t_hw: np.ndarray, start: int = 0):
    # arr_t_hw: (T,H,W)
    T = arr_t_hw.shape[0]
    if T < 7:
        raise ValueError(f"Need at least 7 frames, got {T}.")
    start = int(np.clip(start, 0, T - 7))
    return arr_t_hw[start:start+7]


def read_timelapse_from_upload(upload_path: str):
    """
    Returns (T,H,W) float32 array from:
      - multipage tif/tiff
      - zip of frames
      - npy
    """
    p = Path(upload_path)
    suf = p.suffix.lower()

    if suf == ".npy":
        arr = np.load(p)
        if arr.ndim != 3:
            raise ValueError(f".npy must be shape (T,H,W); got {arr.shape}")
        return arr.astype(np.float32)

    if suf in [".tif", ".tiff"]:
        arr = tifffile.imread(str(p))  # (T,H,W) or (H,W) if single
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(f"TIFF must decode to (T,H,W); got {arr.shape}")
        return arr.astype(np.float32)

    if suf == ".zip":
        import zipfile
        tmpdir = Path(tempfile.mkdtemp(prefix="upl_zip_"))
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(tmpdir)

        # gather frames (prefer t*.tif/png/jpg)
        frames = []
        for ext in ("tif","tiff","png","jpg","jpeg"):
            frames += sorted(tmpdir.rglob(f"t*.{ext}"))
        if not frames:
            # fallback: any image file
            for ext in ("tif","tiff","png","jpg","jpeg"):
                frames += sorted(tmpdir.rglob(f"*.{ext}"))

        if not frames:
            raise ValueError("ZIP contains no readable frames (tif/png/jpg).")

        imgs = []
        for fp in frames:
            if fp.suffix.lower() in [".tif",".tiff"]:
                im = tifffile.imread(str(fp))
            else:
                from PIL import Image
                im = np.array(Image.open(fp))
            if im.ndim == 3:
                im = im[..., 0]  # take first channel if RGB
            imgs.append(im.astype(np.float32))

        # stack to (T,H,W)
        H, W = imgs[0].shape
        for i, im in enumerate(imgs):
            if im.shape != (H, W):
                raise ValueError(f"Frame {i} shape {im.shape} != {(H,W)} (all frames must match)")
        return np.stack(imgs, axis=0).astype(np.float32)

    raise ValueError(f"Unsupported upload type: {suf}. Use .tif/.tiff, .zip, or .npy")


# ---------------------------
# HF caching
# ---------------------------
CACHE_DIR = Path("./_hf_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_CACHE = {}
_DATASET_PATH = None
_TEST_KEYS = None


def list_checkpoints(repo_id: str):
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    pts = [f for f in files if f.endswith(".pt") or f.endswith(".pth")]
    pts = sorted(pts)
    return pts


def ensure_test_h5():
    global _DATASET_PATH, _TEST_KEYS
    if _DATASET_PATH and Path(_DATASET_PATH).exists() and _TEST_KEYS:
        return _DATASET_PATH, _TEST_KEYS

    if "YOUR_USERNAME/YOUR_H5_DATASET_REPO" in HF_DATASET_REPO:
        raise RuntimeError("Set HF_DATASET_REPO in Space Settings → Variables (dataset repo that contains new_test.h5).")

    local = Path("./dataset") / HF_DATASET_FILENAME
    local.parent.mkdir(parents=True, exist_ok=True)

    if not local.exists():
        p = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=HF_DATASET_FILENAME,
            repo_type="dataset",
            cache_dir=str(CACHE_DIR),
        )
        # copy into ./dataset
        Path(p).replace(local) if not local.exists() else None

    _DATASET_PATH = str(local)
    with h5py.File(_DATASET_PATH, "r") as f:
        keys = sorted(list(f["lr"].keys()), key=lambda s: int(s) if s.isdigit() else s)
    _TEST_KEYS = keys
    return _DATASET_PATH, _TEST_KEYS


def load_model(ckpt_name: str):
    if ckpt_name in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_name]

    ckpt_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=ckpt_name,
        repo_type="model",
        cache_dir=str(CACHE_DIR),
    )

    dev = get_device()
    model = DPATISR(**MODEL_CFG)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(dev)
    model.eval()
    if MODEL_CFG.get("bayesian", False):
        enable_dropout(model)

    _MODEL_CACHE[ckpt_name] = model
    return model


# ---------------------------
# Inference core
# ---------------------------
def run_mc_inference_lr(lr_7_hw: np.ndarray, ckpt_name: str, ensembles: int, epsilon: float, frame_index: int):
    """
    lr_7_hw: (7,H,W) float32
    Returns sr, conf, data_unc, model_unc arrays (H*factor, W*factor) float32
    """
    lr_7_hw = lr_7_hw.astype(np.float32)

    # normalize to 0..1 robustly (works for 0..255 or weird ranges)
    lr01 = robust_norm01(lr_7_hw)

    inp = torch.from_numpy(lr01).unsqueeze(0).unsqueeze(2)  # (1,7,1,H,W)
    inp = inp.to(get_device()).float()

    model = load_model(ckpt_name)

    means = []
    stds = []
    t0 = time.time()
    with torch.no_grad():
        for _ in range(ensembles):
            out = model(inp)                      # (1,7,2, H*F, W*F) for bayesian
            out = out[:, frame_index, :, :, :]    # (1,2, H*F, W*F)
            mean = out[0, 0].detach().cpu().numpy().astype(np.float32)
            means.append(mean)
            if out.shape[1] > 1:
                std = out[0, 1].detach().cpu().numpy().astype(np.float32)
                stds.append(std)

    means = np.stack(means, axis=0)              # (E, Hs, Ws)
    sr = means.mean(axis=0)                      # (Hs, Ws)
    model_unc = means.std(axis=0)                # (Hs, Ws)

    if len(stds) > 0:
        stds = np.stack(stds, axis=0)            # (E, Hs, Ws)
        data_unc = stds.mean(axis=0)
        conf = interval_confidence(means, stds, epsilon, ensembles).astype(np.float32)
    else:
        data_unc = np.zeros_like(sr, dtype=np.float32)
        conf = np.zeros_like(sr, dtype=np.float32)

    dt = time.time() - t0
    return sr, conf, data_unc, model_unc, dt


def pack_zip(sr_rgb, conf_rgb, datau_rgb, modelu_rgb, meta: dict):
    tmpdir = Path(tempfile.mkdtemp(prefix="tisr_out_"))
    out_zip = tmpdir / "outputs.zip"

    from PIL import Image
    Image.fromarray(sr_rgb).save(tmpdir / "SR.png")
    Image.fromarray(conf_rgb).save(tmpdir / "Confidence.png")
    Image.fromarray(datau_rgb).save(tmpdir / "DataUncertainty.png")
    Image.fromarray(modelu_rgb).save(tmpdir / "ModelUncertainty.png")
    (tmpdir / "meta.json").write_text(json.dumps(meta, indent=2))

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in ["SR.png","Confidence.png","DataUncertainty.png","ModelUncertainty.png","meta.json"]:
            z.write(tmpdir / fn, arcname=fn)

    return str(out_zip)


# ---------------------------
# Gradio functions
# ---------------------------
def infer_from_upload(upload_file, ckpt_name, ensembles, epsilon, frame_index, start_index):
    if upload_file is None:
        raise gr.Error("Please upload an LR timelapse file (.tif/.zip/.npy).")

    arr = read_timelapse_from_upload(upload_file.name)  # (T,H,W)
    lr7 = ensure_7_frames(arr, start=int(start_index))  # (7,H,W)

    sr, conf, datau, modelu, dt = run_mc_inference_lr(
        lr7, ckpt_name=ckpt_name, ensembles=int(ensembles), epsilon=float(epsilon), frame_index=int(frame_index)
    )

    # For UI: SR as grayscale, others as color maps
    sr01 = robust_norm01(sr)
    conf01 = robust_norm01(conf)
    datau01 = robust_norm01(datau)
    modelu01 = robust_norm01(modelu)

    sr_rgb = np.stack([to_uint8_gray(sr01)]*3, axis=-1)
    conf_rgb = apply_cmap(conf01, "viridis")
    datau_rgb = apply_cmap(datau01, "magma")
    modelu_rgb = apply_cmap(modelu01, "magma")

    meta = {
        "checkpoint": ckpt_name,
        "ensembles": int(ensembles),
        "epsilon": float(epsilon),
        "frame_index": int(frame_index),
        "start_index": int(start_index),
        "input_shape_T_H_W": [int(x) for x in arr.shape],
        "used_window_shape": [7, int(lr7.shape[1]), int(lr7.shape[2])],
        "runtime_sec": float(dt),
        "device": get_device(),
        "note": "No PSNR/SSIM computed because HR ground truth not provided in upload."
    }
    out_zip = pack_zip(sr_rgb, conf_rgb, datau_rgb, modelu_rgb, meta)

    md = (
        f"**Done.** Runtime: `{dt:.2f}s` on `{get_device()}`  \n"
        f"Checkpoint: `{ckpt_name}`  \n"
        f"Input: `T={arr.shape[0]}, H={arr.shape[1]}, W={arr.shape[2]}` (using 7 frames starting at {start_index})"
    )

    return sr_rgb, conf_rgb, datau_rgb, modelu_rgb, md, out_zip


def infer_example_from_h5(sample_index, ckpt_name, ensembles, epsilon, frame_index):
    h5_path, keys = ensure_test_h5()
    sample_index = int(np.clip(int(sample_index), 0, len(keys) - 1))
    key = keys[sample_index]

    with h5py.File(h5_path, "r") as f:
        lr = f["lr"][key][()]   # (7,1,h,w)
        hr = f["hr"][key][()]   # (7,1,H,W)

    lr7 = lr[:, 0].astype(np.float32)  # (7,h,w)
    hr_mid = hr[int(frame_index), 0].astype(np.float32)

    sr, conf, datau, modelu, dt = run_mc_inference_lr(
        lr7, ckpt_name=ckpt_name, ensembles=int(ensembles), epsilon=float(epsilon), frame_index=int(frame_index)
    )

    # metrics possible here
    def psnr(a, b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        mse = float(np.mean((a-b)**2))
        return 99.0 if mse <= 1e-12 else float(-10*np.log10(mse))

    ps = psnr(robust_norm01(sr), robust_norm01(hr_mid))

    sr01 = robust_norm01(sr)
    conf01 = robust_norm01(conf)
    datau01 = robust_norm01(datau)
    modelu01 = robust_norm01(modelu)

    sr_rgb = np.stack([to_uint8_gray(sr01)]*3, axis=-1)
    conf_rgb = apply_cmap(conf01, "viridis")
    datau_rgb = apply_cmap(datau01, "magma")
    modelu_rgb = apply_cmap(modelu01, "magma")

    meta = {
        "checkpoint": ckpt_name,
        "sample_key": key,
        "sample_index": sample_index,
        "ensembles": int(ensembles),
        "epsilon": float(epsilon),
        "frame_index": int(frame_index),
        "runtime_sec": float(dt),
        "device": get_device(),
        "psnr_robust01": float(ps),
        "dataset_repo": HF_DATASET_REPO,
        "h5": HF_DATASET_FILENAME
    }
    out_zip = pack_zip(sr_rgb, conf_rgb, datau_rgb, modelu_rgb, meta)

    md = (
        f"**Example from H5** key=`{key}` | runtime `{dt:.2f}s` on `{get_device()}`  \n"
        f"Checkpoint `{ckpt_name}` | approx PSNR (robust-normed) `{ps:.2f} dB`"
    )

    return sr_rgb, conf_rgb, datau_rgb, modelu_rgb, md, out_zip


# ---------------------------
# UI
# ---------------------------
def build_ui():
    ckpts = list_checkpoints(HF_MODEL_REPO)
    default_ckpt = DEFAULT_CKPT if DEFAULT_CKPT in ckpts else ckpts[0]

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Upload LR Time-Lapse → Bayesian DPA-TISR Super-Resolution\n"
            "Upload a 7-frame LR time-lapse and get SR + uncertainty maps using your fine-tuned checkpoint."
        )

        with gr.Tab("Upload LR (your own)"):
            upload = gr.File(label="Upload LR time-lapse (.tif/.tiff multipage, .zip frames, or .npy)")
            with gr.Row():
                ckpt = gr.Dropdown(choices=ckpts, value=default_ckpt, label="Checkpoint")
                ensembles = gr.Slider(1, 10, value=DEFAULT_ENSEMBLES, step=1, label="MC Dropout ensembles")
                eps = gr.Number(value=DEFAULT_EPSILON, label="epsilon")

            with gr.Row():
                frame = gr.Slider(0, 6, value=DEFAULT_FRAME_INDEX, step=1, label="Frame index (0..6)")
                start = gr.Slider(0, 200, value=0, step=1, label="Start index (only used if upload has >7 frames)")

            run = gr.Button("Run on Upload", variant="primary")

            with gr.Row():
                out_sr = gr.Image(label="SR (RGB for display)", type="numpy")
                out_conf = gr.Image(label="Confidence", type="numpy")
                out_datau = gr.Image(label="Data Uncertainty", type="numpy")
                out_modelu = gr.Image(label="Model Uncertainty", type="numpy")

            md = gr.Markdown()
            zf = gr.File(label="Download ZIP (PNG + meta.json)")

            run.click(
                infer_from_upload,
                inputs=[upload, ckpt, ensembles, eps, frame, start],
                outputs=[out_sr, out_conf, out_datau, out_modelu, md, zf]
            )

        with gr.Tab("Example (from new_test.h5)"):
            gr.Markdown("This uses your hosted `new_test.h5` for a quick demo without uploading.")
            with gr.Row():
                sample = gr.Slider(0, 9999, value=0, step=1, label="Sample index (auto-clipped)")
                ckpt2 = gr.Dropdown(choices=ckpts, value=default_ckpt, label="Checkpoint")
                ensembles2 = gr.Slider(1, 10, value=DEFAULT_ENSEMBLES, step=1, label="MC Dropout ensembles")
            with gr.Row():
                eps2 = gr.Number(value=DEFAULT_EPSILON, label="epsilon")
                frame2 = gr.Slider(0, 6, value=DEFAULT_FRAME_INDEX, step=1, label="Frame index (0..6)")
                run2 = gr.Button("Run Example", variant="secondary")

            with gr.Row():
                out_sr2 = gr.Image(label="SR", type="numpy")
                out_conf2 = gr.Image(label="Confidence", type="numpy")
                out_datau2 = gr.Image(label="Data Uncertainty", type="numpy")
                out_modelu2 = gr.Image(label="Model Uncertainty", type="numpy")

            md2 = gr.Markdown()
            zf2 = gr.File(label="Download ZIP")

            run2.click(
                infer_example_from_h5,
                inputs=[sample, ckpt2, ensembles2, eps2, frame2],
                outputs=[out_sr2, out_conf2, out_datau2, out_modelu2, md2, zf2]
            )

        gr.Markdown(
            "### Space setup\n"
            "- Set `HF_DATASET_REPO` in Space Settings → Variables (dataset repo containing `new_test.h5`).\n"
            "- For best speed use a GPU Space.\n"
        )

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()
