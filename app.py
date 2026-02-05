#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=============================================================================
PRODUCT:    MCTS Image Optimizer (MIO) - Streamlit Web Demo v0.1 (FIXED)
AUTHOR:     Hideyoshi Murakami 
DATE:       2026-02-02

Goal:
  Optimize image compression not by fixed presets, but by:
    - Search (MCTS)
    - Evaluation (size × SSIM × time)
    - Heuristic constraints
  and expose it as a Streamlit web demo suitable for Render deployment.

UI Philosophy (similar feel to your previous PDF demo):
  - Sidebar: main tuning parameters (budget, target SSIM, timeout, weights)
  - Main: upload -> run -> results (before/after + metrics + download)
  - Batch: multiple images / ZIP input -> ZIP output

Dependencies:
  pip install streamlit pillow numpy
  (Optional) pip install pillow-avif-plugin  # if you want AVIF support

Key fixes (from review/user feedback):
  1) Remove double-counting of root visits (avoid UCT bias)
  2) Use the correct Pillow constant for PNG palette conversion
  3) Block "compressed output is larger than original" by default
  4) Apply per-action timeout across preprocess/encode/decode/ssim
  5) Exclude GIF animation in v0.1 for safety
  6) Normalize score by relative reduction ratio to reduce size-bias
  7) Use tiled SSIM (2x2) approximation to be slightly more sensitive to local artifacts

Disclaimer:
  - This SSIM is a lightweight approximation for search.
    For strict quality assessment, consider scikit-image SSIM or VMAF.

UI Notice:
  - X (Twitter): @nagisa7654321 is shown in the UI footer and disclaimer section.
=============================================================================
"""

from __future__ import annotations

import io
import math
import os
import random
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

import streamlit as st

# --- Optional AVIF support: if pillow-avif-plugin is installed, Pillow may handle AVIF
try:
    import pillow_avif  # noqa: F401
    _AVIF_PLUGIN_OK = True
except Exception:
    _AVIF_PLUGIN_OK = False


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class ImageProfile:
    """Lightweight profile to reduce search space and set constraints."""
    width: int
    height: int
    has_alpha: bool
    is_text_like: bool
    edge_density: float
    colorfulness: float
    estimated_colors: int


@dataclass(frozen=True)
class Action:
    """
    One compression pipeline action.

    codec: "webp" / "jpeg" / "png" / "avif"
    quality: 1-100 (ignored by PNG)
    resize_scale: 1.0 / 0.90 / 0.80 / ...
    grayscale: True/False
    denoise: True/False
    sharpen: True/False
    subsampling: JPEG only (0=4:4:4, 1=4:2:2, 2=4:2:0)
    strip_metadata: True/False
    """
    codec: str
    quality: int
    resize_scale: float
    grayscale: bool
    denoise: bool
    sharpen: bool
    subsampling: int
    strip_metadata: bool

    def key(self) -> Tuple:
        return (self.codec, self.quality, self.resize_scale, self.grayscale,
                self.denoise, self.sharpen, self.subsampling, self.strip_metadata)


@dataclass
class EvalResult:
    ok: bool
    size_bytes: int
    ssim: float
    time_ms: float
    score: float
    out_format: str
    out_bytes: bytes
    reason: str = ""


# ============================================================
# SSIM (lightweight approximation for search)
# ============================================================

def _to_luma_np(img: Image.Image, max_side: int = 512) -> np.ndarray:
    """
    Convert to a resized Luma(Y) float32 ndarray in [0,1].
    This is intentionally fast for search-time evaluation.
    """
    im = img.convert("RGB")
    w, h = im.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)

    arr = np.asarray(im).astype(np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y


def _ssim_global_from_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """
    Global SSIM approximation (not windowed).
    Works as a coarse search signal.
    """
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    if h <= 1 or w <= 1:
        return 0.0
    a = a[:h, :w]
    b = b[:h, :w]

    mu_a = float(a.mean())
    mu_b = float(b.mean())
    var_a = float(a.var())
    var_b = float(b.var())
    cov = float(((a - mu_a) * (b - mu_b)).mean())

    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)

    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    if den <= 0:
        return 0.0
    val = num / den
    return float(max(-1.0, min(1.0, val)))


def ssim_fast(img_a: Image.Image, img_b: Image.Image, max_side: int = 512, tiles: int = 4,
              reduce_mode: str = "avg") -> float:
    """
    Search-time SSIM approximation:
      - Resize to max_side
      - tiles=1 => global SSIM
      - tiles=4 => 2x2 tiled SSIM => avg/min

    reduce_mode:
      - "avg": average (good for photos)
      - "min": minimum tile (more conservative for text/line-art)
    """
    a = _to_luma_np(img_a, max_side=max_side)
    b = _to_luma_np(img_b, max_side=max_side)

    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]

    if tiles <= 1:
        return _ssim_global_from_arrays(a, b)

    hs = [0, h // 2, h]
    ws = [0, w // 2, w]
    vals = []
    for i in range(2):
        for j in range(2):
            aa = a[hs[i]:hs[i+1], ws[j]:ws[j+1]]
            bb = b[hs[i]:hs[i+1], ws[j]:ws[j+1]]
            vals.append(_ssim_global_from_arrays(aa, bb))

    if not vals:
        return 0.0
    if reduce_mode == "min":
        return float(min(vals))
    return float(sum(vals) / len(vals))


# ============================================================
# Profile estimation (cheap heuristics)
# ============================================================

def estimate_profile(img: Image.Image) -> ImageProfile:
    """
    Estimate image characteristics at low cost:
      - edge density
      - colorfulness
      - rough color count
    Then decide if it's "text-like" (screenshots, scans, line art).
    """
    w, h = img.size
    has_alpha = ("A" in img.getbands())

    thumb = img.convert("RGB")
    max_side = 256
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        thumb = thumb.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    arr = np.asarray(thumb).astype(np.float32) / 255.0

    rg = arr[..., 0] - arr[..., 1]
    yb = 0.5 * (arr[..., 0] + arr[..., 1]) - arr[..., 2]
    colorfulness = float((rg.var() + yb.var()) ** 0.5)

    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gx = abs(gray[:, 1:] - gray[:, :-1])
    gy = abs(gray[1:, :] - gray[:-1, :])
    edge_thr = 0.08
    edge_density = float((gx > edge_thr).mean() * 0.5 + (gy > edge_thr).mean() * 0.5)

    mini = thumb.resize((64, 64), Image.BILINEAR)
    m = np.asarray(mini)
    q = (m // 32).astype("uint8")
    uniq = np.unique(q.reshape(-1, 3), axis=0)
    estimated_colors = int(uniq.shape[0])

    is_text_like = (edge_density > 0.10 and colorfulness < 0.18 and estimated_colors < 140)

    return ImageProfile(
        width=w, height=h, has_alpha=has_alpha,
        is_text_like=is_text_like,
        edge_density=edge_density,
        colorfulness=colorfulness,
        estimated_colors=estimated_colors
    )


# ============================================================
# Transform pipeline ("structural intervention")
# ============================================================

def alpha_composite_on_white(img: Image.Image) -> Image.Image:
    """Composite alpha images onto white background for RGB-only codecs."""
    if "A" not in img.getbands():
        return img.convert("RGB")
    rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def preprocess_image(img: Image.Image, action: Action, profile: ImageProfile) -> Image.Image:
    """Apply action-defined preprocessing (resize/grayscale/denoise/sharpen)."""
    im = img

    if action.resize_scale < 0.999:
        w, h = im.size
        nw = max(1, int(w * action.resize_scale))
        nh = max(1, int(h * action.resize_scale))
        resample = Image.LANCZOS if profile.is_text_like else Image.BICUBIC
        im = im.resize((nw, nh), resample=resample)

    if action.grayscale:
        if profile.has_alpha:
            im = alpha_composite_on_white(im)
        im = im.convert("L")

    if action.denoise:
        im = im.filter(ImageFilter.MedianFilter(size=3))

    if action.sharpen:
        im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))

    return im


# ============================================================
# Encode / Decode
# ============================================================

def _strip_exif_safe(save_kwargs: dict) -> None:
    save_kwargs["exif"] = b""


def encode_image(img: Image.Image, action: Action, profile: ImageProfile) -> Tuple[bytes, str]:
    """Encode to bytes with the requested codec."""
    codec = action.codec.lower()
    im = img

    if profile.has_alpha and codec in ("jpeg", "jpg"):
        im = alpha_composite_on_white(im)

    bio = io.BytesIO()
    save_kwargs = {}

    if codec == "png":
        im2 = im
        if (not profile.has_alpha) and profile.estimated_colors < 192 and (im2.mode in ("RGB", "RGBA")):
            try:
                im2 = im2.convert("P", palette=Image.ADAPTIVE, colors=min(256, max(16, profile.estimated_colors)))
            except Exception:
                im2 = im
        im2.save(bio, format="PNG", optimize=True)
        return bio.getvalue(), "PNG"

    if codec in ("jpeg", "jpg"):
        if action.strip_metadata:
            try:
                _strip_exif_safe(save_kwargs)
            except Exception:
                pass
        im2 = im.convert("RGB")
        im2.save(
            bio, format="JPEG",
            quality=int(action.quality),
            optimize=True,
            progressive=True,
            subsampling=int(action.subsampling),
            **save_kwargs
        )
        return bio.getvalue(), "JPEG"
    if codec == "webp_lossless":
        if action.strip_metadata:
            try:
                _strip_exif_safe(save_kwargs)
            except Exception:
                pass
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        im.save(bio, format="WEBP", lossless=True, method=6, **save_kwargs)
        return bio.getvalue(), "WEBP"


    if codec == "webp":
        if action.strip_metadata:
            try:
                _strip_exif_safe(save_kwargs)
            except Exception:
                pass
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        im.save(bio, format="WEBP", quality=int(action.quality), method=6, **save_kwargs)
        return bio.getvalue(), "WEBP"

    if codec == "avif":
        if action.strip_metadata:
            try:
                _strip_exif_safe(save_kwargs)
            except Exception:
                pass
        if im.mode not in ("RGB", "RGBA", "L"):
            im = im.convert("RGB")
        im.save(bio, format="AVIF", quality=int(action.quality), **save_kwargs)
        return bio.getvalue(), "AVIF"

    raise ValueError(f"Unsupported codec: {codec}")


def decode_bytes_to_image(b: bytes) -> Image.Image:
    bio = io.BytesIO(b)
    im = Image.open(bio)
    im.load()
    return im


# ============================================================
# Scoring (size × SSIM × time + constraints)
# ============================================================

@dataclass
class ScoringConfig:
    size_weight: float = 1.0
    ssim_weight: float = 3.0
    time_weight: float = 0.002

    target_ssim: float = 0.97
    ssim_max_side: int = 512
    ssim_tiles: int = 4

    per_action_timeout_s: float = 1.2

    allow_size_increase_pct: float = 0.0


def _timeout_exceeded(t0: float, cfg: ScoringConfig) -> bool:
    return (time.perf_counter() - t0) > cfg.per_action_timeout_s


def evaluate_action(original: Image.Image, original_bytes_len: int, profile: ImageProfile,
                    action: Action, cfg: ScoringConfig) -> EvalResult:
    t0 = time.perf_counter()

    try:
        im_proc = preprocess_image(original, action, profile)
    except Exception as e:
        return EvalResult(False, 10**18, 0.0, 0.0, -10**18, "", b"", reason=f"preprocess failed: {e}")
    if _timeout_exceeded(t0, cfg):
        return EvalResult(False, 10**18, 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, "", b"",
                          reason="timeout in preprocess")

    try:
        out_bytes, out_fmt = encode_image(im_proc, action, profile)
    except Exception as e:
        return EvalResult(False, 10**18, 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, "", b"",
                          reason=f"encode failed: {e}")
    if _timeout_exceeded(t0, cfg):
        return EvalResult(False, len(out_bytes), 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, out_fmt, out_bytes,
                          reason="timeout in encode")

    out_size = len(out_bytes)

    allow = cfg.allow_size_increase_pct / 100.0
    if original_bytes_len > 0 and out_size > int(original_bytes_len * (1.0 + allow)):
        return EvalResult(False, out_size, 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, out_fmt, out_bytes,
                          reason="output larger than original (blocked)")

    try:
        decoded = decode_bytes_to_image(out_bytes)
    except Exception as e:
        return EvalResult(False, out_size, 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, out_fmt, out_bytes,
                          reason=f"decode failed: {e}")
    if _timeout_exceeded(t0, cfg):
        return EvalResult(False, out_size, 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, out_fmt, out_bytes,
                          reason="timeout in decode")

    try:
        reduce_mode = "min" if profile.is_text_like else "avg"
        s = ssim_fast(original, decoded, max_side=cfg.ssim_max_side, tiles=cfg.ssim_tiles, reduce_mode=reduce_mode)
    except Exception as e:
        return EvalResult(False, out_size, 0.0, (time.perf_counter() - t0) * 1000.0, -10**18, out_fmt, out_bytes,
                          reason=f"ssim failed: {e}")
    if _timeout_exceeded(t0, cfg):
        return EvalResult(False, out_size, s, (time.perf_counter() - t0) * 1000.0, -10**18, out_fmt, out_bytes,
                          reason="timeout in ssim")

    t_ms = (time.perf_counter() - t0) * 1000.0

    target = cfg.target_ssim
    if profile.is_text_like:
        target = max(target, 0.975)
        if action.codec == "jpeg" and action.subsampling >= 2 and action.quality < 55:
            return EvalResult(False, out_size, s, t_ms, -10**18, out_fmt, out_bytes,
                              reason="text-like: too aggressive jpeg subsampling+quality")
        if action.resize_scale < 0.70:
            return EvalResult(False, out_size, s, t_ms, -10**18, out_fmt, out_bytes,
                              reason="text-like: resize too much")

    if s < target:
        return EvalResult(False, out_size, s, t_ms, -10**18, out_fmt, out_bytes,
                          reason=f"ssim below target ({s:.4f} < {target:.4f})")

    if original_bytes_len <= 0:
        return EvalResult(False, out_size, s, t_ms, -10**18, out_fmt, out_bytes, reason="invalid original size")

    reduction_ratio = (original_bytes_len - out_size) / float(original_bytes_len)
    ssim_margin = (s - target)

    score = (
        cfg.size_weight * (reduction_ratio * 1000.0)
        + cfg.ssim_weight * (ssim_margin * 1000.0)
        - cfg.time_weight * t_ms
    )
    return EvalResult(True, out_size, s, t_ms, float(score), out_fmt, out_bytes, reason="ok")


# ============================================================
# MCTS (2-level tree: codec -> action)
# ============================================================

class MCTSNode:
    def __init__(self, parent: Optional["MCTSNode"], name: str, payload=None):
        self.parent = parent
        self.name = name
        self.payload = payload
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value_sum = 0.0

    @property
    def value_mean(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def add_child(self, child: "MCTSNode"):
        self.children.append(child)

    def uct_score(self, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return self.value_mean + exploration * math.sqrt(math.log(self.parent.visits + 1) / self.visits)


def build_action_space(profile: ImageProfile, allow_grayscale: bool) -> Dict[str, List[Action]]:
    codecs: List[str] = ["webp", "webp_lossless", "jpeg", "png"]
    if _AVIF_PLUGIN_OK:
        codecs.append("avif")

    if profile.is_text_like:
        qualities = [55, 65, 75, 85, 92]
        scales = [1.0, 0.90, 0.80, 0.70]
        allow_gray = [False, True] if allow_grayscale else [False]
        allow_denoise = [False, True]
        allow_sharp = [False, True]
    else:
        qualities = [45, 55, 65, 75, 85, 92]
        scales = [1.0, 0.90, 0.80]
        allow_gray = [False, True] if allow_grayscale else [False]
        allow_denoise = [False, True]
        allow_sharp = [False]

    subs_list = [0, 1] if profile.is_text_like else [2, 1]

    actions_by_codec: Dict[str, List[Action]] = {c: [] for c in codecs}

    for scale in scales:
        for gray in allow_gray:
            for dn in allow_denoise:
                for sh in allow_sharp:
                    actions_by_codec["png"].append(Action(
                        codec="png", quality=100, resize_scale=scale, grayscale=gray,
                        denoise=dn, sharpen=sh, subsampling=0, strip_metadata=True
                    ))

    for q in qualities:
        for scale in scales:
            for gray in allow_gray:
                for dn in allow_denoise:
                    for sh in allow_sharp:
                        actions_by_codec["webp"].append(Action(
                            codec="webp", quality=q, resize_scale=scale, grayscale=gray,
                            denoise=dn, sharpen=sh, subsampling=0, strip_metadata=True
                        ))
    # WebP lossless (small search space, screenshot-friendly)
    for scale in [1.0]:
        for gray in allow_gray:
            for dn in allow_denoise:
                for sh in allow_sharp:
                    actions_by_codec["webp_lossless"].append(Action(
                        codec="webp_lossless", quality=100, resize_scale=scale, grayscale=gray,
                        denoise=dn, sharpen=sh, subsampling=0, strip_metadata=True
                    ))



    for q in qualities:
        for scale in scales:
            for dn in allow_denoise:
                for sh in allow_sharp:
                    for subs in subs_list:
                        actions_by_codec["jpeg"].append(Action(
                            codec="jpeg", quality=q, resize_scale=scale, grayscale=False,
                            denoise=dn, sharpen=sh, subsampling=subs, strip_metadata=True
                        ))

    if "avif" in actions_by_codec:
        avif_qualities = [45, 55, 65, 75, 85] if not profile.is_text_like else [60, 70, 80, 90]
        avif_scales = scales if profile.is_text_like else [1.0, 0.90]
        for q in avif_qualities:
            for scale in avif_scales:
                actions_by_codec["avif"].append(Action(
                    codec="avif", quality=q, resize_scale=scale, grayscale=False,
                    denoise=False, sharpen=False, subsampling=0, strip_metadata=True
                ))

    if profile.has_alpha and "jpeg" in actions_by_codec:
        actions_by_codec["jpeg"] = actions_by_codec["jpeg"][: max(10, len(actions_by_codec["jpeg"]) // 4)]

    return actions_by_codec


def pick_codec_priors(profile: ImageProfile) -> Dict[str, float]:
    priors = {"webp": 1.0, "webp_lossless": 0.9, "jpeg": 1.0, "png": 1.0}
    if _AVIF_PLUGIN_OK:
        priors["avif"] = 0.8

    if profile.is_text_like:
        priors["webp"] *= 1.15
        priors["webp_lossless"] *= 1.25
        priors["png"] *= 1.1
        priors["jpeg"] *= 0.8
        if "avif" in priors:
            priors["avif"] *= 1.0
    else:
        priors["jpeg"] *= 1.1
        priors["png"] *= 0.7
        if "avif" in priors:
            priors["avif"] *= 1.1

    s = sum(priors.values())
    return {k: v / s for k, v in priors.items()}


def select_uct_child(node: MCTSNode, exploration: float) -> MCTSNode:
    unvisited = [c for c in node.children if c.visits == 0]
    if unvisited:
        return random.choice(unvisited)
    best, best_score = None, -1e30
    for c in node.children:
        sc = c.uct_score(exploration)
        if sc > best_score:
            best_score, best = sc, c
    assert best is not None
    return best


def backprop(node: MCTSNode, reward: float):
    cur = node
    while cur is not None:
        cur.visits += 1
        cur.value_sum += reward
        cur = cur.parent


def select_codec_node(root: MCTSNode, exploration: float, priors: Dict[str, float]) -> MCTSNode:
    unvisited = [c for c in root.children if c.visits == 0]
    if unvisited:
        weights = [priors.get(n.payload, 1.0) for n in unvisited]
        return random.choices(unvisited, weights=weights, k=1)[0]
    best, best_score = None, -1e30
    for n in root.children:
        prior = priors.get(n.payload, 1e-6)
        bonus = math.log(prior + 1e-9) * 0.05
        score = n.uct_score(exploration) + bonus
        if score > best_score:
            best_score, best = score, n
    assert best is not None
    return best


def mcts_optimize(original: Image.Image, original_bytes: bytes, original_bytes_len: int, profile: ImageProfile,
                  cfg: ScoringConfig, budget: int = 120, exploration: float = 1.2, seed: int = 0, allow_grayscale: bool = False) -> Tuple[EvalResult, Dict]:
    random.seed(seed)

    actions_by_codec = build_action_space(profile, allow_grayscale=allow_grayscale)
    priors = pick_codec_priors(profile)

    root = MCTSNode(None, "root", payload=None)
    for codec, acts in actions_by_codec.items():
        if not acts:
            continue
        cn = MCTSNode(root, f"codec:{codec}", payload=codec)
        root.add_child(cn)
        for a in acts:
            cn.add_child(MCTSNode(cn, "action", payload=a))

    cache: Dict[Tuple, EvalResult] = {}
    best: Optional[EvalResult] = None
    best_action: Optional[Action] = None
    logs = []
    t_start = time.perf_counter()

    for it in range(budget):
        codec_node = select_codec_node(root, exploration, priors)
        action_node = select_uct_child(codec_node, exploration)
        action: Action = action_node.payload

        if action.key() in cache:
            res = cache[action.key()]
        else:
            res = evaluate_action(original, original_bytes_len, profile, action, cfg)
            cache[action.key()] = res

        reward = res.score if res.ok else -10**12
        backprop(action_node, reward)

        if res.ok and (best is None or res.score > best.score):
            best, best_action = res, action

        if (it % max(1, budget // 20)) == 0:
            logs.append({
                "iter": it,
                "codec": action.codec,
                "q": action.quality,
                "scale": action.resize_scale,
                "gray": action.grayscale,
                "dn": action.denoise,
                "sh": action.sharpen,
                "sub": action.subsampling,
                "ok": res.ok,
                "size": res.size_bytes,
                "ssim": res.ssim,
                "score": res.score,
                "reason": res.reason
            })

        if (time.perf_counter() - t_start) > max(2.5, cfg.per_action_timeout_s * max(30, budget) * 0.10):
            break

    if best is None:
        # Fallback strategy:
        # 1) Try a safe PNG optimize pass.
        # 2) If it still fails (often because the original is already highly optimized or constraints are strict),
        #    return the original bytes as a VALID result instead of raising an error.
        fallback = Action(codec="png", quality=100, resize_scale=1.0, grayscale=False,
                          denoise=False, sharpen=False, subsampling=0, strip_metadata=True)
        res = evaluate_action(original, original_bytes_len, profile, fallback, cfg)
        debug = {"priors": priors, "logs": logs, "best_action": None, "cache_size": len(cache)}
        if res.ok:
            return res, debug

        # No valid candidate met constraints (size/SSIM/timeout). Return original.
        return EvalResult(True, original_bytes_len, 1.0, 0.0, 0.0, "ORIG", original_bytes,
                          reason="no valid smaller candidate; returned original"), debug

    debug = {"priors": priors, "logs": logs, "best_action": best_action, "cache_size": len(cache)}
    return best, debug


# ============================================================
# UI Helpers
# ============================================================

def load_image_from_bytes(b: bytes) -> Image.Image:
    bio = io.BytesIO(b)
    im = Image.open(bio)
    im.load()
    return im


def choose_output_ext(fmt: str) -> str:
    fmt = (fmt or "").upper()
    return { "JPEG": ".jpg", "WEBP": ".webp", "PNG": ".png", "AVIF": ".avif" }.get(fmt, ".bin")


def safe_filename(name: str) -> str:
    name = os.path.basename(name)
    return name.replace("\\", "_").replace("/", "_")


def make_zip(results: List[Tuple[str, bytes]]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn, data in results:
            zf.writestr(fn, data)
    return bio.getvalue()


# ============================================================
# Streamlit UI
# ============================================================

X_HANDLE = "@nagisa7654321"

DISCLAIMER_TEXT = f"""
**Disclaimer (No Warranty / Use at Your Own Risk)**  
This demo is provided for experimental and educational purposes only.  
- Outputs may be larger than the original depending on constraints and environment.  
- The SSIM used here is a lightweight approximation intended for search-time guidance.  
- Do not rely on this tool for medical, legal, safety-critical, or archival workflows.  
- Always keep backups of your original files.  

**Author / Contact**  
X (Twitter): {X_HANDLE}
"""


def render_header():
    st.markdown("""
    <style>
      .footer {font-size: 0.90rem; opacity: 0.85; padding-top: 0.6rem;}
    </style>
    """, unsafe_allow_html=True)
    st.title("MCTS Image Optimizer (MIO) — Search-based Image Compression Demo")
    st.warning("Disclaimer: This is an experimental demo provided AS-IS with no warranty. Use at your own risk. Do not upload sensitive/private data. If an input is already well-optimized, reduction may be 0% (the original file is returned). Author: @nagisa7654321", icon="⚠️")
    st.write("**MCTS × Heuristics × Visual Quality (SSIM) × Structural Intervention** — searches an optimal compression pipeline per image.")
    st.caption("Streamlit demo for Render deployment • Single & Batch/ZIP • SSIM is downscaled and tiled for speed")


def sidebar_controls() -> Tuple[ScoringConfig, int, int, bool, bool, bool]:
    st.sidebar.header("Optimization Parameters (Search)")

    budget = st.sidebar.slider("Search iterations (budget)", 20, 400, 120, 10,
                               help="Higher => better but slower. Free tiers: 60–160 is realistic.")
    target_ssim = st.sidebar.slider("Quality floor (target SSIM)", 0.900, 0.995, 0.970, 0.001,
                                    help="Text-like: 0.975–0.99. Photos: 0.95–0.97.")
    ssim_max_side = st.sidebar.select_slider("SSIM downscale (max side)", options=[256, 384, 512, 768], value=512)
    ssim_tiles = st.sidebar.select_slider("SSIM tiling", options=[1, 4], value=4)
    per_action_timeout = st.sidebar.slider("Per-candidate timeout (seconds)", 0.3, 3.0, 1.2, 0.1)

    st.sidebar.divider()
    st.sidebar.subheader("Score Weights (Advanced)")
    size_w = st.sidebar.slider("Size weight", 0.2, 3.0, 1.0, 0.1)
    ssim_w = st.sidebar.slider("SSIM weight", 0.5, 8.0, 3.0, 0.1)
    time_w = st.sidebar.slider("Time penalty", 0.000, 0.010, 0.002, 0.001)

    st.sidebar.divider()
    st.sidebar.subheader("Safety")
    allow_inc = st.sidebar.slider("Allow size increase (%)", 0.0, 2.0, 0.0, 0.1)

    seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=0, step=1)

    show_debug = st.sidebar.checkbox("Show debug logs", value=False)
    show_baseline = st.sidebar.checkbox("Show baseline comparison (fixed preset)", value=True)
    allow_grayscale = st.sidebar.checkbox("Allow grayscale candidates (higher compression / may remove color)", value=False)

    st.sidebar.divider()
    with st.sidebar.expander("Disclaimer / Author", expanded=False):
        st.markdown(DISCLAIMER_TEXT)

    cfg = ScoringConfig(
        size_weight=float(size_w),
        ssim_weight=float(ssim_w),
        time_weight=float(time_w),
        target_ssim=float(target_ssim),
        ssim_max_side=int(ssim_max_side),
        ssim_tiles=int(ssim_tiles),
        per_action_timeout_s=float(per_action_timeout),
        allow_size_increase_pct=float(allow_inc),
    )
    return cfg, int(budget), int(seed), bool(show_debug), bool(show_baseline), bool(allow_grayscale)


def show_profile(profile: ImageProfile):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", f"{profile.width}×{profile.height}")
    c2.metric("Alpha", "Yes" if profile.has_alpha else "No")
    c3.metric("Text-like", "Yes" if profile.is_text_like else "No")
    c4.metric("Est. colors", f"{profile.estimated_colors}")
    st.caption(f"edge_density={profile.edge_density:.3f} / colorfulness={profile.colorfulness:.3f}")


def _baseline_action(profile: ImageProfile) -> Action:
    if profile.is_text_like:
        return Action(codec="webp", quality=85, resize_scale=1.0, grayscale=False,
                      denoise=False, sharpen=False, subsampling=0, strip_metadata=True)
    return Action(codec="jpeg", quality=75, resize_scale=1.0, grayscale=False,
                  denoise=False, sharpen=False, subsampling=2, strip_metadata=True)


def show_result(original_bytes: bytes, best: EvalResult, in_name: str):
    orig_size = len(original_bytes)
    out_size = best.size_bytes
    red_pct = ((orig_size - out_size) / orig_size * 100.0) if orig_size > 0 else 0.0
    ratio = (orig_size / out_size) if out_size > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original", f"{orig_size/1024:.1f} KB")
    c2.metric("Compressed", f"{out_size/1024:.1f} KB")
    c3.metric("Reduction", f"{red_pct:.2f}%")
    c4.metric("Ratio", f"x{ratio:.2f}")

    st.write(f"**Output:** {best.out_format} / **SSIM:** `{best.ssim:.4f}` / **Eval:** `{best.time_ms:.1f} ms` / **score:** `{best.score:.3f}`")
    if (best.out_format or "").upper() == "ORIG":
        st.info("No candidate met constraints (size/SSIM/timeout). Returning the original file. Try lowering target SSIM, increasing timeout, or allowing a small size increase.")

    ext = choose_output_ext(best.out_format)
    base = os.path.splitext(safe_filename(in_name))[0]
    out_name = f"{base}_mio{ext}"

    st.download_button("⬇️ Download compressed file", best.out_bytes, file_name=out_name, mime="application/octet-stream")


def main():
    st.set_page_config(page_title="MCTS Image Optimizer (MIO)", layout="wide")
    render_header()
    cfg, budget, seed, show_debug, show_baseline, allow_grayscale = sidebar_controls()

    tab1, tab2 = st.tabs(["Single Image", "Batch / ZIP"])

    with tab1:
        st.subheader("Single-image optimization")
        up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "avif"])
        if up is None:
            st.info("Upload an image to start.")
        else:
            in_name = up.name
            original_bytes = up.read()
            try:
                img = load_image_from_bytes(original_bytes)
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                return

            profile = estimate_profile(img)

            left, right = st.columns(2)
            with left:
                st.markdown("### Original")
                st.image(img, use_container_width=True)
            with right:
                st.markdown("### Profile")
                show_profile(profile)
                if show_baseline:
                    base_act = _baseline_action(profile)
                    base_res = evaluate_action(img, len(original_bytes), profile, base_act, cfg)
                    if base_res.ok and len(original_bytes) > 0:
                        red_pct = (len(original_bytes) - base_res.size_bytes) / len(original_bytes) * 100.0
                        st.caption(f"Baseline: {base_act.codec.upper()} q={base_act.quality} / SSIM={base_res.ssim:.4f} / Reduction={red_pct:.2f}%")
                    else:
                        st.caption(f"Baseline: invalid ({base_res.reason})")

                run = st.button("🚀 Start optimization", type="primary", use_container_width=True)

            if run:
                with st.spinner("Searching… (MCTS evaluates candidates)"):
                    t0 = time.perf_counter()
                    best, debug = mcts_optimize(img, original_bytes, len(original_bytes), profile, cfg, budget=budget, exploration=1.2, seed=seed, allow_grayscale=allow_grayscale)
                    total = time.perf_counter() - t0

                if not best.ok:
                    st.error(f"Optimization failed: {best.reason}")
                else:
                    with right:
                        st.success(f"Done (total: {total:.2f}s)")
                        show_result(original_bytes, best, in_name)
                        if show_debug:
                            st.json({"priors": debug.get("priors"), "cache_size": debug.get("cache_size"), "best_action": str(debug.get("best_action"))})
                            st.dataframe(debug.get("logs", []), use_container_width=True)

                    st.markdown("### Visual comparison (Before / After)")
                    try:
                        after_img = load_image_from_bytes(best.out_bytes)
                        cA, cB = st.columns(2)
                        cA.image(img, caption="Original", use_container_width=True)
                        cB.image(after_img, caption=f"Compressed ({best.out_format})", use_container_width=True)
                    except Exception:
                        st.warning("Preview failed (download is still available).")

    with tab2:
        st.subheader("Batch optimization (Multiple images / ZIP input → ZIP output)")
        mode = st.radio("Input mode", ["Select multiple files", "Upload a ZIP"], horizontal=True)
        files = []
        zip_bytes = None

        if mode == "Select multiple files":
            ups = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "avif"], accept_multiple_files=True)
            if ups:
                files = ups
        else:
            zup = st.file_uploader("Upload a ZIP", type=["zip"])
            if zup:
                zip_bytes = zup.read()

        run_batch = st.button("🚀 Start batch optimization", type="primary", use_container_width=True)

        if run_batch:
            items: List[Tuple[str, bytes]] = []
            if zip_bytes is not None:
                with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
                    for name in zf.namelist():
                        if name.endswith("/"):
                            continue
                        ext = os.path.splitext(name)[1].lower()
                        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".avif"]:
                            items.append((name, zf.read(name)))
            else:
                for f in files:
                    items.append((f.name, f.read()))

            if not items:
                st.warning("No valid images found to optimize.")
                return

            out_results: List[Tuple[str, bytes]] = []
            rows = []

            with st.spinner("Batch searching… (optimizing sequentially)"):
                for i, (name, b) in enumerate(items):
                    try:
                        img = load_image_from_bytes(b)
                        profile = estimate_profile(img)
                        best, _ = mcts_optimize(img, b, len(b), profile, cfg, budget=budget, exploration=1.2, seed=seed + i, allow_grayscale=allow_grayscale)
                        if best.ok:
                            ext = choose_output_ext(best.out_format)
                            base = os.path.splitext(name)[0]
                            out_name = f"{base}_mio{ext}"
                            out_results.append((out_name, best.out_bytes))
                            red_pct = ((len(b) - best.size_bytes) / len(b) * 100.0) if len(b) > 0 else 0.0
                            rows.append({"file": name, "out": out_name, "text_like": profile.is_text_like,
                                         "orig_kb": round(len(b)/1024, 1), "out_kb": round(best.size_bytes/1024, 1),
                                         "reduction_%": round(red_pct, 2), "ssim": round(best.ssim, 4), "fmt": best.out_format})
                        else:
                            out_results.append((safe_filename(name), b))
                            rows.append({"file": name, "out": name, "text_like": profile.is_text_like,
                                         "orig_kb": round(len(b)/1024, 1), "out_kb": round(len(b)/1024, 1),
                                         "reduction_%": 0.0, "ssim": 1.0, "fmt": "ORIG", "reason": best.reason})
                    except Exception as e:
                        out_results.append((safe_filename(name), b))
                        rows.append({"file": name, "out": name, "text_like": False,
                                     "orig_kb": round(len(b)/1024, 1), "out_kb": round(len(b)/1024, 1),
                                     "reduction_%": 0.0, "ssim": 1.0, "fmt": "ORIG", "error": str(e)})

            zip_out = make_zip(out_results)
            st.success("Batch optimization complete.")
            st.dataframe(rows, use_container_width=True)
            st.download_button("⬇️ Download ZIP results", zip_out, file_name="mio_results.zip", mime="application/zip")

    st.divider()
    st.caption(f"Tip: For text-like images, try target SSIM 0.975–0.99.  •  X (Twitter): {X_HANDLE}")
    st.markdown(f"<div class='footer'>X (Twitter): <b>{X_HANDLE}</b> • {time.strftime('%Y-%m-%d')}</div>",
                unsafe_allow_html=True)


    st.caption("If the input is already well-optimized, the reduction may be 0% (the original file is returned).")

if __name__ == "__main__":
    main()
