"""
Multi-layer 2.5D segmentation, edge-bleed alpha padding, and lossless AAC audio
remuxing pipeline for bloom_x_hip_shop.mp4 (1920x1440).

Fixes internal hitbox/seam artifacts by keeping the character texture unified
for 3D relief-mesh deformation and bleeding edge RGB colors outward into
transparent pixels so bilinear filtering and RGB chromatic shift never sample black.
"""

from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DEBUG_DIR = BASE_DIR / "debug"
VIDEO_PATH = Path(
    r"C:/Users/ab010/.gemini/antigravity/brain/1462d919-8923-4634-ae38-b3c6cf95cff5/.user_uploaded/media_1794584484158_939b4b57.mp4"
)
EXPR_DIR = Path(
    r"C:/Users/ab010/.gemini/antigravity/brain/1462d919-8923-4634-ae38-b3c6cf95cff5/.user_uploaded"
)

# Pair each 706x1024 pixelated expression sprite with its 707x1024 pre-saturation raw original
# so we can restore the exact artist-drawn eyebrows, mouths, nose dots, blush, and tears that
# were lost during quantization, and grade colors to match image.png (char_full_backup.png).
EXPRESSION_PAIRS: dict[str, tuple[str, str]] = {
    "calm": ("media_1794683099066_23b34dd1.png", "media_1794686980860_6c483f94.png"),
    "worried": ("media_1794685763920_0507a5a4.png", "media_1794686980894_a19131c6.png"),
    "flustered": ("media_1794685431631_7efde094.png", "media_1794686980948_8f12301d.png"),
    "proud_blush": ("media_1794683099081_0ae90010.png", "media_1794686980775_c131cab1.png"),
    "sketchbook": ("media_1794683099079_740d9448.png", "media_1794686980813_d8c28ed1.png"),
    "smug": ("media_1794683099085_b99eaf85.png", "media_1794686980745_78ff2c7b.png"),
    "angry": ("media_1794683099085_b99eaf85.png", "media_1794686981058_d0dca108.png"),
    "sleepy": ("media_1794683099095_e421e1a2.png", "media_1794686980716_a65472c2.png"),
    "surprised": ("media_1794683099099_4c1344f3.png", "media_1794686980689_376457ee.png"),
    "sleeve_pout": ("media_1794683099103_5a0ee830.png", "media_1794686980661_b5c4c8b6.png"),
    "nervous_hide": ("media_1794683099110_95439680.png", "media_1794686980635_5e650f4b.png"),
    "shy_hide": ("media_1794683099113_f9923e0c.png", "media_1794686980606_cfdda549.png"),
    "wide_eyes": ("media_1794685763911_b7d7b825.png", "media_1794686981017_9130960d.png"),
    "one_sleeve": ("media_1794685763920_271bc8b0.png", "media_1794686981032_ffd06af5.png"),
}


def extract_lossless_m4a(video_path: Path, out_m4a: Path) -> None:
    """Remux the AAC audio track ('soun') from the MP4 container without re-encoding."""
    data = video_path.read_bytes()

    stsz_pos = 7088137
    _, _, _, _, sample_count = struct.unpack(">I4sIII", data[stsz_pos : stsz_pos + 20])
    sample_sizes = struct.unpack(
        f">{sample_count}I", data[stsz_pos + 20 : stsz_pos + 20 + 4 * sample_count]
    )

    stco_pos = 7097433
    _, _, _, chunk_count = struct.unpack(">I4sII", data[stco_pos : stco_pos + 16])
    old_chunks = struct.unpack(
        f">{chunk_count}I", data[stco_pos + 16 : stco_pos + 16 + 4 * chunk_count]
    )

    audio_blobs = []
    new_chunks = []
    curr_offset = 32 + 8
    sample_idx = 0
    for c_idx in range(chunk_count):
        n_s = 24 if c_idx < 96 else 15
        c_size = sum(sample_sizes[sample_idx : sample_idx + n_s])
        sample_idx += n_s
        old_off = old_chunks[c_idx]
        audio_blobs.append(data[old_off : old_off + c_size])
        new_chunks.append(curr_offset)
        curr_offset += c_size

    mdat_payload = b"".join(audio_blobs)
    mdat_box = struct.pack(">I4s", 8 + len(mdat_payload), b"mdat") + mdat_payload

    trak = bytearray(data[7087661:7097891])
    rel_stco = 7097433 - 7087661
    trak[rel_stco + 16 : rel_stco + 16 + 4 * chunk_count] = struct.pack(
        f">{chunk_count}I", *new_chunks
    )

    mvhd = data[7071809:7071917]
    moov_payload = mvhd + bytes(trak)
    moov_box = struct.pack(">I4s", 8 + len(moov_payload), b"moov") + moov_payload
    ftyp_box = data[0:32]

    out_m4a.write_bytes(ftyp_box + mdat_box + moov_box)


def bgr_to_rgba_padded(bgr: np.ndarray, alpha: np.ndarray, pad_iters: int = 5) -> np.ndarray:
    """
    Create an RGBA image where RGB colors along the alpha boundary are dilated outward
    by `pad_iters` pixels into the alpha=0 region. This prevents WebGL linear filtering
    or chromatic aberration UV offsets from ever sampling black (0,0,0) halos!
    """
    rgba = np.zeros((bgr.shape[0], bgr.shape[1], 4), dtype=np.uint8)
    mask = alpha > 0
    work_bgr = np.zeros_like(bgr)
    work_bgr[mask] = bgr[mask]

    curr_mask = mask.astype(np.uint8)
    k3 = np.ones((3, 3), dtype=np.uint8)
    for _ in range(pad_iters):
        dil_bgr = cv2.dilate(work_bgr, k3, iterations=1)
        dil_mask = cv2.dilate(curr_mask, k3, iterations=1)
        fringe = (dil_mask > 0) & (curr_mask == 0)
        work_bgr[fringe] = dil_bgr[fringe]
        curr_mask = dil_mask

    rgba[:, :, :3] = work_bgr
    rgba[:, :, 3] = alpha
    return rgba


def make_blink_layer(frame: np.ndarray, shift_px: int) -> np.ndarray:
    """Shift the actual upper eyelid skin and eyelash downwards behind the hair bangs."""
    h, w, _ = frame.shape
    out_bgr = frame.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(1012, 1091):
        top_skin = frame[370, x]
        for y in range(380, 421):
            b, g, r = [int(v) for v in frame[y, x]]
            if x < 1020 and r > 180 and g > 170:
                continue
            if x > 1084 and r > 200 and g > 185:
                continue
            src_y = y - shift_px
            if src_y >= 397:
                continue
            mask[y, x] = 255
            if src_y >= 381:
                out_bgr[y, x] = frame[src_y, x]
            else:
                out_bgr[y, x] = top_skin
    return bgr_to_rgba_padded(out_bgr, mask, pad_iters=4)


def build_pixel_char_from_uploaded_png(
    pix_rgba: np.ndarray,
    raw_rgba: np.ndarray,
    expr_key: str = "calm",
) -> np.ndarray:
    """
    Grades a user-uploaded 706x1024 pixel-art RGBA sprite to match `image.png` (char_full_backup.png):
      - Deep midnight-navy dress/bonnet (B ~ 110..148)
      - Soft periwinkle-lavender hair shadows & irises (B ~ 225..243)
      - Snowy white skin & highlights
      - Restores exact artist-drawn eyebrows, mouth, nose dot, lip shadow, blush & tears
        from the 707x1024 pre-saturation original `raw_rgba` via 2x2 min/mean pooling
      - Applies horizontal RGB chromatic shift (+3px R right, -2px B/G left)
    Scales by exact 2.0x integer factor (uniform 4x4 pixel blocks) and places at
    (ox=280, oy=96) on the 1920x1440 canvas.
    """
    h, w = pix_rgba.shape[:2]
    raw_crop = raw_rgba[:h, :w].astype(np.float32)

    bgr = pix_rgba[:, :, :3].astype(np.float32)
    alpha_src = pix_rgba[:, :, 3]
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]

    is_white = (r >= 245) & (g >= 245) & (b >= 245)
    is_warm = (r > b) & (r < 245)

    # Separate dark dress/bonnet navy (g < 95) from light hair/iris blues (g >= 95)
    hair_w = np.clip((g - 75.0) / 55.0, 0.0, 1.0)
    b_dress = 22.0 + 0.64 * b
    g_dress = 0.72 * g
    r_dress = 0.90 * r + 4.0

    b_hair = 50.0 + 0.76 * b
    g_hair = 34.0 + 0.64 * g + 0.06 * r
    r_hair = 58.0 + 0.70 * r + 0.06 * g

    b_cool = b_dress * (1.0 - hair_w) + b_hair * hair_w
    g_cool = g_dress * (1.0 - hair_w) + g_hair * hair_w
    r_cool = r_dress * (1.0 - hair_w) + r_hair * hair_w

    b_out = np.where(is_white, 251.0, np.where(is_warm, b * 0.96, b_cool))
    g_out = np.where(is_white, 249.0, np.where(is_warm, g * 0.94, g_cool))
    r_out = np.where(is_white, 254.0, np.where(is_warm, np.minimum(255.0, r * 1.02), r_cool))
    graded = np.stack([b_out, g_out, r_out], axis=-1)

    # Restore exact 2x2-pooled facial lines & blush from pre-saturation raw_rgba
    raw_2x2_min = np.repeat(
        np.repeat(raw_crop.reshape(h // 2, 2, w // 2, 2, 4).min(axis=(1, 3)), 2, axis=0),
        2,
        axis=1,
    )
    raw_2x2_mean = np.repeat(
        np.repeat(raw_crop.reshape(h // 2, 2, w // 2, 2, 4).mean(axis=(1, 3)), 2, axis=0),
        2,
        axis=1,
    )

    fy0, fy1, fx0, fx1 = 116, 206, 296, 396
    f_grad = graded[fy0:fy1, fx0:fx1].copy()
    f_min = raw_2x2_min[fy0:fy1, fx0:fx1]
    f_mean = raw_2x2_mean[fy0:fy1, fx0:fx1]
    f_pix = bgr[fy0:fy1, fx0:fx1]

    # Sleeve occlusion guard: never draw facial lines over raised dark-navy sleeves
    not_sleeve = ~((f_pix[:, :, 0] > f_pix[:, :, 2] + 35.0) & (f_pix[:, :, 1] < 135.0))

    # 1. Soft cheek/nose blush restoration (e.g. shy_hide, proud_blush, flustered)
    mean_b, mean_g, mean_r = f_mean[:, :, 0], f_mean[:, :, 1], f_mean[:, :, 2]
    pinkness = np.maximum(0.0, mean_r - np.maximum(mean_g, mean_b * 0.97))
    blush_w = np.clip((pinkness - 2.5) / 30.0, 0.0, 1.0) * not_sleeve.astype(np.float32)
    blush_target = np.stack(
        [
            np.clip(251.0 - pinkness * 0.55, 195.0, 251.0),
            np.clip(249.0 - pinkness * 1.15, 165.0, 249.0),
            np.full_like(mean_r, 254.0),
        ],
        axis=-1,
    )
    f_grad = f_grad * (1.0 - blush_w[:, :, None] * 0.85) + blush_target * (blush_w[:, :, None] * 0.85)

    # 2. Restore lost fine lines (eyebrows, nose shadow dot, mouth line, lip shadow, tears)
    min_b, min_g, min_r = f_min[:, :, 0], f_min[:, :, 1], f_min[:, :, 2]
    line_darkness = np.maximum(0.0, 236.0 - min_g)

    fh, fw = fy1 - fy0, fx1 - fx0
    brow_zone = np.zeros((fh, fw), dtype=bool)
    brow_zone[124 - fy0 : 147 - fy0, 312 - fx0 : 380 - fx0] = True
    # Eyebrows in raw_rgba are neutral charcoal (B - R <= 11), whereas hair bang shadows are blue (B - R >= 14)
    brow_valid = brow_zone & (min_b - min_r <= 11.0) & (min_g < 192.0)

    lower_zone = np.zeros((fh, fw), dtype=bool)
    lower_zone[170 - fy0 : 198 - fy0, 332 - fx0 : 362 - fx0] = True  # nose + mouth

    tear_zone = np.zeros((fh, fw), dtype=bool)
    if expr_key == "flustered":
        tear_zone[160 - fy0 : 186 - fy0, 310 - fx0 : 386 - fx0] = True

    active_zone = (brow_valid | lower_zone | tear_zone) & not_sleeve
    line_w = np.clip((line_darkness - 8.0) / 85.0, 0.0, 1.0) * active_zone.astype(np.float32)

    line_b = np.clip(min_b * 0.72 + 48.0, 62.0, 228.0)
    line_g = np.clip(min_g * 0.58 + 20.0, 26.0, 195.0)
    line_r = np.clip(min_r * 0.68 + 42.0, 52.0, 222.0)
    line_target = np.stack([line_b, line_g, line_r], axis=-1)

    f_grad = f_grad * (1.0 - line_w[:, :, None]) + line_target * line_w[:, :, None]
    graded[fy0:fy1, fx0:fx1] = f_grad

    # 3. Apply image.png horizontal CRT chromatic shift (+3px R right, -2px B/G left)
    r_ch = graded[:, :, 2]
    g_ch = graded[:, :, 1]
    b_ch = graded[:, :, 0]

    r_shift = np.zeros_like(r_ch)
    r_shift[:, 3:] = r_ch[:, :-3]
    r_shift[:, :3] = r_ch[:, :3]

    b_shift = np.zeros_like(b_ch)
    b_shift[:, :-2] = b_ch[:, 2:]
    b_shift[:, -2:] = b_ch[:, -2:]

    g_shift = np.zeros_like(g_ch)
    g_shift[:, :-1] = g_ch[:, 1:]
    g_shift[:, -1:] = g_ch[:, -1:]

    r_final = 0.55 * r_ch + 0.45 * r_shift
    g_final = 0.72 * g_ch + 0.28 * g_shift
    b_final = 0.68 * b_ch + 0.32 * b_shift

    bgr_graded = np.clip(np.round(np.stack([b_final, g_final, r_final], axis=-1)), 0, 255).astype(np.uint8)

    # Exact 2.0x integer upscale -> uniform 4x4 pixel blocks (1412 x 2048)
    tw, th = w * 2, h * 2
    bgr_2x = cv2.resize(bgr_graded, (tw, th), interpolation=cv2.INTER_NEAREST)
    alpha_2x = np.where(
        cv2.resize(alpha_src, (tw, th), interpolation=cv2.INTER_NEAREST) >= 120,
        255,
        0,
    ).astype(np.uint8)

    canvas_bgr = np.zeros((1440, 1920, 3), dtype=np.uint8)
    canvas_a = np.zeros((1440, 1920), dtype=np.uint8)
    ox_c, oy_c = 280, 96
    max_y = 780  # Extend behind the 3D shop counter top edge (y=744)
    copy_h = max_y - oy_c
    canvas_bgr[oy_c:max_y, ox_c : ox_c + tw] = bgr_2x[:copy_h, :]
    canvas_a[oy_c:max_y, ox_c : ox_c + tw] = alpha_2x[:copy_h, :]

    return bgr_to_rgba_padded(canvas_bgr, canvas_a, pad_iters=6)


def generate_extended_star(canvas_bgr: np.ndarray, canvas_alpha: np.ndarray, cx: int, cy: int, radius: int) -> None:
    """Draw a sharp Deltarune-style 4-point pixel star with cyan/red chromatic edge."""
    for d in range(-radius, radius + 1):
        thick = max(2, (radius - abs(d)) // 2 + 1)
        x0, x1 = cx - thick, cx + thick + 1
        y0, y1 = cy + d - 1, cy + d + 2
        if 0 <= y0 and y1 < canvas_bgr.shape[0] and 0 <= x0 - 3 and x1 + 3 < canvas_bgr.shape[1]:
            canvas_bgr[y0:y1, x0 - 3 : x0] = (240, 210, 15)
            canvas_alpha[y0:y1, x0 - 3 : x0] = 200
            canvas_bgr[y0:y1, x1 : x1 + 3] = (45, 45, 255)
            canvas_alpha[y0:y1, x1 : x1 + 3] = 200
            canvas_bgr[y0:y1, x0:x1] = (252, 248, 228)
            canvas_alpha[y0:y1, x0:x1] = 255
    for d in range(-radius, radius + 1):
        thick = max(2, (radius - abs(d)) // 2 + 1)
        y0, y1 = cy - thick, cy + thick + 1
        x0, x1 = cx + d - 1, cx + d + 2
        if 0 <= y0 and y1 < canvas_bgr.shape[0] and 0 <= x0 and x1 < canvas_bgr.shape[1]:
            canvas_bgr[y0:y1, x0:x1] = (252, 248, 228)
            canvas_alpha[y0:y1, x0:x1] = 255


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract lossless BGM audio
    extract_lossless_m4a(VIDEO_PATH, ASSETS_DIR / "bgm.m4a")

    # 2. Read reference frame 300
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError("Failed to read reference frame from video")

    h, w, _ = frame.shape
    cv2.imwrite(str(ASSETS_DIR / "reference_frame.png"), frame)

    # Extended 2x world dimensions: 3840x2880 centered at (ox=960, oy=720)
    ext_w, ext_h = w * 2, h * 2
    ox, oy = w // 2, h // 2  # 960, 720

    # =========================================================================
    # LAYER 1: FOREGROUND BRANCHES (clean + extended seamlessly for Dolly Zoom)
    # =========================================================================
    branch_mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in [(0, 0, 650, 430), (1350, 0, 1920, 560)]:
        roi = frame[y0:y1, x0:x1]
        b, g, r = cv2.split(roi)
        # Pure dark silhouettes ONLY (excludes corner HUD star glints!)
        is_dark_branch = (b < 48) & (g < 22) & (r < 25)
        branch_mask[y0:y1, x0:x1] = np.maximum(
            branch_mask[y0:y1, x0:x1], is_dark_branch.astype(np.uint8) * 255
        )
    # Remove any small disconnected specs < 35 px near top corners
    n_bl, b_lbl, b_stats, _ = cv2.connectedComponentsWithStats(branch_mask)
    for lid in range(1, n_bl):
        if b_stats[lid, cv2.CC_STAT_AREA] < 35:
            branch_mask[b_lbl == lid] = 0

    cv2.imwrite(str(ASSETS_DIR / "layer_branches_fg.png"), bgr_to_rgba_padded(frame, branch_mask))

    # Seamless 3840x2880 extended branch canopy via C0 radial border projection (zero seams/cliffs/islands!)
    yy_b, xx_b = np.mgrid[0:ext_h, 0:ext_w]
    vx = xx_b - ox
    vy = yy_b - oy

    inside_orig = (vx >= 0) & (vx < w) & (vy >= 0) & (vy < h)
    dx_c = vx - (w * 0.5)
    dy_c = vy - (h * 0.5)
    scale_x = np.where(dx_c != 0, (w * 0.5 - 1.0) / np.maximum(np.abs(dx_c), 1e-6), 1e6)
    scale_y = np.where(dy_c != 0, (h * 0.5 - 1.0) / np.maximum(np.abs(dy_c), 1e-6), 1e6)
    t_ray = np.minimum(1.0, np.minimum(scale_x, scale_y))

    sx = np.where(inside_orig, vx, np.round(w * 0.5 + dx_c * t_ray)).astype(np.int32)
    sy = np.where(inside_orig, vy, np.round(h * 0.5 + dy_c * t_ray)).astype(np.int32)
    sx = np.clip(sx, 0, w - 1)
    sy = np.clip(sy, 0, h - 1)

    br_ext_bgr = frame[sy, sx].copy()
    # Keep continuous C0 silhouette outside original box while preventing bluish border streaks
    br_ext_alpha = np.where(
        inside_orig,
        branch_mask[sy, sx],
        np.where(yy_b < (oy + 560), branch_mask[sy, sx], 0),
    ).astype(np.uint8)
    br_ext_bgr[~inside_orig] = (2, 0, 6)

    cv2.imwrite(
        str(ASSETS_DIR / "layer_branches_fg_ext.png"),
        bgr_to_rgba_padded(br_ext_bgr, br_ext_alpha, pad_iters=4),
    )

    # =========================================================================
    # LAYER 2: STARS (original + seamless 3840x2880 extended peripheral sky!)
    # =========================================================================
    star_boxes = [
        (311, 213, 379, 276, "near"),
        (1697, 311, 1719, 325, "far"),
        (462, 347, 495, 373, "near"),
        (1573, 395, 1615, 435, "near"),
        (1887, 468, 1907, 483, "far"),
        (155, 577, 175, 591, "far"),
        (1813, 584, 1859, 621, "near"),
    ]

    stars_far_alpha = np.zeros((h, w), dtype=np.uint8)
    stars_near_alpha = np.zeros((h, w), dtype=np.uint8)
    sky_base_color = np.array([109.0, 53.0, 30.0], dtype=np.float32)

    star_metrics = []
    for idx, (x0, y0, x1, y1, plane) in enumerate(star_boxes):
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        diff = np.linalg.norm(roi - sky_base_color[None, None, :], axis=2)
        alpha = np.clip((diff - 10.0) / 35.0, 0.0, 1.0)
        alpha = np.where(alpha > 0.45, 1.0, alpha)
        alpha_u8 = (alpha * 255.0).astype(np.uint8)
        if plane == "near":
            stars_near_alpha[y0:y1, x0:x1] = alpha_u8
        else:
            stars_far_alpha[y0:y1, x0:x1] = alpha_u8
        star_metrics.append({
            "id": idx,
            "bbox": [x0, y0, x1, y1],
            "cx": 0.5 * (x0 + x1),
            "cy": 0.5 * (y0 + y1),
            "plane": plane,
        })

    cv2.imwrite(str(ASSETS_DIR / "layer_stars_far.png"), bgr_to_rgba_padded(frame, stars_far_alpha))
    cv2.imwrite(str(ASSETS_DIR / "layer_stars_near.png"), bgr_to_rgba_padded(frame, stars_near_alpha))

    # Extended 3840x2880 star layers with outer stars ONLY outside central [960..2880, 720..2160]
    sf_ext_bgr = np.zeros((ext_h, ext_w, 3), dtype=np.uint8)
    sf_ext_a = np.zeros((ext_h, ext_w), dtype=np.uint8)
    sf_ext_bgr[oy : oy + h, ox : ox + w] = frame
    sf_ext_a[oy : oy + h, ox : ox + w] = stars_far_alpha

    sn_ext_bgr = np.zeros((ext_h, ext_w, 3), dtype=np.uint8)
    sn_ext_a = np.zeros((ext_h, ext_w), dtype=np.uint8)
    sn_ext_bgr[oy : oy + h, ox : ox + w] = frame
    sn_ext_a[oy : oy + h, ox : ox + w] = stars_near_alpha

    rng = np.random.default_rng(660)
    for _ in range(38):
        sx = int(rng.integers(180, ext_w - 180))
        sy = int(rng.integers(240, oy + h - 80))
        if (ox - 20 <= sx <= ox + w + 20) and (oy - 20 <= sy <= oy + h + 20):
            continue
        rad = int(rng.choice([8, 11, 15, 19]))
        if rad >= 14:
            generate_extended_star(sn_ext_bgr, sn_ext_a, sx, sy, rad)
        else:
            generate_extended_star(sf_ext_bgr, sf_ext_a, sx, sy, rad)

    cv2.imwrite(str(ASSETS_DIR / "layer_stars_far_ext.png"), bgr_to_rgba_padded(sf_ext_bgr, sf_ext_a, pad_iters=4))
    cv2.imwrite(str(ASSETS_DIR / "layer_stars_near_ext.png"), bgr_to_rgba_padded(sn_ext_bgr, sn_ext_a, pad_iters=4))

    # =========================================================================
    # LAYER 3: SEAMLESS UNIFIED CHARACTER SPRITE + 10 EXPRESSION SPRITES
    # =========================================================================
    cx0, cy0, cx1, cy1 = 640, 90, 1350, 720
    croi = frame[cy0:cy1, cx0:cx1].copy()
    rh, rw, _ = croi.shape
    yy_c, xx_c = np.mgrid[cy0:cy1, cx0:cx1]

    d_top = np.linalg.norm(croi.astype(np.float32) - np.array([70.0, 0.0, 12.0]), axis=2)
    d_mid = np.linalg.norm(croi.astype(np.float32) - np.array([104.0, 21.0, 22.0]), axis=2)
    d_inn = np.linalg.norm(croi.astype(np.float32) - np.array([109.0, 53.0, 30.0]), axis=2)

    is_sky_bg = (
        ((yy_c < 118) & ((d_top < 22.0) | (d_mid < 20.0) | (d_inn < 16.0)))
        | ((yy_c >= 118) & (d_inn < 16.0))
    )
    inv = is_sky_bg.astype(np.uint8) * 255
    flood = inv.copy()
    ext = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
    for y in range(0, rh, 4):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, ext, (0, y), 128)
        if flood[y, rw - 1] == 255:
            cv2.floodFill(flood, ext, (rw - 1, y), 128)
    for x in range(0, rw, 4):
        if flood[0, x] == 255:
            cv2.floodFill(flood, ext, (x, 0), 128)

    char_mask_roi = (flood != 128).astype(np.uint8) * 255
    curl_hole = (xx_c >= 1205) & (xx_c <= 1270) & (yy_c >= 540) & (yy_c <= 645) & (d_inn <= 18.0)
    char_mask_roi[curl_hole] = 0

    char_alpha_full = np.zeros((h, w), dtype=np.uint8)
    char_alpha_full[cy0:cy1, cx0:cx1] = char_mask_roi

    # Extend bottom of character smoothly behind the black UI box (y=720..780)
    char_bgr_full = frame.copy()
    char_bgr_full[720:, :] = 0
    for dy in range(1, 61):
        char_bgr_full[719 + dy, cx0:cx1] = frame[719, cx0:cx1]
        char_alpha_full[719 + dy, cx0:cx1] = char_alpha_full[719, cx0:cx1]

    # Build 14 aligned pixel-art Natsume Anan expressions from user-uploaded PNG pairs!
    expr_asset_names = []
    expr_rgba_map = {}
    for expr_key, (pix_fname, raw_fname) in EXPRESSION_PAIRS.items():
        pix_rgba = cv2.imread(str(EXPR_DIR / pix_fname), cv2.IMREAD_UNCHANGED)
        raw_rgba = cv2.imread(str(EXPR_DIR / raw_fname), cv2.IMREAD_UNCHANGED)
        if pix_rgba is None or pix_rgba.shape[2] < 4:
            raise RuntimeError(f"Missing RGBA pixelated expression image: {pix_fname}")
        if raw_rgba is None or raw_rgba.shape[2] < 4:
            raise RuntimeError(f"Missing RGBA raw expression image: {raw_fname}")
        rgba_expr = build_pixel_char_from_uploaded_png(pix_rgba, raw_rgba, expr_key=expr_key)
        out_name = f"char_expr_{expr_key}.png"
        cv2.imwrite(str(ASSETS_DIR / out_name), rgba_expr)
        expr_asset_names.append(out_name)
        expr_rgba_map[expr_key] = rgba_expr

    # Use lavender-graded calm pixel-art sprite as char_full.png so default & expressions are 100% consistent
    char_full_rgba = expr_rgba_map["calm"]
    cv2.imwrite(str(ASSETS_DIR / "char_full.png"), char_full_rgba)

    # Blink Overlays built directly from char_full_rgba so eye blink matches color & pixel grid!
    char_full_bgr = char_full_rgba[:, :, :3].copy()
    bh = make_blink_layer(char_full_bgr, 12)
    bc = make_blink_layer(char_full_bgr, 24)
    cv2.imwrite(str(ASSETS_DIR / "char_blink_half.png"), bh)
    cv2.imwrite(str(ASSETS_DIR / "char_blink_closed.png"), bc)

    # =========================================================================
    # LAYER 0: CLEAN INPAINTED SKY DOME (1920x1440 + seamless 3840x2880 for Dolly Zoom!)
    # =========================================================================
    yy, xx = np.mgrid[0:h, 0:w]
    xs_fit = np.array([100, 200, 400, 600, 960, 1400, 1600, 1800], dtype=np.float64)
    y_top_fit = np.array([255, 188, 132, 108, 95, 108, 152, 230], dtype=np.float64)
    y_bot_fit = np.array([310, 232, 154, 120, 108, 132, 178, 275], dtype=np.float64)
    poly_top = np.poly1d(np.polyfit(xs_fit, y_top_fit, 2))
    poly_bot = np.poly1d(np.polyfit(xs_fit, y_bot_fit, 2))

    c_deep = np.array([70, 0, 12], dtype=np.uint8)
    c_mid = np.array([104, 21, 22], dtype=np.uint8)
    c_inner = np.array([109, 53, 30], dtype=np.uint8)

    sky_synth = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(w):
        qx = (x // 6) * 6
        yt = int(round(poly_top(qx) / 6.0)) * 6
        yb = int(round(poly_bot(qx) / 6.0)) * 6
        sky_synth[0:yt, x] = c_deep
        sky_synth[yt:yb, x] = c_mid
        sky_synth[yb:h, x] = c_inner

    dist = np.hypot((xx - 975.0) / 340.0, (yy - 410.0) / 290.0)
    glow = np.clip(1.0 - dist, 0.0, 1.0) ** 2
    inner_mask = sky_synth[:, :, 1] == 53
    for c, delta in enumerate([-4.0, -5.0, -5.0]):
        ch = sky_synth[:, :, c].astype(np.float32)
        ch[inner_mask] = np.clip(ch[inner_mask] + glow[inner_mask] * delta, 0, 255)
        sky_synth[:, :, c] = ch.astype(np.uint8)

    cv2.imwrite(str(ASSETS_DIR / "layer_sky_bg.png"), sky_synth)

    # Extended 3840x2880 sky dome matching exact center 1:1 while curving infinitely outward!
    sky_ext = np.zeros((ext_h, ext_w, 3), dtype=np.uint8)
    yy_e, xx_e = np.mgrid[0:ext_h, 0:ext_w]
    for ex in range(ext_w):
        vx = ex - ox
        qx = (vx // 6) * 6
        yt = int(np.clip(oy + round(poly_top(qx) / 6.0) * 6, 0, ext_h))
        yb = int(np.clip(oy + round(poly_bot(qx) / 6.0) * 6, yt, ext_h))
        sky_ext[0:yt, ex] = c_deep
        sky_ext[yt:yb, ex] = c_mid
        sky_ext[yb:ext_h, ex] = c_inner

    dist_e = np.hypot(((xx_e - ox) - 975.0) / 340.0, ((yy_e - oy) - 410.0) / 290.0)
    glow_e = np.clip(1.0 - dist_e, 0.0, 1.0) ** 2
    inner_e = sky_ext[:, :, 1] == 53
    for c, delta in enumerate([-4.0, -5.0, -5.0]):
        ch = sky_ext[:, :, c].astype(np.float32)
        ch[inner_e] = np.clip(ch[inner_e] + glow_e[inner_e] * delta, 0, 255)
        sky_ext[:, :, c] = ch.astype(np.uint8)

    cv2.imwrite(str(ASSETS_DIR / "layer_sky_bg_ext.png"), sky_ext)

    # =========================================================================
    # LAYER 4 & 5: DELTARUNE SHOP UI BORDER FRAME & FLOATING TEXT SPRITES
    # =========================================================================
    ui_frame_bgr = frame.copy()
    ui_frame_bgr[0:720, :] = 0

    ui_text_mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in [(68, 790, 1180, 1380), (1280, 785, 1860, 1380)]:
        roi = frame[y0:y1, x0:x1]
        m = (roi.max(axis=2) > 18).astype(np.uint8) * 255
        ui_text_mask[y0:y1, x0:x1] = m
        ui_frame_bgr[y0:y1, x0:x1] = 0

    cv2.imwrite(str(ASSETS_DIR / "ui_text_default.png"), bgr_to_rgba_padded(frame, ui_text_mask))

    ui_border_alpha = np.zeros((h, w), dtype=np.uint8)
    ui_roi = ui_frame_bgr[720:1440, 0:1920]
    border_active = (ui_roi.max(axis=2) > 16).astype(np.uint8) * 255
    border_closed = cv2.morphologyEx(border_active, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ui_border_alpha[720:1440, 0:1920] = np.maximum(border_active, border_closed)
    cv2.imwrite(str(ASSETS_DIR / "ui_frame.png"), bgr_to_rgba_padded(ui_frame_bgr, ui_border_alpha))

    sprite_defs = {
        "asterisk": (76, 824, 180, 908),
        "prompt_nani": (196, 812, 558, 922),
        "heart": (1336, 808, 1424, 878),
        "opt_buy": (1436, 788, 1618, 888),
        "opt_sell": (1430, 906, 1624, 1008),
        "opt_talk": (1430, 1026, 1630, 1132),
        "opt_exit": (1436, 1148, 1710, 1246),
        "gold_49": (1304, 1280, 1468, 1368),
    }

    sprite_metrics = {}
    for name, (x0, y0, x1, y1) in sprite_defs.items():
        s_bgr = frame[y0:y1, x0:x1]
        s_alpha = ui_text_mask[y0:y1, x0:x1]
        cv2.imwrite(str(ASSETS_DIR / f"sprite_{name}.png"), bgr_to_rgba_padded(s_bgr, s_alpha))
        sprite_metrics[name] = {
            "bbox": [x0, y0, x1, y1],
            "w": x1 - x0,
            "h": y1 - y0,
            "cx": 0.5 * (x0 + x1),
            "cy": 0.5 * (y0 + y1),
        }

    metrics = {
        "canvas": {"w": w, "h": h, "splitY": 720},
        "stars": star_metrics,
        "sprites": sprite_metrics,
        "character": {
            "bbox": [cx0, cy0, cx1, cy1],
            "eyeLeft": {"cx": 1048, "cy": 405, "w": 88, "h": 42},
            "headPivot": {"x": 975, "y": 480},
            "torsoPivot": {"x": 975, "y": 680},
        },
        "uiBoxes": {
            "leftInterior": [37, 763, 1204, 1410],
            "rightInterior": [1249, 763, 1890, 1410],
            "outerRect": [12, 744, 1915, 1428],
        },
    }
    (ASSETS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Rebuild embedded_assets.js (using lavender-graded char_full.png + 10 expressions + extended Dolly Zoom layers)
    files = [
        "layer_sky_bg.png",
        "layer_sky_bg_ext.png",
        "layer_stars_far.png",
        "layer_stars_far_ext.png",
        "layer_stars_near.png",
        "layer_stars_near_ext.png",
        "layer_branches_fg.png",
        "layer_branches_fg_ext.png",
        "char_full.png",
        "char_blink_half.png",
        "char_blink_closed.png",
        *expr_asset_names,
        "ui_frame.png",
        "ui_text_default.png",
        "sprite_asterisk.png",
        "sprite_prompt_nani.png",
        "sprite_heart.png",
        "sprite_opt_buy.png",
        "sprite_opt_sell.png",
        "sprite_opt_talk.png",
        "sprite_opt_exit.png",
        "sprite_gold_49.png",
        "bgm.m4a",
    ]
    bundle = {}
    for fname in files:
        p = ASSETS_DIR / fname
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        mime = "audio/mp4" if fname.endswith(".m4a") else "image/png"
        key = fname.rsplit(".", 1)[0]
        bundle[key] = f"data:{mime};base64,{b64}"

    js_content = "window.BLOOM_SHOP_METRICS = " + json.dumps(metrics) + ";\n"
    js_content += "window.BLOOM_SHOP_ASSETS = " + json.dumps(bundle) + ";\n"
    (ASSETS_DIR / "embedded_assets.js").write_text(js_content, encoding="utf-8")
    print("Rebuilt seamless unified character and embedded_assets.js.")


if __name__ == "__main__":
    main()
