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

EXPRESSION_FILES = {
    "shy_hide": "media_1794590797246_8047e3d0.jpg",
    "nervous_hide": "media_1794590798978_a3a9c32e.jpg",
    "sleeve_pout": "media_1794590800937_c34e32c3.jpg",
    "sketchbook": "media_1794590802812_98e86193.jpg",
    "flustered": "media_1794590804323_e34242b9.jpg",
    "calm": "media_1794590805700_0dbd98b4.jpg",
    "proud_blush": "media_1794590808274_cedf58a2.jpg",
    "sleepy": "media_1794590811394_bdb9ff1c.jpg",
    "smug": "media_1794590814142_da1e7bbe.jpg",
    "surprised": "media_1794590815911_50a3aad4.jpg",
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


def extract_clean_matte_4x_pixel_expr(
    img_bgr: np.ndarray,
    ref_bonnet_gray: np.ndarray,
    tint_scale: tuple[float, float, float] = (1.03, 0.99, 0.96),
) -> np.ndarray:
    """
    Align an uploaded Natsume Anan official sprite to exact reference coordinates (1920x1440),
    extract a clean hole-free foreground alpha matte, stylize to 4x4 pixel-art blocks
    matching char_full.png, extend bottom rows behind y=720, and apply edge RGB bleed padding.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    top_g = g[: min(330, g.shape[0]), :]

    best = (-1.0, 2.33, 340, 22)
    for S in np.linspace(2.20, 2.75, 56):
        rw = int(round(ref_bonnet_gray.shape[1] / S))
        rh = int(round(ref_bonnet_gray.shape[0] / S))
        if rw >= top_g.shape[1] or rh >= top_g.shape[0] or rw < 40 or rh < 40:
            continue
        tmpl = cv2.resize(ref_bonnet_gray, (rw, rh), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(top_g, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best[0]:
            best = (mv, float(S), int(ml[0]), int(ml[1]))

    _, S, x0, y0 = best
    M = np.float32([[S, 0.0, 760.0 - x0 * S], [0.0, S, 110.0 - y0 * S]])
    warped = cv2.warpAffine(
        img_bgr, M, (1920, 1440),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(12, 10, 10),
    )

    # Extract matte inside character workspace x=600..1380, y=80..740
    roi = warped[80:740, 600:1380]
    rh, rw = roi.shape[:2]
    bg_l = np.median(roi[:, :24], axis=1)
    bg_r = np.median(roi[:, -24:], axis=1)
    bg_row = 0.5 * (bg_l + bg_r)

    diff_bg = np.linalg.norm(roi.astype(np.float32) - bg_row[:, None, :], axis=2)
    b, gr, r = cv2.split(roi)
    chroma = np.abs(b.astype(np.int16) - r.astype(np.int16))
    is_bg_cand = (diff_bg < 16.0) & (chroma < 7)

    flood = is_bg_cand.astype(np.uint8) * 255
    ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
    for y in range(0, rh, 2):
        if flood[y, 0] == 255:
            cv2.floodFill(flood, ff_mask, (0, y), 128)
        if flood[y, rw - 1] == 255:
            cv2.floodFill(flood, ff_mask, (rw - 1, y), 128)
    for x in range(0, rw, 2):
        if flood[0, x] == 255:
            cv2.floodFill(flood, ff_mask, (x, 0), 128)

    fg_bin = (flood != 128).astype(np.uint8) * 255
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(fg_bin)
    if n_lbl > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg_bin = (labels == largest).astype(np.uint8) * 255

    # Punch out known interior studio background pockets between outer hair curls & skirt
    yy_r, xx_r = np.mgrid[80:740, 600:1380]
    right_curl_pocket = (xx_r >= 1190) & (xx_r <= 1280) & (yy_r >= 520) & (yy_r <= 670) & is_bg_cand
    left_hair_pocket = (xx_r >= 715) & (xx_r <= 790) & (yy_r >= 560) & (yy_r <= 720) & is_bg_cand
    fg_bin[right_curl_pocket | left_hair_pocket] = 0

    alpha_roi = cv2.GaussianBlur(fg_bin, (3, 3), 0.7)
    alpha_full = np.zeros((1440, 1920), dtype=np.uint8)
    alpha_full[80:740, 600:1380] = alpha_roi

    # Extend rows 719..779 downward behind black UI box so head/breath motion never reveals bottom edge
    for dy in range(1, 61):
        warped[719 + dy, 600:1380] = warped[719, 600:1380]
        alpha_full[719 + dy, 600:1380] = alpha_full[719, 600:1380]

    # Harmonize subtle nocturnal blue color temperature & pixelate to 480x360 (4x4 grid)
    work = warped.astype(np.float32)
    for c, mul in enumerate(tint_scale):
        work[:, :, c] = np.clip(work[:, :, c] * mul, 0, 255)

    # Slight contrast boost & cool shadow tint so white hair & navy dress match char_full.png
    pix_w, pix_h = 480, 360
    small_bgr = cv2.resize(work.astype(np.uint8), (pix_w, pix_h), interpolation=cv2.INTER_AREA)
    small_a = cv2.resize(alpha_full, (pix_w, pix_h), interpolation=cv2.INTER_AREA)
    pix_bgr = cv2.resize(small_bgr, (1920, 1440), interpolation=cv2.INTER_NEAREST)
    pix_a = np.where(
        cv2.resize(small_a, (1920, 1440), interpolation=cv2.INTER_NEAREST) >= 120,
        255,
        0,
    ).astype(np.uint8)

    return bgr_to_rgba_padded(pix_bgr, pix_a, pad_iters=6)


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

    char_full_rgba = bgr_to_rgba_padded(char_bgr_full, char_alpha_full, pad_iters=6)
    cv2.imwrite(str(ASSETS_DIR / "char_full.png"), char_full_rgba)

    # Build 10 aligned pixel-art Natsume Anan expressions!
    ref_bonnet_gray = cv2.cvtColor(char_full_rgba[110:370, 760:1200, :3], cv2.COLOR_BGR2GRAY)
    expr_asset_names = []
    for expr_key, fname in EXPRESSION_FILES.items():
        img_bgr = cv2.imread(str(EXPR_DIR / fname))
        if img_bgr is None:
            raise RuntimeError(f"Missing expression image: {fname}")
        rgba_expr = extract_clean_matte_4x_pixel_expr(img_bgr, ref_bonnet_gray)
        out_name = f"char_expr_{expr_key}.png"
        cv2.imwrite(str(ASSETS_DIR / out_name), rgba_expr)
        expr_asset_names.append(out_name)

    # Blink Overlays (padded so no edge outline ever appears)
    bh = make_blink_layer(frame, 12)
    bc = make_blink_layer(frame, 25)
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

    # Rebuild embedded_assets.js (using original char_full.png + extended Dolly Zoom layers)
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
