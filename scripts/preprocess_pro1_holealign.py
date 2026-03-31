import argparse
from pathlib import Path

import cv2
import numpy as np


def to_grayscale(img):
    # 컬러 이미지를 gray scale로 변환한다.
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def collect_files(input_dir, pattern):
    # 콤마로 전달된 복수 패턴을 합쳐 입력 파일 목록을 만든다.
    patterns = [p.strip() for p in str(pattern).split(",") if p.strip()]
    if not patterns:
        patterns = ["*.BMP"]

    merged = []
    for pat in patterns:
        merged.extend(input_dir.glob(pat))

    # 중복 제거 + 정렬
    files = sorted({p.resolve() for p in merged if p.is_file()})
    return [Path(p) for p in files]


def extract_main_component(gray, foreground="bright"):
    # 가장 큰 연결 성분(카테터 외곽)을 추출한다.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if foreground == "bright":
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return None

    h, w = gray.shape
    img_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    diag = max(float(np.hypot(w, h)), 1.0)

    best_idx = None
    best_score = -1.0
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 100:
            continue

        c = centroids[idx].astype(np.float32)
        dist = float(np.linalg.norm(c - img_center) / diag)
        score = float(area) * (1.5 - min(dist, 1.2))

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[labels == best_idx] = 255

    x = int(stats[best_idx, cv2.CC_STAT_LEFT])
    y = int(stats[best_idx, cv2.CC_STAT_TOP])
    bw = int(stats[best_idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[best_idx, cv2.CC_STAT_HEIGHT])
    cx, cy = centroids[best_idx]
    area = int(stats[best_idx, cv2.CC_STAT_AREA])

    return {
        "mask": mask,
        "center": (float(cx), float(cy)),
        "bbox": (x, y, bw, bh),
        "area": area,
    }


def extract_target_outer_component_pro1(gray):
    # pro1 타깃 전용: 고임계 기반으로 밝은 카테터 링 후보를 우선 선택한다.
    h, w = gray.shape
    area_total = float(h * w)
    img_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    diag = max(float(np.hypot(w, h)), 1.0)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best = None

    for thr in [245, 240, 235, 230, 225, 220, 215, 210, 205, 200]:
        _, binary = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            contour_area = float(cv2.contourArea(cnt))
            if contour_area < area_total * 0.001 or contour_area > area_total * 0.35:
                continue

            peri = float(cv2.arcLength(cnt, True))
            circularity = 0.0 if peri == 0 else float(4.0 * np.pi * contour_area / (peri * peri))

            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = float(bw / max(bh, 1))
            ar_score = 1.0 - min(abs(np.log(max(ar, 1e-6))), 1.0)

            fill_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(fill_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            fill_area = float(np.count_nonzero(fill_mask))
            if fill_area <= 0:
                continue

            comp_mask = np.zeros_like(gray, dtype=np.uint8)
            comp_mask[(binary > 0) & (fill_mask > 0)] = 255
            bright_area = float(np.count_nonzero(comp_mask))
            fill_ratio = bright_area / fill_area

            pts = gray[fill_mask > 0]
            mean_intensity = float(pts.mean()) if pts.size > 0 else 0.0

            m = cv2.moments(fill_mask)
            if m["m00"] == 0:
                cx, cy = float(x + bw * 0.5), float(y + bh * 0.5)
            else:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
            dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - img_center) / diag)

            score = (
                3.0 * min(max(circularity, 0.0), 1.0)
                + 2.5 * min(max(fill_ratio, 0.0), 1.0)
                + 1.0 * min(max((mean_intensity - 120.0) / 135.0, 0.0), 1.0)
                + 1.0 * (1.0 - min(dist * 1.2, 1.0))
                + 0.5 * max(ar_score, 0.0)
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "mask": comp_mask,
                    "center": (cx, cy),
                    "bbox": (x, y, int(bw), int(bh)),
                    "area": int(bright_area),
                }

    if best is not None:
        return {
            "mask": best["mask"],
            "center": best["center"],
            "bbox": best["bbox"],
            "area": best["area"],
        }

    # 고임계 검출 실패 시 기존 공통 로직으로 폴백한다.
    return extract_main_component(gray, foreground="bright")


def fill_section_mask(outer_mask):
    # 외곽 contour를 채워서 "단면 전체 영역" 마스크를 만든다.
    contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(outer_mask, dtype=np.uint8)
    cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    return filled


def mask_centroid(mask):
    # 이진 마스크의 무게중심을 구한다.
    m = cv2.moments(mask)
    if m["m00"] == 0:
        h, w = mask.shape
        return (float(w) * 0.5, float(h) * 0.5)
    return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))


def rotate_image(img, angle_deg, center, interpolation, border_value=0, border_mode=cv2.BORDER_CONSTANT):
    # 중심(center) 기준으로 이미지를 회전한다.
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def iou_score(mask_a, mask_b):
    # 두 마스크의 IoU 점수를 계산한다.
    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def place_to_template(gray_target, outer_mask, section_mask, template_shape, template_center, scale):
    # 타깃(gray/mask)을 템플릿 좌표계로 스케일/평행이동한다.
    h_t, w_t = template_shape

    scaled_gray = cv2.resize(gray_target, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_outer = cv2.resize(outer_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    scaled_section = cv2.resize(section_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    src_center = mask_centroid(scaled_section)
    dx = float(template_center[0] - src_center[0])
    dy = float(template_center[1] - src_center[1])
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    placed_gray = cv2.warpAffine(
        scaled_gray,
        matrix,
        (w_t, h_t),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    placed_outer = cv2.warpAffine(
        scaled_outer,
        matrix,
        (w_t, h_t),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    placed_section = cv2.warpAffine(
        scaled_section,
        matrix,
        (w_t, h_t),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return placed_gray, placed_outer, placed_section, dx, dy


def _extract_hole_components(binary_mask, section_area, min_ratio, max_ratio, min_circularity):
    # hole 후보 contour를 면적/원형도 기준으로 필터링한다.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rows = []
    min_area = float(section_area) * float(min_ratio)
    max_area = float(section_area) * float(max_ratio)

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        peri = float(cv2.arcLength(cnt, True))
        circularity = 0.0 if peri == 0 else float(4.0 * np.pi * area / (peri * peri))
        if circularity < float(min_circularity):
            continue

        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue

        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        rows.append(
            {
                "contour": cnt,
                "area": area,
                "circularity": circularity,
                "center": (cx, cy),
            }
        )

    rows.sort(key=lambda x: x["area"], reverse=True)
    return rows


def _split_component_into_two(component, shape):
    # 하나로 붙어 검출된 성분을 k-means(2클러스터)로 분리한다.
    comp_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(comp_mask, [component["contour"]], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(comp_mask > 0)
    if xs.size < 200:
        return None

    data = np.stack([xs, ys], axis=1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, _ = cv2.kmeans(data, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    split_components = []
    for idx in [0, 1]:
        m = np.zeros(shape, dtype=np.uint8)
        sel = labels == idx
        m[ys[sel], xs[sel]] = 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue

        peri = float(cv2.arcLength(cnt, True))
        circularity = 0.0 if peri == 0 else float(4.0 * np.pi * area / (peri * peri))
        moment = cv2.moments(cnt)
        if moment["m00"] == 0:
            continue

        cx = float(moment["m10"] / moment["m00"])
        cy = float(moment["m01"] / moment["m00"])
        split_components.append(
            {
                "contour": cnt,
                "area": area,
                "circularity": circularity,
                "center": (cx, cy),
            }
        )

    if len(split_components) != 2:
        return None
    split_components.sort(key=lambda x: x["area"], reverse=True)
    return split_components


def _augment_components_if_merged_for_2(components, shape, section_area, min_ratio, max_ratio, min_circularity):
    # 1개로 검출되면 분할을 시도해서 2개 hole을 복원한다.
    if len(components) != 1:
        return components

    split = _split_component_into_two(components[0], shape)
    if split is None:
        return components

    min_area = float(section_area) * float(min_ratio)
    max_area = float(section_area) * float(max_ratio)

    filtered = []
    for comp in split:
        if not (min_area <= comp["area"] <= max_area):
            continue
        if comp["circularity"] < float(min_circularity) * 0.85:
            continue
        filtered.append(comp)

    filtered.sort(key=lambda x: x["area"], reverse=True)
    if len(filtered) == 2:
        return filtered
    return components


def _build_hole_group_masks_2(shape, components):
    # 면적 상위 2개를 사용해 hole1/hole2/all 마스크를 만든다.
    top2 = sorted(components, key=lambda x: x["area"], reverse=True)[:2]
    if len(top2) < 2:
        return None

    # 방향 불확실성을 줄이기 위해 y좌표 기준으로 위/아래 순서로 고정한다.
    top2 = sorted(top2, key=lambda x: x["center"][1])
    hole_1, hole_2 = top2[0], top2[1]

    hole1_mask = np.zeros(shape, dtype=np.uint8)
    hole2_mask = np.zeros(shape, dtype=np.uint8)
    all_mask = np.zeros(shape, dtype=np.uint8)

    cv2.drawContours(hole1_mask, [hole_1["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(hole2_mask, [hole_2["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(all_mask, [hole_1["contour"], hole_2["contour"]], -1, 255, thickness=cv2.FILLED)

    area_ratio = float(min(hole_1["area"], hole_2["area"]) / max(hole_1["area"], hole_2["area"], 1.0))

    return {
        "top2": top2,
        "hole_1": hole_1,
        "hole_2": hole_2,
        "hole1_mask": hole1_mask,
        "hole2_mask": hole2_mask,
        "all_mask": all_mask,
        "similarity": area_ratio,
    }


def detect_template_holes(template_gray, template_section_mask):
    # 템플릿 전용: high-threshold 기반으로 2개 lumen을 안정적으로 추출한다.
    sec_area = max(float(np.count_nonzero(template_section_mask)), 1.0)
    blurred = cv2.GaussianBlur(template_gray, (31, 31), 0)

    best = None
    for thr in [245, 242, 240, 238, 235, 232, 228, 224, 220]:
        cand = np.zeros_like(template_gray, dtype=np.uint8)
        cand[(blurred >= thr) & (template_section_mask > 0)] = 255

        cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        comps = _extract_hole_components(
            binary_mask=cand,
            section_area=sec_area,
            min_ratio=0.01,
            max_ratio=0.46,
            min_circularity=0.08,
        )
        comps = _augment_components_if_merged_for_2(
            components=comps,
            shape=template_gray.shape,
            section_area=sec_area,
            min_ratio=0.01,
            max_ratio=0.46,
            min_circularity=0.08,
        )

        grouped = _build_hole_group_masks_2(template_gray.shape, comps)
        if grouped is None:
            if best is None or len(comps) > best["count"]:
                best = {"count": len(comps), "thr": thr, "grouped": None}
            continue

        score = 1000.0 + grouped["similarity"] * 100.0
        if best is None or score > best.get("score", -1e9):
            best = {"count": len(comps), "thr": thr, "grouped": grouped, "score": score}

    if best is None or best.get("grouped") is None:
        raise RuntimeError("template_hole_detection_failed")

    out = best["grouped"]
    out["threshold"] = best["thr"]
    return out


def detect_target_holes(target_gray, target_section_mask):
    # 타깃 전용: Otsu 기반 dark-threshold + 완화 파라미터로 2개 lumen을 찾는다.
    sec_area = max(float(np.count_nonzero(target_section_mask)), 1.0)
    inner_section = cv2.erode(target_section_mask, np.ones((11, 11), np.uint8), iterations=1)

    best = None
    for ksize in [31, 25, 35]:
        blurred = cv2.GaussianBlur(target_gray, (ksize, ksize), 0)
        vals = blurred[inner_section > 0]
        if vals.size < 100:
            continue

        otsu_thr, _ = cv2.threshold(vals.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for offset in [0, 5, -5, 10, -10, 15, -15]:
            thr = float(np.clip(otsu_thr + offset, 0, 255))

            cand = np.zeros_like(target_gray, dtype=np.uint8)
            cand[(blurred <= thr) & (inner_section > 0)] = 255
            cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

            comps = _extract_hole_components(
                binary_mask=cand,
                section_area=sec_area,
                min_ratio=0.004,
                max_ratio=0.50,
                min_circularity=0.08,
            )
            comps = _augment_components_if_merged_for_2(
                components=comps,
                shape=target_gray.shape,
                section_area=sec_area,
                min_ratio=0.004,
                max_ratio=0.50,
                min_circularity=0.08,
            )

            grouped = _build_hole_group_masks_2(target_gray.shape, comps)
            count_top = min(len(comps), 2)
            if count_top == 0:
                continue

            top = comps[:count_top]
            smallest = float(top[-1]["area"])
            mean_circ = float(np.mean([x["circularity"] for x in top]))
            similarity = 0.0
            if grouped is not None:
                similarity = float(grouped["similarity"])

            # 2개 검출을 최우선으로, 그다음 안정적인 형상(원형도+면적 균형)을 선호한다.
            score = count_top * 10000.0 + smallest + mean_circ * 100.0 + similarity * 300.0

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "ksize": ksize,
                    "threshold": thr,
                    "components": comps,
                    "grouped": grouped,
                }

    if best is None:
        return None

    return best


def find_best_rotation_with_hole_groups(target_grouped, template_grouped, center, coarse_step=3.0, fine_step=0.25):
    # 2개 hole 마스크를 짝지어 회전 점수를 계산한다(순서 뒤바뀜 permutation 허용).
    t1 = target_grouped["hole1_mask"]
    t2 = target_grouped["hole2_mask"]
    t_all = target_grouped["all_mask"]

    ref1 = template_grouped["hole1_mask"]
    ref2 = template_grouped["hole2_mask"]
    ref_all = template_grouped["all_mask"]

    def score_at(angle):
        r1 = rotate_image(
            t1,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        r2 = rotate_image(
            t2,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        r_all = rotate_image(
            t_all,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )

        # hole 크기가 비슷해 라벨이 바뀔 수 있으므로 두 경우 중 더 좋은 매칭을 사용한다.
        s_11 = iou_score(r1, ref1)
        s_22 = iou_score(r2, ref2)
        s_12 = iou_score(r1, ref2)
        s_21 = iou_score(r2, ref1)

        s_pair_a = 0.5 * (s_11 + s_22)
        s_pair_b = 0.5 * (s_12 + s_21)
        s_pair = max(s_pair_a, s_pair_b)
        s_all = iou_score(r_all, ref_all)

        score = 0.75 * s_pair + 0.25 * s_all
        return score, s_pair, s_all

    best = {"angle": 0.0, "score": -1.0, "s_pair": 0.0, "s_all": 0.0}

    for angle in np.arange(-180.0, 180.0, coarse_step, dtype=np.float32):
        score, s_pair, s_all = score_at(float(angle))
        if score > best["score"]:
            best = {
                "angle": float(angle),
                "score": float(score),
                "s_pair": float(s_pair),
                "s_all": float(s_all),
            }

    for angle in np.arange(best["angle"] - coarse_step, best["angle"] + coarse_step + fine_step, fine_step, dtype=np.float32):
        score, s_pair, s_all = score_at(float(angle))
        if score > best["score"]:
            best = {
                "angle": float(angle),
                "score": float(score),
                "s_pair": float(s_pair),
                "s_all": float(s_all),
            }

    return best


def find_best_rotation_outer_mask(placed_outer_mask, template_outer_mask, center, coarse_step=3.0, fine_step=0.25):
    # hole 검출 실패 시 외곽 마스크 IoU로 fallback 회전을 계산한다.
    best_angle = 0.0
    best_score = -1.0

    for angle in np.arange(-180.0, 180.0, coarse_step, dtype=np.float32):
        rot = rotate_image(
            placed_outer_mask,
            angle_deg=float(angle),
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        score = iou_score(rot, template_outer_mask)
        if score > best_score:
            best_score = float(score)
            best_angle = float(angle)

    for angle in np.arange(best_angle - coarse_step, best_angle + coarse_step + fine_step, fine_step, dtype=np.float32):
        rot = rotate_image(
            placed_outer_mask,
            angle_deg=float(angle),
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        score = iou_score(rot, template_outer_mask)
        if score > best_score:
            best_score = float(score)
            best_angle = float(angle)

    return best_angle, best_score


def center_crop_with_padding(gray, crop_size=(600, 600), center=None):
    # 지정 중심 기준으로 크롭하고, 경계를 벗어나면 reflect 패딩한다.
    h, w = gray.shape
    cw, ch = crop_size

    if center is None:
        cx = w // 2
        cy = h // 2
    else:
        cx = int(round(center[0]))
        cy = int(round(center[1]))

    x1 = cx - cw // 2
    y1 = cy - ch // 2
    x2 = x1 + cw
    y2 = y1 + ch

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        gray = cv2.copyMakeBorder(
            gray,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    return gray[y1:y2, x1:x2]


def mask_bbox_size(mask):
    # 마스크의 bbox 크기(width, height)를 반환한다.
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    w = int(xs.max() - xs.min() + 1)
    h = int(ys.max() - ys.min() + 1)
    return (w, h)


def scale_about_center(img, center, scale, interpolation, border_mode, border_value=0):
    # 중심을 고정한 채 이미지를 등방 축소/확대한다.
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, 0.0, float(scale))
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )


def overlay_preview(template_bgr, aligned_gray, aligned_mask, alpha=0.45):
    # 템플릿 위에 정렬된 결과를 반투명 오버레이한다.
    proc_bgr = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(template_bgr, 1.0 - alpha, proc_bgr, alpha, 0.0)

    out = template_bgr.copy()
    mask_bool = aligned_mask > 0
    out[mask_bool] = blended[mask_bool]
    return out


def prepare_template(template_path):
    # 템플릿 기준 마스크/홀 정보를 준비한다.
    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise RuntimeError(f"failed_to_read_template: {template_path}")

    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    template_outer_info = extract_main_component(template_gray, foreground="dark")
    if template_outer_info is None:
        raise RuntimeError("template_outer_component_not_found")

    template_section = fill_section_mask(template_outer_info["mask"])
    if template_section is None:
        raise RuntimeError("template_section_mask_not_found")

    template_holes = detect_template_holes(template_gray, template_section)

    return {
        "bgr": template_bgr,
        "gray": template_gray,
        "outer_mask": template_outer_info["mask"],
        "section_mask": template_section,
        "center": template_outer_info["center"],
        "area": template_outer_info["area"],
        "holes": template_holes,
    }


def process_one(
    image_path,
    template_model,
    output_dir,
    overlay_dir,
    crop_size=(600, 600),
    alpha=0.45,
    scale_adjust=1.0,
    save_overlay=True,
):
    # 단일 이미지 전처리: gray -> center/scale align -> hole-aware rotation -> crop.
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "imread_failed"

    gray = to_grayscale(img)

    target_outer_info = extract_target_outer_component_pro1(gray)
    if target_outer_info is None:
        return False, "target_outer_component_not_found"

    target_section = fill_section_mask(target_outer_info["mask"])
    if target_section is None:
        return False, "target_section_mask_not_found"

    target_area = max(float(target_outer_info["area"]), 1.0)
    template_area = max(float(template_model["area"]), 1.0)
    scale = np.sqrt(template_area / target_area) * float(scale_adjust)

    placed_gray, placed_outer, placed_section, dx, dy = place_to_template(
        gray_target=gray,
        outer_mask=target_outer_info["mask"],
        section_mask=target_section,
        template_shape=template_model["gray"].shape,
        template_center=template_model["center"],
        scale=scale,
    )

    target_hole_result = detect_target_holes(placed_gray, placed_section)
    center = template_model["center"]

    if target_hole_result is not None and target_hole_result.get("grouped") is not None:
        rot = find_best_rotation_with_hole_groups(
            target_grouped=target_hole_result["grouped"],
            template_grouped=template_model["holes"],
            center=center,
            coarse_step=3.0,
            fine_step=0.25,
        )
        best_angle = float(rot["angle"])
        metric_msg = (
            f"hole[ score={rot['score']:.3f}, k={target_hole_result['ksize']}, thr={target_hole_result['threshold']:.1f}, "
            f"iou_pair={rot['s_pair']:.3f}, iou_all={rot['s_all']:.3f} ]"
        )
    else:
        best_angle, outer_iou = find_best_rotation_outer_mask(
            placed_outer_mask=placed_outer,
            template_outer_mask=template_model["outer_mask"],
            center=center,
            coarse_step=3.0,
            fine_step=0.25,
        )
        metric_msg = f"fallback[ score={outer_iou:.3f}, iou_outer={outer_iou:.3f} ]"

    aligned_gray = rotate_image(
        placed_gray,
        angle_deg=best_angle,
        center=center,
        interpolation=cv2.INTER_CUBIC,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    aligned_section = rotate_image(
        placed_section,
        angle_deg=best_angle,
        center=center,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )

    # 템플릿은 각도 추정에만 사용하고, 최종 출력은 원본 스케일로 저장한다.
    original_center = target_outer_info["center"]
    rotated_original = rotate_image(
        gray,
        angle_deg=best_angle,
        center=original_center,
        interpolation=cv2.INTER_CUBIC,
        border_value=0,
        border_mode=cv2.BORDER_REFLECT_101,
    )

    rotated_outer_original = rotate_image(
        target_outer_info["mask"],
        angle_deg=best_angle,
        center=original_center,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    rotated_section_original = fill_section_mask(rotated_outer_original)

    # 단면 지름이 crop보다 큰 경우, 중심 고정 축소로 잘림을 방지한다.
    auto_scale = 1.0
    crop_w, crop_h = crop_size
    margin = 14
    if rotated_section_original is not None:
        bbox = mask_bbox_size(rotated_section_original)
        if bbox is not None:
            bw, bh = bbox
            fit_w = (crop_w - 2 * margin) / max(float(bw), 1.0)
            fit_h = (crop_h - 2 * margin) / max(float(bh), 1.0)
            auto_scale = min(1.0, fit_w, fit_h)

            if auto_scale < 0.999:
                rotated_original = scale_about_center(
                    rotated_original,
                    center=original_center,
                    scale=auto_scale,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_REFLECT_101,
                    border_value=0,
                )
                rotated_section_original = scale_about_center(
                    rotated_section_original,
                    center=original_center,
                    scale=auto_scale,
                    interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    border_value=0,
                )

    crop_center = original_center
    if rotated_section_original is not None:
        crop_center = mask_centroid(rotated_section_original)

    cropped = center_crop_with_padding(
        rotated_original,
        crop_size=crop_size,
        center=crop_center,
    )

    out_path = output_dir / image_path.name
    if not cv2.imwrite(str(out_path), cropped):
        return False, "imwrite_processed_failed"

    if save_overlay:
        ov = overlay_preview(
            template_bgr=template_model["bgr"],
            aligned_gray=aligned_gray,
            aligned_mask=aligned_section,
            alpha=alpha,
        )

        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.drawMarker(ov, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        ov_path = overlay_dir / f"{image_path.stem}_overlay.png"
        if not cv2.imwrite(str(ov_path), ov):
            return False, "imwrite_overlay_failed"

    geom_msg = (
        f"geom[ scale={scale:.4f}, shift=({dx:.1f},{dy:.1f}), "
        f"rot={best_angle:.2f}, fit={auto_scale:.3f} ]"
    )

    return (
        True,
        f"\"ok\" {geom_msg}\n{metric_msg}",
    )


def run_preprocess(
    input_dir,
    pattern,
    template_path,
    output_dir,
    overlay_dir,
    crop_size=(600, 600),
    alpha=0.45,
    scale_adjust=1.0,
    save_overlay=True,
):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not template_path.exists():
        raise FileNotFoundError(f"template not found: {template_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files(input_dir, pattern)
    total = len(files)

    template_model = prepare_template(template_path)

    print(f"{total}장 pro1 전처리 시작 (template registration + 2-hole mask align)")
    print(f"input:    {input_dir} ({pattern})")
    print(f"template: {template_path}")
    print(f"output:   {output_dir}")
    if save_overlay:
        print(f"overlay:  {overlay_dir}")

    failures = []
    for idx, path in enumerate(files, start=1):
        ok, msg = process_one(
            image_path=path,
            template_model=template_model,
            output_dir=output_dir,
            overlay_dir=overlay_dir,
            crop_size=crop_size,
            alpha=alpha,
            scale_adjust=scale_adjust,
            save_overlay=save_overlay,
        )

        if ok:
            print(f"[{idx}/{total}] OK   {path.name:<18} \n{msg}")
        else:
            failures.append((path.name, msg))
            print(f"[{idx}/{total}] FAIL {path.name:<18} \n{msg}")

    if failures:
        print(f"완료 (실패 {len(failures)}건)")
        for name, msg in failures[:10]:
            print(f"  - {name}: {msg}")
    else:
        print("완료! 모든 이미지를 템플릿 기준으로 정렬했습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="pro1 전처리: 1) Gray -> 2) Template Registration(2-hole 중심+회전) -> 3) Crop"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_1"),
        help="입력 폴더",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.BMP,*.bmp,*.png",
        help="파일 패턴(콤마 구분 예: *.BMP,*.png)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/raw/label_orign/pro_1_endpoint.png"),
        help="기준 템플릿 이미지",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro1_holealign"),
        help="정렬+크롭 결과 저장 폴더",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro1_holealign"),
        help="템플릿 오버레이 결과 저장 폴더",
    )
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        default=(600, 600),
        metavar=("W", "H"),
        help="최종 크롭 크기",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="오버레이 투명도 (0~1)",
    )
    parser.add_argument(
        "--scale-adjust",
        type=float,
        default=1.0,
        help="자동 스케일 보정 배율",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="오버레이 결과 저장을 비활성화",
    )

    args = parser.parse_args()
    run_preprocess(
        input_dir=args.input_dir,
        pattern=args.pattern,
        template_path=args.template,
        output_dir=args.output_dir,
        overlay_dir=args.overlay_dir,
        crop_size=tuple(args.crop_size),
        alpha=args.alpha,
        scale_adjust=args.scale_adjust,
        save_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
