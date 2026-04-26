import argparse
from pathlib import Path

import cv2
import numpy as np


SOURCE_DIR = Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_2")
TEMPLATE_RESULT_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_templates/pro2_endpoint")
OUTPUT_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro2_holealign")
OVERLAY_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro2_holealign")
STAGE_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/stage_images/pro2_holealign")


############################## 단계별 결과 이미지 저장 함수 ##############################
def save_stage_image(output_dir, index, stage_name, image):

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{index:02d}_{stage_name}.png"
    ok = cv2.imwrite(str(out_path), image)

    if not ok:
        raise RuntimeError(f"failed_to_write_image: {out_path}")

    return out_path


############################## 선택적 단계별 결과 저장 함수 ##############################
def save_optional_stage(stage_root_dir, item_name, index, stage_name, image):

    if stage_root_dir is None:
        return None

    return save_stage_image(stage_root_dir / item_name, index, stage_name, image)


############################## gray scale 함수 ##############################
def bgr_to_grayscale(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img


############################## 마스크 중심 계산 함수 ##############################
def mask_centroid(mask):

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        h, w = mask.shape
        return (float(w) * 0.5, float(h) * 0.5)

    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


############################## 외곽 마스크를 채워 section mask를 만드는 함수 ##############################
def fill_section_mask(outer_mask):

    contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(outer_mask, dtype=np.uint8)
    cv2.drawContours(filled, [main_contour], -1, 255, thickness=cv2.FILLED)

    return filled


############################## 대표 외곽 성분 추출 함수 ##############################
def extract_main_component(gray, foreground="bright"):

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
        min_area = max(100, int(h * w * 1e-4))
        if area < min_area:
            continue

        centroid = centroids[idx].astype(np.float32)
        dist = float(np.linalg.norm(centroid - img_center) / diag)
        score = float(area) * (1.5 - min(dist, 1.2))

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[labels == best_idx] = 255

    area = int(stats[best_idx, cv2.CC_STAT_AREA])
    cx, cy = centroids[best_idx]

    return {
        "mask": mask,
        "center": (float(cx), float(cy)),
        "area": area,
    }


############################## source/template를 template 좌표계에 배치하는 함수 ##############################
def place_to_template(gray_source, outer_mask, section_mask, template_shape, template_center, scale):

    h_t, w_t = template_shape

    scaled_gray = cv2.resize(gray_source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
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


############################## lumen 후보 성분 1차 필터링 함수 ##############################
def extract_lumen_components(binary_mask, section_area, min_ratio, max_ratio, min_circularity):

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

        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
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


############################## 붙은 lumen 후보를 2개로 분리하는 함수 ##############################
def split_component_into_two(component, shape):

    comp_mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(comp_mask, [component["contour"]], -1, 255, thickness=cv2.FILLED)

    ys, xs = np.where(comp_mask > 0)
    if xs.size < 200:
        return None

    data = xs.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, _ = cv2.kmeans(
        data=data,
        K=2,
        bestLabels=None,
        criteria=criteria,
        attempts=5,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    labels = labels.reshape(-1)
    split_components = []

    for idx in [0, 1]:
        cluster_mask = np.zeros(shape, dtype=np.uint8)
        sel = labels == idx
        cluster_mask[ys[sel], xs[sel]] = 255
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue

        peri = float(cv2.arcLength(cnt, True))
        circularity = 0.0 if peri == 0 else float(4.0 * np.pi * area / (peri * peri))
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
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

    return split_components


############################## 2개 lumen 검출 시 3개로 복원 시도하는 함수 ##############################
def augment_components_if_merged(components, shape, section_area, min_ratio, max_ratio, min_circularity):

    # pro2는 큰 홀 1개 + 작은 홀 2개 구조라서, 2개만 검출되면 큰 성분 분할을 시도한다.
    if len(components) != 2:
        return components

    split = split_component_into_two(components[0], shape)
    if split is None:
        return components

    merged = [components[1]] + split
    min_area = float(section_area) * float(min_ratio)
    max_area = float(section_area) * float(max_ratio)

    filtered = []
    for comp in merged:
        if not (min_area <= comp["area"] <= max_area):
            continue
        if comp["circularity"] < float(min_circularity) * 0.90:
            continue
        filtered.append(comp)

    filtered.sort(key=lambda x: x["area"], reverse=True)
    return filtered


############################## 3개 lumen을 big/small/all로 그룹화하는 함수 ##############################
def build_hole_group_masks(shape, components):

    top3 = sorted(components, key=lambda x: x["area"], reverse=True)[:3]
    if len(top3) < 3:
        return None

    big = top3[0]
    small = top3[1:3]

    big_mask = np.zeros(shape, dtype=np.uint8)
    small_mask = np.zeros(shape, dtype=np.uint8)
    all_mask = np.zeros(shape, dtype=np.uint8)

    cv2.drawContours(big_mask, [big["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(small_mask, [small[0]["contour"], small[1]["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(
        all_mask,
        [big["contour"], small[0]["contour"], small[1]["contour"]],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return {
        "top3": top3,
        "big": big,
        "small": small,
        "big_mask": big_mask,
        "small_mask": small_mask,
        "all_mask": all_mask,
        "small_mean": float((small[0]["area"] + small[1]["area"]) * 0.5),
    }


############################## pro2 source lumen 검출 함수 ##############################
def detect_source_lumens(source_gray, source_section_mask):

    sec_area = max(float(np.count_nonzero(source_section_mask)), 1.0)
    inner_section = cv2.erode(source_section_mask, np.ones((11, 11), np.uint8), iterations=1)

    best = None
    for ksize in [31, 25, 35]:
        blurred = cv2.GaussianBlur(source_gray, (ksize, ksize), 0)
        values = blurred[inner_section > 0]
        if values.size < 100:
            continue

        otsu_thr, _ = cv2.threshold(values.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        for offset in [0, 5, -5, 10, -10]:
            thr = float(np.clip(otsu_thr + offset, 0, 255))

            candidate = np.zeros_like(source_gray, dtype=np.uint8)
            candidate[(blurred <= thr) & (inner_section > 0)] = 255
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

            components = extract_lumen_components(
                binary_mask=candidate,
                section_area=sec_area,
                min_ratio=0.003,
                max_ratio=0.45,
                min_circularity=0.20,
            )
            components = augment_components_if_merged(
                components=components,
                shape=source_gray.shape,
                section_area=sec_area,
                min_ratio=0.003,
                max_ratio=0.45,
                min_circularity=0.20,
            )

            grouped = build_hole_group_masks(source_gray.shape, components)
            count_top = min(len(components), 3)
            if count_top == 0:
                continue

            top = components[:count_top]
            smallest = float(top[-1]["area"])
            mean_circularity = float(np.mean([row["circularity"] for row in top]))
            gap = 0.0
            if grouped is not None:
                gap = float(grouped["big"]["area"] / max(grouped["small_mean"], 1.0))

            # pro2는 3개 복원 성공을 최우선으로 두고, 작은 홀 유지와 분리도 안정성을 함께 본다.
            score = count_top * 10000.0 + smallest + mean_circularity * 100.0 + gap * 50.0

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "ksize": ksize,
                    "threshold": thr,
                    "components": components,
                    "grouped": grouped,
                }

    if best is None:
        return None

    return best


############################## 두 마스크의 IoU 점수 계산 함수 ##############################
def iou_score(mask_a, mask_b):

    a = mask_a > 0
    b = mask_b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0

    return float(inter / union)


############################## 이미지 회전 함수 ##############################
def rotate_image(img, angle_deg, center, interpolation, border_value=0, border_mode=cv2.BORDER_CONSTANT):

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


############################## 지정 중심 기준 크롭 함수 ##############################
def center_crop_with_padding(gray, crop_size=(600, 600), center=None):

    h, w = gray.shape
    crop_w, crop_h = crop_size

    if center is None:
        cx = w // 2
        cy = h // 2
    else:
        cx = int(round(center[0]))
        cy = int(round(center[1]))

    x1 = cx - crop_w // 2
    y1 = cy - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

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


############################## 중심 고정 등방 스케일 함수 ##############################
def scale_about_center(img, center, scale, interpolation, border_mode, border_value=0):

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


############################## 마스크 bbox 크기 계산 함수 ##############################
def mask_bbox_size(mask):

    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None

    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return (width, height)


############################## 오버레이 미리보기 생성 함수 ##############################
def overlay_preview(template_gray, aligned_gray, aligned_mask, alpha=0.45):

    template_bgr = cv2.cvtColor(template_gray, cv2.COLOR_GRAY2BGR)
    proc_bgr = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(template_bgr, 1.0 - alpha, proc_bgr, alpha, 0.0)

    out = template_bgr.copy()
    mask_bool = aligned_mask > 0
    out[mask_bool] = blended[mask_bool]

    return out


############################## 템플릿/정렬 중심 마커 그리기 함수 ##############################
def draw_alignment_centers(overlay_img, template_center, aligned_mask_center):

    out = overlay_img.copy()
    tx, ty = int(round(template_center[0])), int(round(template_center[1]))
    ax, ay = int(round(aligned_mask_center[0])), int(round(aligned_mask_center[1]))

    cv2.drawMarker(out, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.drawMarker(out, (ax, ay), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)

    return out


############################## hole 그룹 기반 최적 회전 탐색 함수 ##############################
def find_best_rotation_with_hole_groups(source_grouped, template_grouped, center, coarse_step=3.0, fine_step=0.25):

    source_big = source_grouped["big_mask"]
    source_small = source_grouped["small_mask"]
    source_all = source_grouped["all_mask"]

    template_big = template_grouped["big_mask"]
    template_small = template_grouped["small_mask"]
    template_all = template_grouped["all_mask"]

    def score_at(angle):
        rotated_big = rotate_image(
            source_big,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        rotated_small = rotate_image(
            source_small,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        rotated_all = rotate_image(
            source_all,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )

        score_big = iou_score(rotated_big, template_big)
        score_small = iou_score(rotated_small, template_small)
        score_all = iou_score(rotated_all, template_all)
        score = 0.60 * score_big + 0.25 * score_small + 0.15 * score_all

        return score, score_big, score_small, score_all

    best = {"angle": 0.0, "score": -1.0, "s_big": 0.0, "s_small": 0.0, "s_all": 0.0}

    for angle in np.arange(-180.0, 180.0, coarse_step, dtype=np.float32):
        score, score_big, score_small, score_all = score_at(float(angle))
        if score > best["score"]:
            best = {
                "angle": float(angle),
                "score": float(score),
                "s_big": float(score_big),
                "s_small": float(score_small),
                "s_all": float(score_all),
            }

    for angle in np.arange(best["angle"] - coarse_step, best["angle"] + coarse_step + fine_step, fine_step, dtype=np.float32):
        score, score_big, score_small, score_all = score_at(float(angle))
        if score > best["score"]:
            best = {
                "angle": float(angle),
                "score": float(score),
                "s_big": float(score_big),
                "s_small": float(score_small),
                "s_all": float(score_all),
            }

    return best


############################## 외곽 마스크 기반 fallback 회전 탐색 함수 ##############################
def find_best_rotation_outer_mask(placed_outer_mask, template_outer_mask, center, coarse_step=3.0, fine_step=0.25):

    best_angle = 0.0
    best_score = -1.0

    for angle in np.arange(-180.0, 180.0, coarse_step, dtype=np.float32):
        rotated = rotate_image(
            placed_outer_mask,
            angle_deg=float(angle),
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        score = iou_score(rotated, template_outer_mask)
        if score > best_score:
            best_score = float(score)
            best_angle = float(angle)

    for angle in np.arange(best_angle - coarse_step, best_angle + coarse_step + fine_step, fine_step, dtype=np.float32):
        rotated = rotate_image(
            placed_outer_mask,
            angle_deg=float(angle),
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        score = iou_score(rotated, template_outer_mask)
        if score > best_score:
            best_score = float(score)
            best_angle = float(angle)

    return best_angle, best_score


############################## 템플릿 기준 정보 준비 함수 ##############################
def prepare_template_model(template_result_dir, stage_root_dir=None):

    template_gray_path = template_result_dir / "01_pro2_endpoint_grayscale.png"
    template_outer_path = template_result_dir / "04_pro2_endpoint_outer_mask.png"
    template_section_path = template_result_dir / "05_pro2_endpoint_section_mask.png"
    template_big_path = template_result_dir / "06_pro2_endpoint_lumen_big_mask.png"
    template_small_path = template_result_dir / "07_pro2_endpoint_lumen_small_mask.png"
    template_all_path = template_result_dir / "08_pro2_endpoint_lumen_all_mask.png"

    template_gray = cv2.imread(str(template_gray_path), cv2.IMREAD_GRAYSCALE)
    if template_gray is None:
        raise RuntimeError(f"failed_to_read_template_gray: {template_gray_path}")

    template_outer_mask = cv2.imread(str(template_outer_path), cv2.IMREAD_GRAYSCALE)
    if template_outer_mask is None:
        raise RuntimeError(f"failed_to_read_template_outer_mask: {template_outer_path}")

    template_section_mask = cv2.imread(str(template_section_path), cv2.IMREAD_GRAYSCALE)
    if template_section_mask is None:
        raise RuntimeError(f"failed_to_read_template_section_mask: {template_section_path}")

    template_big_mask = cv2.imread(str(template_big_path), cv2.IMREAD_GRAYSCALE)
    if template_big_mask is None:
        raise RuntimeError(f"failed_to_read_template_lumen_big_mask: {template_big_path}")

    template_small_mask = cv2.imread(str(template_small_path), cv2.IMREAD_GRAYSCALE)
    if template_small_mask is None:
        raise RuntimeError(f"failed_to_read_template_lumen_small_mask: {template_small_path}")

    template_all_mask = cv2.imread(str(template_all_path), cv2.IMREAD_GRAYSCALE)
    if template_all_mask is None:
        raise RuntimeError(f"failed_to_read_template_lumen_all_mask: {template_all_path}")

    save_optional_stage(stage_root_dir, "template", 1, "pro2_endpoint_grayscale", template_gray)
    save_optional_stage(stage_root_dir, "template", 2, "pro2_endpoint_outer_mask", template_outer_mask)
    save_optional_stage(stage_root_dir, "template", 3, "pro2_endpoint_section_mask", template_section_mask)
    save_optional_stage(stage_root_dir, "template", 4, "pro2_endpoint_lumen_big_mask", template_big_mask)
    save_optional_stage(stage_root_dir, "template", 5, "pro2_endpoint_lumen_small_mask", template_small_mask)
    save_optional_stage(stage_root_dir, "template", 6, "pro2_endpoint_lumen_all_mask", template_all_mask)

    return {
        "gray": template_gray,
        "outer_mask": template_outer_mask,
        "section_mask": template_section_mask,
        "center": mask_centroid(template_section_mask),
        "area": int(np.count_nonzero(template_outer_mask)),
        "holes": {
            "big_mask": template_big_mask,
            "small_mask": template_small_mask,
            "all_mask": template_all_mask,
        },
    }


############################## source를 template 좌표계에 배치하는 함수 ##############################
def place_source_to_template(source_gray, template_model, scale_adjust=1.0):

    source_outer_info = extract_main_component(source_gray, foreground="bright")
    if source_outer_info is None:
        raise RuntimeError("failed_to_extract_source_outer_component")

    source_section_mask = fill_section_mask(source_outer_info["mask"])
    if source_section_mask is None:
        raise RuntimeError("failed_to_build_source_section_mask")

    source_area = max(float(source_outer_info["area"]), 1.0)
    template_area = max(float(template_model["area"]), 1.0)
    scale = np.sqrt(template_area / source_area) * float(scale_adjust)

    placed_gray, placed_outer, placed_section, dx, dy = place_to_template(
        gray_source=source_gray,
        outer_mask=source_outer_info["mask"],
        section_mask=source_section_mask,
        template_shape=template_model["gray"].shape,
        template_center=template_model["center"],
        scale=scale,
    )

    return {
        "source_outer_mask": source_outer_info["mask"],
        "source_section_mask": source_section_mask,
        "source_center": source_outer_info["center"],
        "placed_gray": placed_gray,
        "placed_outer_mask": placed_outer,
        "placed_section_mask": placed_section,
        "scale": scale,
        "dx": dx,
        "dy": dy,
    }


############################## source 이미지 전처리 파이프라인 ##############################
def process_source(
    source_path,
    template_model,
    output_dir,
    overlay_dir,
    stage_root_dir=None,
    crop_size=(600, 600),
    alpha=0.45,
    scale_adjust=1.0,
    save_overlay=True,
):

    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        return False, "imread_failed"

    source_gray = bgr_to_grayscale(source_bgr)
    save_optional_stage(stage_root_dir, source_path.stem, 1, f"{source_path.stem}_grayscale", source_gray)

    try:
        placed_result = place_source_to_template(source_gray, template_model, scale_adjust=scale_adjust)
    except RuntimeError as exc:
        return False, str(exc)

    save_optional_stage(stage_root_dir, source_path.stem, 2, f"{source_path.stem}_outer_mask", placed_result["source_outer_mask"])
    save_optional_stage(stage_root_dir, source_path.stem, 3, f"{source_path.stem}_section_mask", placed_result["source_section_mask"])
    save_optional_stage(stage_root_dir, source_path.stem, 4, f"{source_path.stem}_placed_gray", placed_result["placed_gray"])
    save_optional_stage(stage_root_dir, source_path.stem, 5, f"{source_path.stem}_placed_outer_mask", placed_result["placed_outer_mask"])
    save_optional_stage(stage_root_dir, source_path.stem, 6, f"{source_path.stem}_placed_section_mask", placed_result["placed_section_mask"])

    source_lumen_result = detect_source_lumens(
        placed_result["placed_gray"],
        placed_result["placed_section_mask"],
    )

    source_grouped = None
    if source_lumen_result is not None:
        source_grouped = source_lumen_result.get("grouped")

    if source_grouped is not None:
        save_optional_stage(stage_root_dir, source_path.stem, 7, f"{source_path.stem}_lumen_big_mask", source_grouped["big_mask"])
        save_optional_stage(stage_root_dir, source_path.stem, 8, f"{source_path.stem}_lumen_small_mask", source_grouped["small_mask"])
        save_optional_stage(stage_root_dir, source_path.stem, 9, f"{source_path.stem}_lumen_all_mask", source_grouped["all_mask"])

        rot = find_best_rotation_with_hole_groups(
            source_grouped=source_grouped,
            template_grouped=template_model["holes"],
            center=template_model["center"],
            coarse_step=3.0,
            fine_step=0.25,
        )
        best_angle = float(rot["angle"])
        metric_msg = (
            f"hole[ score={rot['score']:.3f}, k={source_lumen_result['ksize']}, thr={source_lumen_result['threshold']:.1f}, "
            f"iou_big={rot['s_big']:.3f}, iou_small={rot['s_small']:.3f}, iou_all={rot['s_all']:.3f} ]"
        )
    else:
        best_angle, outer_iou = find_best_rotation_outer_mask(
            placed_outer_mask=placed_result["placed_outer_mask"],
            template_outer_mask=template_model["outer_mask"],
            center=template_model["center"],
            coarse_step=3.0,
            fine_step=0.25,
        )
        metric_msg = f"fallback[ score={outer_iou:.3f}, iou_outer={outer_iou:.3f} ]"

    aligned_gray = rotate_image(
        placed_result["placed_gray"],
        angle_deg=best_angle,
        center=template_model["center"],
        interpolation=cv2.INTER_CUBIC,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    save_optional_stage(stage_root_dir, source_path.stem, 10, f"{source_path.stem}_aligned_gray", aligned_gray)

    aligned_outer = rotate_image(
        placed_result["placed_outer_mask"],
        angle_deg=best_angle,
        center=template_model["center"],
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    save_optional_stage(stage_root_dir, source_path.stem, 11, f"{source_path.stem}_aligned_outer_mask", aligned_outer)

    aligned_section = rotate_image(
        placed_result["placed_section_mask"],
        angle_deg=best_angle,
        center=template_model["center"],
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    save_optional_stage(stage_root_dir, source_path.stem, 12, f"{source_path.stem}_aligned_section_mask", aligned_section)

    if source_grouped is not None:
        aligned_big = rotate_image(
            source_grouped["big_mask"],
            angle_deg=best_angle,
            center=template_model["center"],
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        aligned_small = rotate_image(
            source_grouped["small_mask"],
            angle_deg=best_angle,
            center=template_model["center"],
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        aligned_all = rotate_image(
            source_grouped["all_mask"],
            angle_deg=best_angle,
            center=template_model["center"],
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )

        save_optional_stage(stage_root_dir, source_path.stem, 13, f"{source_path.stem}_aligned_lumen_big_mask", aligned_big)
        save_optional_stage(stage_root_dir, source_path.stem, 14, f"{source_path.stem}_aligned_lumen_small_mask", aligned_small)
        save_optional_stage(stage_root_dir, source_path.stem, 15, f"{source_path.stem}_aligned_lumen_all_mask", aligned_all)

    # pro2는 템플릿 좌표계는 각도 추정에만 쓰고, 최종 출력은 원본 해상도 좌표계에서 회전/크롭한다.
    original_center = placed_result["source_center"]
    rotated_original = rotate_image(
        source_gray,
        angle_deg=best_angle,
        center=original_center,
        interpolation=cv2.INTER_CUBIC,
        border_value=0,
        border_mode=cv2.BORDER_REFLECT_101,
    )
    rotated_outer_original = rotate_image(
        placed_result["source_outer_mask"],
        angle_deg=best_angle,
        center=original_center,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    rotated_section_original = fill_section_mask(rotated_outer_original)
    if rotated_section_original is None:
        return False, "failed_to_build_rotated_section_mask"

    auto_scale = 1.0
    margin = 14
    crop_center = original_center

    bbox = mask_bbox_size(rotated_section_original)
    if bbox is not None:
        bbox_w, bbox_h = bbox
        fit_w = (crop_size[0] - 2 * margin) / max(float(bbox_w), 1.0)
        fit_h = (crop_size[1] - 2 * margin) / max(float(bbox_h), 1.0)
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

    crop_center = mask_centroid(rotated_section_original)

    final_crop = center_crop_with_padding(
        rotated_original,
        crop_size=crop_size,
        center=crop_center,
    )
    save_optional_stage(stage_root_dir, source_path.stem, 16, f"{source_path.stem}_final_crop_{crop_size[0]}", final_crop)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / source_path.name
    if not cv2.imwrite(str(final_output_path), final_crop):
        return False, f"failed_to_write_final_output: {final_output_path}"

    if save_overlay:
        overlay = overlay_preview(
            template_model["gray"],
            aligned_gray,
            aligned_section,
            alpha=alpha,
        )
        overlay = draw_alignment_centers(
            overlay,
            template_center=template_model["center"],
            aligned_mask_center=mask_centroid(aligned_section),
        )
        save_optional_stage(stage_root_dir, source_path.stem, 17, f"{source_path.stem}_template_overlay", overlay)

        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay_output_path = overlay_dir / f"{source_path.stem}_overlay.png"
        if not cv2.imwrite(str(overlay_output_path), overlay):
            return False, f"failed_to_write_overlay_output: {overlay_output_path}"

    geom_msg = (
        f"geom[ scale={placed_result['scale']:.4f}, shift=({placed_result['dx']:.1f},{placed_result['dy']:.1f}), "
        f"rot={best_angle:.2f}, fit={auto_scale:.3f} ]"
    )

    return True, f"ok {geom_msg}\n{metric_msg}"


############################## 전체 source 이미지 전처리 실행 함수 ##############################
def run_preprocess(
    input_dir,
    pattern,
    template_result_dir,
    output_dir,
    overlay_dir,
    crop_size=(600, 600),
    alpha=0.45,
    scale_adjust=1.0,
    save_overlay=True,
    stage_dir=None,
    save_stage_images=False,
):

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not template_result_dir.exists():
        raise FileNotFoundError(f"template result dir not found: {template_result_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if save_stage_images and stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(input_dir.glob(pattern))
    total = len(source_files)
    if total == 0:
        raise FileNotFoundError(f"no_source_files: {input_dir} ({pattern})")

    template_model = prepare_template_model(
        template_result_dir,
        stage_root_dir=stage_dir if save_stage_images else None,
    )

    print("=============== 전처리 시작 (template registration + 3-lumen split) ===============")
    print(f"이미지 개수:  {total}장")
    print("카테터 타입:  pro2 (3-lumen)")
    print(f"input:     {input_dir} ({pattern})")
    print(f"template:  {template_result_dir}")
    print(f"output:    {output_dir}")
    if save_overlay:
        print(f"overlay:   {overlay_dir}")
    if save_stage_images and stage_dir is not None:
        print(f"stages:    {stage_dir}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    failures = []
    for idx, source_path in enumerate(source_files, start=1):
        ok, msg = process_source(
            source_path=source_path,
            template_model=template_model,
            output_dir=output_dir,
            overlay_dir=overlay_dir,
            stage_root_dir=stage_dir if save_stage_images else None,
            crop_size=crop_size,
            alpha=alpha,
            scale_adjust=scale_adjust,
            save_overlay=save_overlay,
        )

        if ok:
            print(f"[{idx}/{total}] OK    {source_path.name:<18}")
            print(f"{msg}\n")
        else:
            failures.append((source_path.name, msg))
            print(f"[{idx}/{total}] FAIL  {source_path.name:<18}")
            print(f"{msg}\n")

    print("==================================== 전처리 완료 ===================================")
    if failures:
        print(f"완료 (실패 {len(failures)} 건)")
        for name, msg in failures[:10]:
            print(f"  - {name}: {msg}")
    else:
        print("완료! 모든 이미지를 템플릿 기준으로 정렬했습니다.")


def main():

    parser = argparse.ArgumentParser(
        description="pro2 전처리: 1) Gray scale -> 2) Template Registration(3-lumen 중심+회전) -> 3) Crop"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SOURCE_DIR,
        help="input data 폴더 경로",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="파일 확장자(e.g. *.png, *.BMP)",
    )
    parser.add_argument(
        "--template-result-dir",
        type=Path,
        default=TEMPLATE_RESULT_DIR,
        help="전처리된 pro2 템플릿 결과 폴더 경로",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="전처리 결과 저장 폴더 경로",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=OVERLAY_DIR,
        help="템플릿 오버레이 결과 저장 폴더 경로",
    )
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        default=(600, 600),
        metavar=("W", "H"),
        help="최종 크롭 사이즈",
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
        help="오버레이 결과 저장 비활성화",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=STAGE_DIR,
        help="전처리 중간 결과 이미지 저장 폴더 경로",
    )
    parser.add_argument(
        "--no-stage-images",
        action="store_true",
        help="전처리 중간 결과 이미지 저장 비활성화",
    )

    args = parser.parse_args()

    run_preprocess(
        input_dir=args.input_dir,
        pattern=args.pattern,
        template_result_dir=args.template_result_dir,
        output_dir=args.output_dir,
        overlay_dir=args.overlay_dir,
        crop_size=tuple(args.crop_size),
        alpha=args.alpha,
        scale_adjust=args.scale_adjust,
        save_overlay=not args.no_overlay,
        stage_dir=args.stage_dir,
        save_stage_images=not args.no_stage_images,
    )


if __name__ == "__main__":
    main()
