import argparse
from pathlib import Path

import cv2
import numpy as np
from preprocess_metrics import make_metrics_row, normalize_error_value, write_metrics_csv


SOURCE_DIR = Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_1")
TEMPLATE_RESULT_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_templates/pro1_endpoint")
OUTPUT_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro1_holealign")
OVERLAY_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro1_holealign")
STAGE_DIR = Path("/home/hjj747/catheter-defect-inspection/data/processed/stage_images/pro1_holealign")


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


############################## 입력 파일 목록 수집 함수 ##############################
def collect_files(input_dir, pattern):

    patterns = [item.strip() for item in str(pattern).split(",") if item.strip()]
    if not patterns:
        patterns = ["*.BMP"]

    merged = []
    for item in patterns:
        merged.extend(input_dir.glob(item))

    files = sorted({path.resolve() for path in merged if path.is_file()})
    return [Path(path) for path in files]


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


############################## pro1 전용 밝은 outer 후보 추출 함수 ##############################
def extract_source_outer_component_pro1(gray):

    h, w = gray.shape
    area_total = float(h * w)
    img_center = np.array([w * 0.5, h * 0.5], dtype=np.float32)
    diag = max(float(np.hypot(w, h)), 1.0)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    best = None

    for threshold in [245, 240, 235, 230, 225, 220, 215, 210, 205, 200]:
        _, binary = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            if contour_area < area_total * 0.001 or contour_area > area_total * 0.35:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            circularity = 0.0 if perimeter == 0 else float(4.0 * np.pi * contour_area / (perimeter * perimeter))

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio = float(bw / max(bh, 1))
            aspect_score = 1.0 - min(abs(np.log(max(aspect_ratio, 1e-6))), 1.0)

            fill_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(fill_mask, [contour], -1, 255, thickness=cv2.FILLED)
            fill_area = float(np.count_nonzero(fill_mask))
            if fill_area <= 0:
                continue

            component_mask = np.zeros_like(gray, dtype=np.uint8)
            component_mask[(binary > 0) & (fill_mask > 0)] = 255
            bright_area = float(np.count_nonzero(component_mask))
            fill_ratio = bright_area / fill_area

            points = gray[fill_mask > 0]
            mean_intensity = float(points.mean()) if points.size > 0 else 0.0

            moments = cv2.moments(fill_mask)
            if moments["m00"] == 0:
                cx = float(x + bw * 0.5)
                cy = float(y + bh * 0.5)
            else:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])

            dist = float(np.linalg.norm(np.array([cx, cy], dtype=np.float32) - img_center) / diag)
            score = (
                3.0 * min(max(circularity, 0.0), 1.0)
                + 2.5 * min(max(fill_ratio, 0.0), 1.0)
                + 1.0 * min(max((mean_intensity - 120.0) / 135.0, 0.0), 1.0)
                + 1.0 * (1.0 - min(dist * 1.2, 1.0))
                + 0.5 * max(aspect_score, 0.0)
            )

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "mask": component_mask,
                    "center": (cx, cy),
                    "area": int(bright_area),
                }

    if best is not None:
        return {
            "mask": best["mask"],
            "center": best["center"],
            "area": best["area"],
        }

    return extract_main_component(gray, foreground="bright")


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

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter == 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < float(min_circularity):
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        rows.append(
            {
                "contour": contour,
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

    data = np.stack([xs, ys], axis=1).astype(np.float32)
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
        selected = labels == idx
        cluster_mask[ys[selected], xs[selected]] = 255
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        circularity = 0.0 if perimeter == 0 else float(4.0 * np.pi * area / (perimeter * perimeter))
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        split_components.append(
            {
                "contour": contour,
                "area": area,
                "circularity": circularity,
                "center": (cx, cy),
            }
        )

    if len(split_components) != 2:
        return None

    split_components.sort(key=lambda x: x["area"], reverse=True)
    return split_components


############################## 1개 lumen 검출 시 2개로 복원 시도하는 함수 ##############################
def augment_components_if_merged(components, shape, section_area, min_ratio, max_ratio, min_circularity):

    if len(components) != 1:
        return components

    split = split_component_into_two(components[0], shape)
    if split is None:
        return components

    min_area = float(section_area) * float(min_ratio)
    max_area = float(section_area) * float(max_ratio)

    filtered = []
    for component in split:
        if not (min_area <= component["area"] <= max_area):
            continue
        if component["circularity"] < float(min_circularity) * 0.85:
            continue
        filtered.append(component)

    filtered.sort(key=lambda x: x["area"], reverse=True)
    if len(filtered) == 2:
        return filtered

    return components


############################## 2개 lumen을 hole1/hole2/all로 그룹화하는 함수 ##############################
def build_hole_group_masks(shape, components):

    top2 = sorted(components, key=lambda x: x["area"], reverse=True)[:2]
    if len(top2) < 2:
        return None

    # pro1은 두 hole 크기가 비슷하므로 위/아래 순서로 라벨을 고정한다.
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


############################## pro1 source lumen 검출 함수 ##############################
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

        for offset in [0, 5, -5, 10, -10, 15, -15]:
            thr = float(np.clip(otsu_thr + offset, 0, 255))

            candidate = np.zeros_like(source_gray, dtype=np.uint8)
            candidate[(blurred <= thr) & (inner_section > 0)] = 255
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

            components = extract_lumen_components(
                binary_mask=candidate,
                section_area=sec_area,
                min_ratio=0.004,
                max_ratio=0.50,
                min_circularity=0.08,
            )
            components = augment_components_if_merged(
                components=components,
                shape=source_gray.shape,
                section_area=sec_area,
                min_ratio=0.004,
                max_ratio=0.50,
                min_circularity=0.08,
            )

            grouped = build_hole_group_masks(source_gray.shape, components)
            count_top = min(len(components), 2)
            if count_top == 0:
                continue

            top = components[:count_top]
            smallest = float(top[-1]["area"])
            mean_circularity = float(np.mean([row["circularity"] for row in top]))
            similarity = 0.0
            if grouped is not None:
                similarity = float(grouped["similarity"])

            # pro1은 2개 복원을 우선하고, 두 hole의 균형과 원형도를 함께 본다.
            score = count_top * 10000.0 + smallest + mean_circularity * 100.0 + similarity * 300.0

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


############################## RMSE 계산 함수 ##############################
def rmse_error(mask_a, mask_b):

    a = (mask_a > 0).astype(np.float32)
    b = (mask_b > 0).astype(np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


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

    source_hole1 = source_grouped["hole1_mask"]
    source_hole2 = source_grouped["hole2_mask"]
    source_all = source_grouped["all_mask"]

    template_hole1 = template_grouped["hole1_mask"]
    template_hole2 = template_grouped["hole2_mask"]
    template_all = template_grouped["all_mask"]

    def score_at(angle):
        rotated_hole1 = rotate_image(
            source_hole1,
            angle_deg=angle,
            center=center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        rotated_hole2 = rotate_image(
            source_hole2,
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

        # pro1은 hole 두 개 크기가 비슷하므로 라벨이 뒤바뀐 경우도 같이 평가한다.
        score_11 = iou_score(rotated_hole1, template_hole1)
        score_22 = iou_score(rotated_hole2, template_hole2)
        score_12 = iou_score(rotated_hole1, template_hole2)
        score_21 = iou_score(rotated_hole2, template_hole1)

        pair_score_a = 0.5 * (score_11 + score_22)
        pair_score_b = 0.5 * (score_12 + score_21)
        pair_score = max(pair_score_a, pair_score_b)
        all_score = iou_score(rotated_all, template_all)
        score = 0.75 * pair_score + 0.25 * all_score

        return score, pair_score, all_score

    best = {"angle": 0.0, "score": -1.0, "s_pair": 0.0, "s_all": 0.0}

    for angle in np.arange(-180.0, 180.0, coarse_step, dtype=np.float32):
        score, pair_score, all_score = score_at(float(angle))
        if score > best["score"]:
            best = {
                "angle": float(angle),
                "score": float(score),
                "s_pair": float(pair_score),
                "s_all": float(all_score),
            }

    for angle in np.arange(best["angle"] - coarse_step, best["angle"] + coarse_step + fine_step, fine_step, dtype=np.float32):
        score, pair_score, all_score = score_at(float(angle))
        if score > best["score"]:
            best = {
                "angle": float(angle),
                "score": float(score),
                "s_pair": float(pair_score),
                "s_all": float(all_score),
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

    template_gray_path = template_result_dir / "01_pro1_endpoint_grayscale.png"
    template_outer_path = template_result_dir / "04_pro1_endpoint_outer_mask.png"
    template_section_path = template_result_dir / "05_pro1_endpoint_section_mask.png"
    template_hole1_path = template_result_dir / "06_pro1_endpoint_lumen_hole1_mask.png"
    template_hole2_path = template_result_dir / "07_pro1_endpoint_lumen_hole2_mask.png"
    template_all_path = template_result_dir / "08_pro1_endpoint_lumen_all_mask.png"

    template_gray = cv2.imread(str(template_gray_path), cv2.IMREAD_GRAYSCALE)
    if template_gray is None:
        raise RuntimeError(f"failed_to_read_template_gray: {template_gray_path}")

    template_outer_mask = cv2.imread(str(template_outer_path), cv2.IMREAD_GRAYSCALE)
    if template_outer_mask is None:
        raise RuntimeError(f"failed_to_read_template_outer_mask: {template_outer_path}")

    template_section_mask = cv2.imread(str(template_section_path), cv2.IMREAD_GRAYSCALE)
    if template_section_mask is None:
        raise RuntimeError(f"failed_to_read_template_section_mask: {template_section_path}")

    template_hole1_mask = cv2.imread(str(template_hole1_path), cv2.IMREAD_GRAYSCALE)
    if template_hole1_mask is None:
        raise RuntimeError(f"failed_to_read_template_hole1_mask: {template_hole1_path}")

    template_hole2_mask = cv2.imread(str(template_hole2_path), cv2.IMREAD_GRAYSCALE)
    if template_hole2_mask is None:
        raise RuntimeError(f"failed_to_read_template_hole2_mask: {template_hole2_path}")

    template_all_mask = cv2.imread(str(template_all_path), cv2.IMREAD_GRAYSCALE)
    if template_all_mask is None:
        raise RuntimeError(f"failed_to_read_template_lumen_all_mask: {template_all_path}")

    save_optional_stage(stage_root_dir, "template", 1, "pro1_endpoint_grayscale", template_gray)
    save_optional_stage(stage_root_dir, "template", 2, "pro1_endpoint_outer_mask", template_outer_mask)
    save_optional_stage(stage_root_dir, "template", 3, "pro1_endpoint_section_mask", template_section_mask)
    save_optional_stage(stage_root_dir, "template", 4, "pro1_endpoint_lumen_hole1_mask", template_hole1_mask)
    save_optional_stage(stage_root_dir, "template", 5, "pro1_endpoint_lumen_hole2_mask", template_hole2_mask)
    save_optional_stage(stage_root_dir, "template", 6, "pro1_endpoint_lumen_all_mask", template_all_mask)

    return {
        "gray": template_gray,
        "outer_mask": template_outer_mask,
        "section_mask": template_section_mask,
        "center": mask_centroid(template_section_mask),
        "area": int(np.count_nonzero(template_outer_mask)),
        "holes": {
            "hole1_mask": template_hole1_mask,
            "hole2_mask": template_hole2_mask,
            "all_mask": template_all_mask,
        },
    }


############################## source를 template 좌표계에 배치하는 함수 ##############################
def place_source_to_template(source_gray, template_model, scale_adjust=1.0):

    source_outer_info = extract_source_outer_component_pro1(source_gray)
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

    metrics = make_metrics_row("pro1", source_path.name)

    def fail(message):
        metrics["error"] = message
        return False, message, metrics

    source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_bgr is None:
        return fail("imread_failed")

    source_gray = bgr_to_grayscale(source_bgr)
    save_optional_stage(stage_root_dir, source_path.stem, 1, f"{source_path.stem}_grayscale", source_gray)

    try:
        placed_result = place_source_to_template(source_gray, template_model, scale_adjust=scale_adjust)
    except RuntimeError as exc:
        return fail(str(exc))

    metrics["scale"] = float(placed_result["scale"])
    metrics["shift_x"] = float(placed_result["dx"])
    metrics["shift_y"] = float(placed_result["dy"])

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
        save_optional_stage(stage_root_dir, source_path.stem, 7, f"{source_path.stem}_lumen_hole1_mask", source_grouped["hole1_mask"])
        save_optional_stage(stage_root_dir, source_path.stem, 8, f"{source_path.stem}_lumen_hole2_mask", source_grouped["hole2_mask"])
        save_optional_stage(stage_root_dir, source_path.stem, 9, f"{source_path.stem}_lumen_all_mask", source_grouped["all_mask"])

        rot = find_best_rotation_with_hole_groups(
            source_grouped=source_grouped,
            template_grouped=template_model["holes"],
            center=template_model["center"],
            coarse_step=3.0,
            fine_step=0.25,
        )
        best_angle = float(rot["angle"])
        metrics["match_mode"] = "hole"
        metrics["score"] = float(rot["score"])
        metrics["ksize"] = int(source_lumen_result["ksize"])
        metrics["threshold"] = float(source_lumen_result["threshold"])
        metrics["iou_pair"] = float(rot["s_pair"])
        metrics["iou_all"] = float(rot["s_all"])
        metric_msg = (
            f"hole[ score={rot['score']:.3f}, k={source_lumen_result['ksize']}, thr={source_lumen_result['threshold']:.1f}, "
            f"iou_pair={rot['s_pair']:.3f}, iou_all={rot['s_all']:.3f} ]"
        )
    else:
        best_angle, outer_iou = find_best_rotation_outer_mask(
            placed_outer_mask=placed_result["placed_outer_mask"],
            template_outer_mask=template_model["outer_mask"],
            center=template_model["center"],
            coarse_step=3.0,
            fine_step=0.25,
        )
        metrics["match_mode"] = "fallback"
        metrics["score"] = float(outer_iou)
        metrics["iou_outer"] = float(outer_iou)
        metric_msg = f"fallback[ score={outer_iou:.3f}, iou_outer={outer_iou:.3f} ]"

    metrics["rotation_deg"] = float(best_angle)

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
    metrics["rmse_section"] = rmse_error(aligned_section, template_model["section_mask"])

    if source_grouped is not None:
        aligned_hole1 = rotate_image(
            source_grouped["hole1_mask"],
            angle_deg=best_angle,
            center=template_model["center"],
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        aligned_hole2 = rotate_image(
            source_grouped["hole2_mask"],
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

        save_optional_stage(stage_root_dir, source_path.stem, 13, f"{source_path.stem}_aligned_lumen_hole1_mask", aligned_hole1)
        save_optional_stage(stage_root_dir, source_path.stem, 14, f"{source_path.stem}_aligned_lumen_hole2_mask", aligned_hole2)
        save_optional_stage(stage_root_dir, source_path.stem, 15, f"{source_path.stem}_aligned_lumen_all_mask", aligned_all)

        rmse_11 = rmse_error(aligned_hole1, template_model["holes"]["hole1_mask"])
        rmse_22 = rmse_error(aligned_hole2, template_model["holes"]["hole2_mask"])
        rmse_12 = rmse_error(aligned_hole1, template_model["holes"]["hole2_mask"])
        rmse_21 = rmse_error(aligned_hole2, template_model["holes"]["hole1_mask"])
        rmse_pair_a = 0.5 * (rmse_11 + rmse_22)
        rmse_pair_b = 0.5 * (rmse_12 + rmse_21)
        rmse_pair = min(rmse_pair_a, rmse_pair_b)
        rmse_all = rmse_error(aligned_all, template_model["holes"]["all_mask"])
        metrics["rmse_pair"] = float(rmse_pair)
        metrics["rmse_all"] = float(rmse_all)
        metrics["rmse_error"] = float(np.sqrt(0.75 * (rmse_pair ** 2) + 0.25 * (rmse_all ** 2)))
    else:
        metrics["rmse_outer"] = rmse_error(aligned_outer, template_model["outer_mask"])
        metrics["rmse_error"] = float(metrics["rmse_outer"])

    metrics["normalized_error"] = normalize_error_value(metrics["rmse_error"])
    metrics["error_percent"] = float(metrics["normalized_error"] * 100.0)
    metric_msg = (
        f"{metric_msg[:-2]}, error={metrics['rmse_error']:.3f}, "
        f"error_norm={metrics['normalized_error']:.3f} ]"
    )

    # pro1도 각도 추정은 템플릿 좌표계에서 하고, 최종 crop은 원본 좌표계에서 수행한다.
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
        return fail("failed_to_build_rotated_section_mask")

    auto_scale = 1.0
    margin = 14

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
    metrics["fit_scale"] = float(auto_scale)
    final_crop = center_crop_with_padding(
        rotated_original,
        crop_size=crop_size,
        center=crop_center,
    )
    save_optional_stage(stage_root_dir, source_path.stem, 16, f"{source_path.stem}_final_crop_{crop_size[0]}", final_crop)

    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / source_path.name
    if not cv2.imwrite(str(final_output_path), final_crop):
        return fail(f"failed_to_write_final_output: {final_output_path}")

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
            return fail(f"failed_to_write_overlay_output: {overlay_output_path}")

    geom_msg = (
        f"geom[ scale={placed_result['scale']:.4f}, shift=({placed_result['dx']:.1f},{placed_result['dy']:.1f}), "
        f"rot={best_angle:.2f}, fit={auto_scale:.3f} ]"
    )

    metrics["ok"] = True
    return True, f"ok {geom_msg}\n{metric_msg}", metrics


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
    metrics_csv_path=None,
):

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not template_result_dir.exists():
        raise FileNotFoundError(f"template result dir not found: {template_result_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = metrics_csv_path if metrics_csv_path is not None else output_dir / "preprocess_metrics.csv"
    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if save_stage_images and stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)

    source_files = collect_files(input_dir, pattern)
    total = len(source_files)
    if total == 0:
        raise FileNotFoundError(f"no_source_files: {input_dir} ({pattern})")

    template_model = prepare_template_model(
        template_result_dir,
        stage_root_dir=stage_dir if save_stage_images else None,
    )

    print("=============== 전처리 시작 (template registration + 2-hole mask align) ===============")
    print(f"이미지 개수:  {total}장")
    print("카테터 타입:  pro1 (2-hole)")
    print(f"input:     {input_dir} ({pattern})")
    print(f"template:  {template_result_dir}")
    print(f"output:    {output_dir}")
    print(f"metrics:   {metrics_csv_path}")
    if save_overlay:
        print(f"overlay:   {overlay_dir}")
    if save_stage_images and stage_dir is not None:
        print(f"stages:    {stage_dir}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    failures = []
    metrics_rows = []
    for idx, source_path in enumerate(source_files, start=1):
        ok, msg, metrics = process_source(
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
        metrics_rows.append(metrics)

        if ok:
            print(f"[{idx}/{total}] OK    {source_path.name:<18}")
            print(f"{msg}\n")
        else:
            failures.append((source_path.name, msg))
            print(f"[{idx}/{total}] FAIL  {source_path.name:<18}")
            print(f"{msg}\n")

    print("==================================== 전처리 완료 ===================================")
    write_metrics_csv(metrics_csv_path, metrics_rows)
    print(f"metrics csv saved: {metrics_csv_path}")
    if failures:
        print(f"완료 (실패 {len(failures)} 건)")
        for name, msg in failures[:10]:
            print(f"  - {name}: {msg}")
    else:
        print("완료! 모든 이미지를 템플릿 기준으로 정렬했습니다.")


def main():

    parser = argparse.ArgumentParser(
        description="pro1 전처리: 1) Gray scale -> 2) Template Registration(2-hole 중심+회전) -> 3) Crop"
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
        default="*.BMP,*.bmp,*.png",
        help="파일 확장자(콤마 구분 예: *.BMP,*.bmp,*.png)",
    )
    parser.add_argument(
        "--template-result-dir",
        type=Path,
        default=TEMPLATE_RESULT_DIR,
        help="전처리된 pro1 템플릿 결과 폴더 경로",
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
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="이미지별 정렬 지표를 저장할 CSV 경로. 기본값은 output-dir/preprocess_metrics.csv",
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
        metrics_csv_path=args.metrics_csv,
    )


if __name__ == "__main__":
    main()
