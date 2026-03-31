import argparse
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


def to_grayscale(img):
    # 컬러 이미지를 gray scale로 변환
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _center_of_contour(cnt):
    # contour 모멘트로 중심을 구하고, 불안정하면 bounding box 중심으로 대체한다.
    m = cv2.moments(cnt)
    if m["m00"] != 0:
        return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
    x, y, w, h = cv2.boundingRect(cnt)
    return x + w // 2, y + h // 2


def _find_main_contours(gray):
    # 밝은 카테터 외곽을 우선 검출하고, 실패 시 Otsu로 재시도한다.
    _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        return contours

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _choose_target_center(gray):
    # 면적 + 중앙 근접도를 함께 사용해 카테터 중심 후보를 고른다.
    h, w = gray.shape[:2]
    img_cx, img_cy = w // 2, h // 2
    contours = _find_main_contours(gray)

    if not contours:
        return img_cx, img_cy

    best = None
    best_score = -1.0
    diag = max(np.hypot(w, h), 1.0)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        c_x, c_y = _center_of_contour(cnt)
        dist = np.hypot(c_x - img_cx, c_y - img_cy) / diag
        score = float(area) * (1.5 - min(dist, 1.5))
        if score > best_score:
            best_score = score
            best = (c_x, c_y)

    if best is None:
        return img_cx, img_cy
    return best


def get_smart_crop(gray, crop_size=(600, 600)):
    # 회전된 원본에서 카테터 중심을 기준으로 최종 크롭한다.
    h, w = gray.shape[:2]
    c_x, c_y = _choose_target_center(gray)
    crop_w, crop_h = crop_size
    x1 = max(0, c_x - crop_w // 2)
    y1 = max(0, c_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    cropped = gray[y1:y2, x1:x2]

    pad_bottom = max(0, crop_h - cropped.shape[0])
    pad_right = max(0, crop_w - cropped.shape[1])
    if pad_bottom > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

    return cropped


def _hole_centers(gray):
    # 카테터 내부의 어두운 두 구멍 후보 중심점을 찾는다.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    area_total = gray.shape[0] * gray.shape[1]
    min_area = area_total * 0.003
    max_area = area_total * 0.2

    centers = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            centers.append(_center_of_contour(cnt))
    return centers


def _pair_angle_from_centers(centers):
    # 가장 멀리 떨어진 두 중심을 이용해 구멍 축 각도를 계산한다.
    if len(centers) < 2:
        return None

    best_pair = None
    best_dist = -1
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[j][0] - centers[i][0]
            dy = centers[j][1] - centers[i][1]
            dist = dx * dx + dy * dy
            if dist > best_dist:
                best_dist = dist
                best_pair = (centers[i], centers[j])

    p1, p2 = best_pair
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def get_hole_axis_angle(gray):
    # 구멍 축 각도(도)를 반환한다. 실패 시 None.
    centers = _hole_centers(gray)
    return _pair_angle_from_centers(centers)


def apply_rotation(img, angle, border_mode=cv2.BORDER_REFLECT_101):
    # 회전을 적용한다. reflect 보더로 검은 삼각 여백을 줄인다.
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
    )


def _vertical_deviation(angle):
    # 90도(수직축)와의 각도 차이를 계산한다.
    return min(abs(angle - 90), abs(angle + 90))


def _extract_rotation_roi(gray):
    # 회전 각도 추정용 ROI를 만든다(최종 크롭이 아니라 각도 계산 전용).
    c_x, c_y = _choose_target_center(gray)
    half = 220
    h, w = gray.shape[:2]
    x1 = max(0, c_x - half)
    y1 = max(0, c_y - half)
    x2 = min(w, c_x + half)
    y2 = min(h, c_y + half)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return gray
    return roi


def choose_horizontal_rotation(gray):
    # 원본 기준으로 회전 각도를 고른 뒤, 내부 막대가 가로가 되도록 후보 중 최적값을 선택한다.
    roi = _extract_rotation_roi(gray)
    angle = get_hole_axis_angle(roi)
    if angle is None:
        return 0.0

    candidates = [angle + 90, angle - 90, -angle + 90, -angle - 90]
    best_rot = candidates[0]
    best_dev = 1e9

    for rot in candidates:
        rotated = apply_rotation(roi, rot)
        new_angle = get_hole_axis_angle(rotated)
        if new_angle is None:
            continue
        dev = _vertical_deviation(new_angle)
        if dev < best_dev:
            best_dev = dev
            best_rot = rot

    return float(best_rot)


def extract_main_component(gray, foreground="bright"):
    # 가장 큰 연결 성분(카테터 단면)을 추출한다.
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

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < 50:
        return None

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[labels == idx] = 255

    cx, cy = centroids[idx]
    x = int(stats[idx, cv2.CC_STAT_LEFT])
    y = int(stats[idx, cv2.CC_STAT_TOP])
    w = int(stats[idx, cv2.CC_STAT_WIDTH])
    h = int(stats[idx, cv2.CC_STAT_HEIGHT])

    return {
        "mask": mask,
        "center": (float(cx), float(cy)),
        "bbox": (x, y, w, h),
        "area": area,
    }


def place_on_template(proc_gray, proc_mask, template_shape, target_center, scale):
    # 처리 이미지를 스케일 + 평행이동하여 템플릿 좌표계에 배치한다.
    h_t, w_t = template_shape

    scaled_gray = cv2.resize(proc_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_mask = cv2.resize(proc_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    proc_info = extract_main_component(scaled_gray, foreground="bright")
    if proc_info is None:
        raise RuntimeError("scaled_processed_component_not_found")

    src_center = proc_info["center"]
    dx = float(target_center[0] - src_center[0])
    dy = float(target_center[1] - src_center[1])

    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    placed_gray = cv2.warpAffine(
        scaled_gray,
        matrix,
        (w_t, h_t),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    placed_mask = cv2.warpAffine(
        scaled_mask,
        matrix,
        (w_t, h_t),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return placed_gray, placed_mask, dx, dy


def overlay_preview(template_bgr, aligned_gray, aligned_mask, alpha=0.45):
    # 템플릿과 정렬 결과를 반투명 오버레이로 시각화한다.
    proc_bgr = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(template_bgr, 1.0 - alpha, proc_bgr, alpha, 0.0)

    out = template_bgr.copy()
    mask_bool = aligned_mask > 0
    out[mask_bool] = blended[mask_bool]
    return out


def prepare_template(template_path):
    # 템플릿 이미지를 읽고, 기준 마스크/중심 정보를 준비한다.
    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise RuntimeError(f"failed_to_read_template: {template_path}")

    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # 템플릿은 밝은 배경 + 회색(어두운) 카테터라서 dark 전경 추출을 사용한다.
    template_info = extract_main_component(template_gray, foreground="dark")
    if template_info is None:
        raise RuntimeError("template_component_not_found")

    return {
        "bgr": template_bgr,
        "gray": template_gray,
        "mask": template_info["mask"],
        "center": template_info["center"],
        "bbox": template_info["bbox"],
        "area": template_info["area"],
    }


def make_overlay_image(final_img, template_model, alpha=0.45, scale_adjust=1.0):
    # 전처리 결과를 템플릿 중심에 정렬해 오버레이 이미지를 만든다.
    proc_info = extract_main_component(final_img, foreground="bright")
    if proc_info is None:
        raise RuntimeError("processed_component_not_found")

    _, _, w_t, h_t = template_model["bbox"]
    _, _, w_p, h_p = proc_info["bbox"]
    template_size = max(w_t, h_t)
    proc_size = max(w_p, h_p)
    if proc_size <= 0:
        raise RuntimeError("invalid_processed_bbox")

    scale = (template_size / proc_size) * float(scale_adjust)
    placed_gray, placed_mask, dx, dy = place_on_template(
        proc_gray=final_img,
        proc_mask=proc_info["mask"],
        template_shape=template_model["gray"].shape,
        target_center=template_model["center"],
        scale=scale,
    )

    ov = overlay_preview(
        template_bgr=template_model["bgr"],
        aligned_gray=placed_gray,
        aligned_mask=placed_mask,
        alpha=alpha,
    )

    cx, cy = int(round(template_model["center"][0])), int(round(template_model["center"][1]))
    cv2.drawMarker(ov, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    return ov, scale, dx, dy


def process_pipeline(
    img_path,
    save_path,
    overlay_path,
    crop_size=(600, 600),
    template_model=None,
    alpha=0.45,
    scale_adjust=1.0,
    save_overlay=True,
):
    # 전처리 메인 파이프라인: 1) gray -> 2) rotation -> 3) crop (+ optional overlay save)
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return str(img_path), False, "imread_failed"

        gray = to_grayscale(img)
        rot = choose_horizontal_rotation(gray)
        rotated = apply_rotation(gray, rot)
        final_img = get_smart_crop(rotated, crop_size=crop_size)

        ok = cv2.imwrite(str(save_path), final_img)
        if not ok:
            return str(img_path), False, "imwrite_processed_failed"

        msg = f"rotation={rot:.2f}, "

        if save_overlay:
            if template_model is None:
                return str(img_path), False, "template_model_missing"
            if overlay_path is None:
                return str(img_path), False, "overlay_path_missing"

            ov, scale, dx, dy = make_overlay_image(
                final_img,
                template_model=template_model,
                alpha=alpha,
                scale_adjust=scale_adjust,
            )

            ov_ok = cv2.imwrite(str(overlay_path), ov)
            if not ov_ok:
                return str(img_path), False, "imwrite_overlay_failed"

            msg = f"{msg}overlay_scale={scale:.4f}, shift=({dx:.1f},{dy:.1f})"

        return str(img_path), True, msg
    except Exception as exc:
        return str(img_path), False, f"error={exc}"


def run_preprocess(
    input_dir,
    pattern,
    output_dir,
    overlay_dir,
    template_path,
    crop_size=(600, 600),
    alpha=0.45,
    scale_adjust=1.0,
    save_overlay=True,
):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(pattern))
    total = len(files)

    template_model = None
    overlay_paths = [None for _ in files]

    if save_overlay:
        if not template_path.exists():
            raise FileNotFoundError(f"template not found: {template_path}")
        overlay_dir.mkdir(parents=True, exist_ok=True)
        template_model = prepare_template(template_path)
        overlay_paths = [overlay_dir / f"{f.stem}_overlay.png" for f in files]

    print(f"{total}장 전처리 시작: 1) Gray -> 2) Rotation -> 3) Crop")
    print(f"input:    {input_dir} ({pattern})")
    print(f"output:   {output_dir}")
    if save_overlay:
        print(f"template: {template_path}")
        print(f"overlay:  {overlay_dir}")

    failures = []
    with ProcessPoolExecutor() as executor:
        for idx, (path, ok, msg) in enumerate(
            executor.map(
                process_pipeline,
                files,
                [output_dir / f.name for f in files],
                overlay_paths,
                repeat(crop_size),
                repeat(template_model),
                repeat(alpha),
                repeat(scale_adjust),
                repeat(save_overlay),
            ),
            start=1,
        ):
            name = Path(path).name
            if not ok:
                failures.append((path, msg))
                print(f"[{idx}/{total}] FAIL {name} | {msg}")
            else:
                print(f"[{idx}/{total}] OK   {name} | {msg}")

    if failures:
        print(f"완료 (실패 {len(failures)}건)")
        for path, msg in failures[:10]:
            print(f"  - {path}: {msg}")
    else:
        if save_overlay:
            print("완료! 전처리 이미지와 오버레이 이미지를 모두 저장했습니다.")
        else:
            print("완료! 모든 결과를 가로 정렬 폴더에 저장했습니다.")


def main():
    # 기본 실행: pro1 전처리 (독립 파일)
    parser = argparse.ArgumentParser(description="pro1 이미지 전처리: 1) Gray -> 2) Rotation -> 3) Crop")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_1"),
        help="입력 폴더",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.BMP",
        help="파일 패턴(예: *.BMP, *.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro1"),
        help="결과 저장 폴더",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro1"),
        help="템플릿 오버레이 결과 저장 폴더",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/raw/label_orign/pro_1_endpoint.png"),
        help="기준 템플릿 이미지",
    )
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("W", "H"),
        help="크롭 크기 수동 지정(예: --crop-size 600 600)",
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
        help="오버레이 자동 스케일 보정 배율",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="오버레이 결과 저장을 비활성화",
    )

    args = parser.parse_args()
    crop_size = tuple(args.crop_size) if args.crop_size is not None else (600, 600)

    run_preprocess(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        overlay_dir=args.overlay_dir,
        template_path=args.template,
        crop_size=crop_size,
        alpha=args.alpha,
        scale_adjust=args.scale_adjust,
        save_overlay=not args.no_overlay,
    )


if __name__ == "__main__":
    main()
