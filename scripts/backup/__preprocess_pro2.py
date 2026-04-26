import argparse
from pathlib import Path

import cv2
import numpy as np


def to_grayscale(img):
    # 컬러 이미지를 gray scale로 변환한다.
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def extract_main_component(gray, foreground="bright"):
    # 가장 큰 연결 성분(카테터 단면)을 추출한다.
    # foreground="bright"면 밝은 물체를, "dark"면 어두운 물체를 전경으로 본다.
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

    # 면적이 큰 성분을 우선하되, 중심에서 너무 먼 성분은 감점한다.
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


def place_to_template(gray_target, mask_target, template_shape, template_center, scale):
    # 타깃 이미지를 템플릿 좌표계로 스케일/이동 배치한다.
    h_t, w_t = template_shape

    scaled_gray = cv2.resize(gray_target, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_mask = cv2.resize(mask_target, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    scaled_info = extract_main_component(scaled_gray, foreground="bright")
    if scaled_info is None:
        raise RuntimeError("scaled_target_component_not_found")

    sx, sy = scaled_info["center"]
    tx, ty = template_center
    dx = float(tx - sx)
    dy = float(ty - sy)

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


def find_best_rotation(placed_mask, template_mask, template_center, coarse_step=3.0, fine_step=0.25):
    # 템플릿과의 IoU가 최대가 되는 회전 각도를 탐색한다.
    coarse_angles = np.arange(-180.0, 180.0, coarse_step, dtype=np.float32)

    best_angle = 0.0
    best_score = -1.0

    for angle in coarse_angles:
        rot_mask = rotate_image(
            placed_mask,
            angle_deg=float(angle),
            center=template_center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
        )
        score = iou_score(rot_mask, template_mask)
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    start = best_angle - coarse_step
    end = best_angle + coarse_step
    fine_angles = np.arange(start, end + fine_step, fine_step, dtype=np.float32)

    for angle in fine_angles:
        rot_mask = rotate_image(
            placed_mask,
            angle_deg=float(angle),
            center=template_center,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
        )
        score = iou_score(rot_mask, template_mask)
        if score > best_score:
            best_score = score
            best_angle = float(angle)

    return float(best_angle), float(best_score)


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
    # 단일 이미지 전처리: gray -> template registration(각도 추정) -> rotation -> crop.
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, "imread_failed"

    gray = to_grayscale(img)

    target_info = extract_main_component(gray, foreground="bright")
    if target_info is None:
        return False, "target_component_not_found"

    # 면적 비율 기반 스케일로 템플릿 크기와 대략 맞춘다.
    target_area = max(float(target_info["area"]), 1.0)
    template_area = max(float(template_model["area"]), 1.0)
    scale = np.sqrt(template_area / target_area) * float(scale_adjust)

    placed_gray, placed_mask, dx, dy = place_to_template(
        gray_target=gray,
        mask_target=target_info["mask"],
        template_shape=template_model["gray"].shape,
        template_center=template_model["center"],
        scale=scale,
    )

    center = template_model["center"]
    best_angle, best_iou = find_best_rotation(
        placed_mask=placed_mask,
        template_mask=template_model["mask"],
        template_center=center,
        coarse_step=3.0,
        fine_step=0.25,
    )

    aligned_gray = rotate_image(
        placed_gray,
        angle_deg=best_angle,
        center=center,
        interpolation=cv2.INTER_CUBIC,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )
    aligned_mask = rotate_image(
        placed_mask,
        angle_deg=best_angle,
        center=center,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
        border_mode=cv2.BORDER_CONSTANT,
    )

    # 템플릿은 각도 추정에만 사용하고, 최종 출력은 원본 스케일을 유지한다.
    # (사용자 요청: 템플릿은 기준 도구로만 사용)
    original_center = target_info["center"]
    rotated_original = rotate_image(
        gray,
        angle_deg=best_angle,
        center=original_center,
        interpolation=cv2.INTER_CUBIC,
        border_value=0,
        border_mode=cv2.BORDER_REFLECT_101,
    )
    cropped = center_crop_with_padding(
        rotated_original,
        crop_size=crop_size,
        center=original_center,
    )

    out_path = output_dir / image_path.name
    if not cv2.imwrite(str(out_path), cropped):
        return False, "imwrite_processed_failed"

    if save_overlay:
        ov = overlay_preview(
            template_bgr=template_model["bgr"],
            aligned_gray=aligned_gray,
            aligned_mask=aligned_mask,
            alpha=alpha,
        )

        # 중심점 마커를 찍어 정렬 확인을 쉽게 만든다.
        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.drawMarker(ov, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        ov_path = overlay_dir / f"{image_path.stem}_overlay.png"
        if not cv2.imwrite(str(ov_path), ov):
            return False, "imwrite_overlay_failed"

    return True, f"\"ok\" scale={scale:.4f}, shift=({dx:.1f},{dy:.1f}), rot={best_angle:.2f}, iou={best_iou:.4f}"


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

    files = sorted(input_dir.glob(pattern))
    total = len(files)

    template_model = prepare_template(template_path)

    print(f"{total}장 pro2 전처리 시작 (template registration / no-virtual-point)")
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
            print(f"[{idx}/{total}] OK   {path.name}\n{msg}")
        else:
            failures.append((path.name, msg))
            print(f"[{idx}/{total}] FAIL {path.name}\n{msg}")

    if failures:
        print(f"완료 (실패 {len(failures)}건)")
        for name, msg in failures[:10]:
            print(f"  - {name}: {msg}")
    else:
        print("완료! 모든 이미지를 템플릿 기준으로 정렬했습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="pro2 전처리: 1) Gray -> 2) Template Registration(중심+회전) -> 3) Crop"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_2"),
        help="입력 폴더",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="파일 패턴(예: *.png, *.BMP)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/raw/label_orign/pro_2_endpoint.png"),
        help="기준 템플릿 이미지",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro2"),
        help="정렬+크롭 결과 저장 폴더",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro2"),
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
