import argparse
from pathlib import Path

import cv2
import numpy as np

PROFILES = {
    "pro1": {
        "template": Path("/home/hjj747/catheter/data/raw/label_orign/pro_1_endpoint.png"),
        "name_prefix": "1_",
    },
    "pro2": {
        "template": Path("/home/hjj747/catheter/data/raw/label_orign/pro_2_endpoint.png"),
        "name_prefix": "2_",
    },
    "pro3": {
        "template": Path("/home/hjj747/catheter/data/raw/label_orign/pro_3_endpoint.png"),
        "name_prefix": "3_",
    },
}


def extract_main_component(gray, foreground="bright"):
    """
    이미지에서 가장 큰 연결 성분(주요 물체) 마스크와 기하 정보를 추출한다.
    foreground:
      - "bright": 밝은 물체를 전경으로 사용
      - "dark": 어두운 물체를 전경으로 사용
    """
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
    """
    처리 이미지를 스케일 + 평행이동하여 템플릿 좌표계에 배치한다.
    중심점(target_center) 기준 정렬을 수행한다.
    """
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


def overlay_images(template_bgr, placed_gray, placed_mask, alpha):
    """
    템플릿 위에 처리 이미지를 반투명으로 겹친다.
    마스크 영역에서만 블렌딩한다.
    """
    proc_bgr = cv2.cvtColor(placed_gray, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(template_bgr, 1.0 - alpha, proc_bgr, alpha, 0.0)

    result = template_bgr.copy()
    mask_bool = placed_mask > 0
    result[mask_bool] = blended[mask_bool]
    return result


def process_one_image(image_path, template_bgr, template_gray, template_info, out_dir, alpha, scale_adjust):
    """
    단일 processed 이미지를 템플릿과 중심 정렬해 겹친 결과를 저장한다.
    """
    proc = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if proc is None:
        return False, "imread_failed"

    proc_info = extract_main_component(proc, foreground="bright")
    if proc_info is None:
        return False, "processed_component_not_found"

    _, _, w_t, h_t = template_info["bbox"]
    _, _, w_p, h_p = proc_info["bbox"]
    template_size = max(w_t, h_t)
    proc_size = max(w_p, h_p)
    if proc_size <= 0:
        return False, "invalid_processed_bbox"

    scale = (template_size / proc_size) * float(scale_adjust)
    placed_gray, placed_mask, dx, dy = place_on_template(
        proc_gray=proc,
        proc_mask=proc_info["mask"],
        template_shape=template_gray.shape,
        target_center=template_info["center"],
        scale=scale,
    )

    merged = overlay_images(template_bgr, placed_gray, placed_mask, alpha=alpha)

    # 중심점 시각화(정렬 확인용)
    tx, ty = int(round(template_info["center"][0])), int(round(template_info["center"][1]))
    cv2.drawMarker(merged, (tx, ty), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    out_path = out_dir / f"{image_path.stem}_overlay.png"
    ok = cv2.imwrite(str(out_path), merged)
    if not ok:
        return False, "imwrite_failed"

    return True, f"saved={out_path.name} scale={scale:.4f} shift=({dx:.1f},{dy:.1f})"


def main():
    parser = argparse.ArgumentParser(description="processed_images를 기준 템플릿에 중심 정렬해 겹쳐 저장")
    parser.add_argument(
        "--type",
        "-t",
        choices=sorted(PROFILES.keys()),
        default="pro1",
        help="처리할 제품 타입 선택",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="정렬된 처리 이미지 폴더 수동 지정(미지정 시 processed_images/<type> 우선 사용)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="겹칠 기준 도면 이미지 수동 지정(미지정 시 타입 프로필 사용)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/hjj747/catheter/data/processed/overlay_images"),
        help="겹친 결과 저장 루트 폴더",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="겹침 투명도(0~1)",
    )
    parser.add_argument(
        "--scale-adjust",
        type=float,
        default=1.0,
        help="자동 스케일 보정 배율(예: 0.98, 1.02)",
    )
    args = parser.parse_args()

    profile = PROFILES[args.type]
    processed_root = Path("/home/hjj747/catheter/data/processed/processed_images")
    default_type_input = processed_root / args.type
    if args.input_dir is not None:
        input_dir = args.input_dir
    else:
        # 기존 단일 폴더 구조를 유지 중인 경우도 지원
        input_dir = default_type_input if default_type_input.exists() else processed_root

    template_path = args.template if args.template is not None else profile["template"]
    output_dir = args.output_root / args.type
    name_prefix = profile["name_prefix"]

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not template_path.exists():
        raise FileNotFoundError(f"template not found: {template_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise RuntimeError(f"failed to read template: {template_path}")
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    # 템플릿은 밝은 배경 + 어두운 도형이므로 dark 전경으로 추출
    template_info = extract_main_component(template_gray, foreground="dark")
    if template_info is None:
        raise RuntimeError("template_component_not_found")

    files = sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            and p.name.startswith(name_prefix)
        ]
    )

    total = len(files)
    print(f"type={args.type} | {total}장 오버레이 시작")
    print(f"input:    {input_dir}")
    print(f"template: {template_path}")
    print(f"output:   {output_dir}")
    print(f"filter:   filename startswith '{name_prefix}'")

    failures = []
    for idx, path in enumerate(files, start=1):
        ok, msg = process_one_image(
            image_path=path,
            template_bgr=template_bgr,
            template_gray=template_gray,
            template_info=template_info,
            out_dir=output_dir,
            alpha=args.alpha,
            scale_adjust=args.scale_adjust,
        )
        if ok:
            print(f"[{idx}/{total}] OK   {path.name} | {msg}")
        else:
            failures.append((path.name, msg))
            print(f"[{idx}/{total}] FAIL {path.name} | {msg}")

    if failures:
        print(f"완료 (실패 {len(failures)}건)")
        for name, msg in failures[:10]:
            print(f"  - {name}: {msg}")
    else:
        print("완료! 모든 이미지를 템플릿에 중심 정렬해 저장했습니다.")


if __name__ == "__main__":
    main()
