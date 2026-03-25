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


def process_pipeline(img_path, save_path, crop_size=(600, 600)):
    # 전처리 메인 파이프라인: 1) gray -> 2) rotation -> 3) crop
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
            return str(img_path), False, "imwrite_failed"
        return str(img_path), True, f"rotation={rot:.2f}"
    except Exception as exc:
        return str(img_path), False, f"error={exc}"


def run_preprocess(input_dir, pattern, output_dir, crop_size=(600, 600)):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob(pattern))
    total = len(files)

    print(f"{total}장 전처리 시작: 1) Gray -> 2) Rotation -> 3) Crop")
    print(f"input:  {input_dir} ({pattern})")
    print(f"output: {output_dir}")

    failures = []
    with ProcessPoolExecutor() as executor:
        for idx, (path, ok, msg) in enumerate(
            executor.map(
                process_pipeline,
                files,
                [output_dir / f.name for f in files],
                repeat(crop_size),
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
        print("완료! 모든 결과를 가로 정렬 폴더에 저장했습니다.")


if __name__ == "__main__":
    # 기본 실행: pro1 전처리 (독립 파일)
    parser = argparse.ArgumentParser(description="pro1 이미지 전처리: 1) Gray -> 2) Rotation -> 3) Crop")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/hjj747/catheter/data/raw/targets/pro_1"),
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
        default=Path("/home/hjj747/catheter/data/processed/processed_images/pro1"),
        help="결과 저장 폴더",
    )
    parser.add_argument(
        "--crop-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("W", "H"),
        help="크롭 크기 수동 지정(예: --crop-size 600 600)",
    )
    args = parser.parse_args()
    crop_size = tuple(args.crop_size) if args.crop_size is not None else (600, 600)
    run_preprocess(
        input_dir=args.input_dir,
        pattern=args.pattern,
        output_dir=args.output_dir,
        crop_size=crop_size,
    )
