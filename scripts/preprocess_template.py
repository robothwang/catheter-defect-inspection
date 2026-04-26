from pathlib import Path

import cv2
import numpy as np


TEMPLATE_JOBS = [
    (
        Path("/home/hjj747/catheter-preprocessing/data/dataset/templates/pro_1_endpoint.png"),
        Path("/home/hjj747/catheter-preprocessing/data/dataset/preprocessed_templates/pro1_endpoint"),
        "pro1_endpoint",
    ),
    (
        Path("/home/hjj747/catheter-preprocessing/data/dataset/templates/pro_2_endpoint.png"),
        Path("/home/hjj747/catheter-preprocessing/data/dataset/preprocessed_templates/pro2_endpoint"),
        "pro2_endpoint",
    ),
    (
        Path("/home/hjj747/catheter-preprocessing/data/dataset/templates/pro_3_endpoint.png"),
        Path("/home/hjj747/catheter-preprocessing/data/dataset/preprocessed_templates/pro3_endpoint"),
        "pro3_endpoint",
    ),
]


############################## 단계별 결과 이미지 저장 함수 ##############################
def save_stage_image(output_dir, index, stage_name, image):

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{index:02d}_{stage_name}.png"
    ok = cv2.imwrite(str(out_path), image)

    if not ok:
        raise RuntimeError(f"failed_to_write_image: {out_path}")

    return out_path


############################## gray scale 함수 ##############################
def rbg_to_grayscale(img):

    # 템플릿 원본(BGR)을 후속 이진화가 쉬운 grayscale로 변환한다.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return gray_img


############################## 이미지 이진화 함수 ##############################
def grayscale_to_binary(gray_img, threshold=240, invert=True):

    # 밝은 배경/어두운 단면 구조를 분리하기 위해 threshold 기반 이진화를 수행한다.
    threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary_img = cv2.threshold(gray_img, threshold, 255, threshold_type)

    return binary_img


############################## 가장 큰 연결성분 추출 함수 ##############################
def extract_largest_component(binary_img):

    # 숫자나 작은 표식보다 면적이 큰 카테터 본체만 남기기 위해 최대 연결성분을 선택한다.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(binary_img)

    best_idx = None
    best_area = -1

    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_idx = idx

    main_component = np.zeros_like(binary_img)
    main_component[labels == best_idx] = 255

    return main_component


############################## key-point 제거 함수 ##############################
def remove_keypoints(mask_img, open_kernel_size=11, close_kernel_size=9):

    # 경계에 붙은 작은 key-point 표식은 opening으로 깎고, 본체 형상은 closing으로 다시 다듬는다.
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_kernel_size, open_kernel_size),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_kernel_size, close_kernel_size),
    )

    cleaned = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    return cleaned


############################## 외곽 마스크를 채워 section mask를 만드는 함수 ##############################
def fill_section_mask(outer_mask):

    # 외곽 contour를 채워 카테터 단면 전체 영역(section)을 만든다.
    contours, _ = cv2.findContours(outer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(outer_mask, dtype=np.uint8)
    cv2.drawContours(filled, [main_contour], -1, 255, thickness=cv2.FILLED)

    return filled


############################## 4개 lumen을 big/small/all로 그룹화하는 함수 ##############################
def build_hole_group_masks_2(shape, components):

    top2 = sorted(components, key=lambda x: x["area"], reverse=True)[:2]
    if len(top2) < 2:
        return None

    # pro1은 위/아래 순서를 고정해 hole1, hole2로 저장한다.
    top2 = sorted(top2, key=lambda x: x["center"][1])
    hole_1, hole_2 = top2[0], top2[1]

    hole1_mask = np.zeros(shape, dtype=np.uint8)
    hole2_mask = np.zeros(shape, dtype=np.uint8)
    all_mask = np.zeros(shape, dtype=np.uint8)

    cv2.drawContours(hole1_mask, [hole_1["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(hole2_mask, [hole_2["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(all_mask, [hole_1["contour"], hole_2["contour"]], -1, 255, thickness=cv2.FILLED)

    return {
        "top2": top2,
        "hole_1": hole_1,
        "hole_2": hole_2,
        "hole1_mask": hole1_mask,
        "hole2_mask": hole2_mask,
        "all_mask": all_mask,
    }


############################## 3개 lumen을 big/small/all로 그룹화하는 함수 ##############################
def build_hole_group_masks_3(shape, components):

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
    }


############################## 4개 lumen을 big/small/all로 그룹화하는 함수 ##############################
def build_hole_group_masks(shape, components):

    top4 = sorted(components, key=lambda x: x["area"], reverse=True)[:4]
    if len(top4) < 4:
        return None

    big = top4[:2]
    small = top4[2:4]

    big_mask = np.zeros(shape, dtype=np.uint8)
    small_mask = np.zeros(shape, dtype=np.uint8)
    all_mask = np.zeros(shape, dtype=np.uint8)

    cv2.drawContours(big_mask, [big[0]["contour"], big[1]["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(small_mask, [small[0]["contour"], small[1]["contour"]], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(
        all_mask,
        [big[0]["contour"], big[1]["contour"], small[0]["contour"], small[1]["contour"]],
        -1,
        255,
        thickness=cv2.FILLED,
    )

    return {
        "top4": top4,
        "big": big,
        "small": small,
        "big_mask": big_mask,
        "small_mask": small_mask,
        "all_mask": all_mask,
    }


############################## outer/section mask로 template lumen을 만드는 함수 ##############################
def build_template_holes_from_masks(template_outer_mask, template_section_mask, lumen_count):

    # outer mask를 section 내부에서만 뒤집어 lumen 전체 mask를 만든다.
    lumen_all_mask = cv2.subtract(template_section_mask, template_outer_mask)

    contours, _ = cv2.findContours(lumen_all_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    components = []
    for cnt in contours:
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
        components.append(
            {
                "contour": cnt,
                "area": area,
                "circularity": circularity,
                "center": (cx, cy),
            }
        )

    components.sort(key=lambda x: x["area"], reverse=True)

    if lumen_count == 2:
        grouped = build_hole_group_masks_2(template_outer_mask.shape, components)
    elif lumen_count == 3:
        grouped = build_hole_group_masks_3(template_outer_mask.shape, components)
    elif lumen_count == 4:
        grouped = build_hole_group_masks(template_outer_mask.shape, components)
    else:
        raise RuntimeError(f"unsupported_lumen_count: {lumen_count}")

    if grouped is None:
        raise RuntimeError("template_hole_detection_failed")

    grouped["all_mask"] = lumen_all_mask
    return grouped


############################## 템플릿 이미지 전처리 파이프라인 ##############################
def process_template(template_path, output_dir, stage_prefix):

    # 1. 템플릿 원본을 읽는다.
    template_bgr = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
    if template_bgr is None:
        raise RuntimeError(f"failed_to_read_template: {template_path}")

    # 2. grayscale 템플릿 저장
    template_gray = rbg_to_grayscale(template_bgr)
    gray_path = save_stage_image(output_dir, 1, f"{stage_prefix}_grayscale", template_gray)
    print(f"saved: {gray_path}\n")

    # 3. 단면 구조가 드러나도록 이진화
    template_binary = grayscale_to_binary(template_gray, threshold=240, invert=True)
    binary_path = save_stage_image(output_dir, 2, f"{stage_prefix}_binary", template_binary)
    print(f"saved: {binary_path}\n")

    # 4. 숫자 등 작은 연결성분을 제거하고 카테터 본체만 유지
    template_main_component = extract_largest_component(template_binary)
    main_component_path = save_stage_image(output_dir, 3, f"{stage_prefix}_main_component", template_main_component)
    print(f"saved: {main_component_path}\n")

    # 5. 경계에 붙어 있는 key-point 표식을 제거된 결과를 템플릿 외곽 마스크로 사용한다.
    template_outer_mask = remove_keypoints(template_main_component)
    outer_mask_path = save_stage_image(output_dir, 4, f"{stage_prefix}_outer_mask", template_outer_mask)
    print(f"saved: {outer_mask_path}\n")

    # 6. 외곽 마스크를 채워 템플릿 section mask를 만든다.
    template_section_mask = fill_section_mask(template_outer_mask)
    if template_section_mask is None:
        raise RuntimeError(f"failed_to_build_template_section_mask: {template_path}")
    section_mask_path = save_stage_image(output_dir, 5, f"{stage_prefix}_section_mask", template_section_mask)
    print(f"saved: {section_mask_path}\n")

    # 7. 각 타입의 lumen 개수에 맞춰 템플릿 hole mask를 미리 저장한다.
    if stage_prefix == "pro1_endpoint":
        template_holes = build_template_holes_from_masks(template_outer_mask, template_section_mask, lumen_count=2)

        hole1_mask_path = save_stage_image(output_dir, 6, f"{stage_prefix}_lumen_hole1_mask", template_holes["hole1_mask"])
        print(f"saved: {hole1_mask_path}\n")

        hole2_mask_path = save_stage_image(output_dir, 7, f"{stage_prefix}_lumen_hole2_mask", template_holes["hole2_mask"])
        print(f"saved: {hole2_mask_path}\n")

        all_mask_path = save_stage_image(output_dir, 8, f"{stage_prefix}_lumen_all_mask", template_holes["all_mask"])
        print(f"saved: {all_mask_path}\n")

    elif stage_prefix == "pro2_endpoint":
        template_holes = build_template_holes_from_masks(template_outer_mask, template_section_mask, lumen_count=3)

        big_mask_path = save_stage_image(output_dir, 6, f"{stage_prefix}_lumen_big_mask", template_holes["big_mask"])
        print(f"saved: {big_mask_path}\n")

        small_mask_path = save_stage_image(output_dir, 7, f"{stage_prefix}_lumen_small_mask", template_holes["small_mask"])
        print(f"saved: {small_mask_path}\n")

        all_mask_path = save_stage_image(output_dir, 8, f"{stage_prefix}_lumen_all_mask", template_holes["all_mask"])
        print(f"saved: {all_mask_path}\n")

    elif stage_prefix == "pro3_endpoint":
        template_holes = build_template_holes_from_masks(template_outer_mask, template_section_mask, lumen_count=4)

        big_mask_path = save_stage_image(output_dir, 6, f"{stage_prefix}_lumen_big_mask", template_holes["big_mask"])
        print(f"saved: {big_mask_path}\n")

        small_mask_path = save_stage_image(output_dir, 7, f"{stage_prefix}_lumen_small_mask", template_holes["small_mask"])
        print(f"saved: {small_mask_path}\n")

        all_mask_path = save_stage_image(output_dir, 8, f"{stage_prefix}_lumen_all_mask", template_holes["all_mask"])
        print(f"saved: {all_mask_path}\n")



def main():

    # pro1, pro2, pro3 템플릿을 같은 파이프라인으로 순차 처리한다.
    for template_path, output_dir, stage_prefix in TEMPLATE_JOBS:
        print(f"processing: {template_path}")
        process_template(template_path, output_dir, stage_prefix)


if __name__ == "__main__":
    main()
