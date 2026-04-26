import argparse
from pathlib import Path

import preprocess_pro1_holealign as pro1_holealign
import preprocess_pro2_holealign as pro2_holealign
import preprocess_pro3_holealign as pro3_holealign


DEFAULT_INPUT_DIRS = {
    "pro1": Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_1"),
    "pro2": Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_2"),
    "pro3": Path("/home/hjj747/catheter-defect-inspection/data/raw/targets/pro_3"),
}

DEFAULT_PATTERNS = {
    "pro1": "*.BMP,*.bmp,*.png",
    "pro2": "*.png",
    "pro3": "*.png",
}

DEFAULT_TEMPLATE_PATHS = {
    "pro1": Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_templates/pro1_endpoint"),
    "pro2": Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_templates/pro2_endpoint"),
    "pro3": Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_templates/pro3_endpoint"),
}

DEFAULT_OUTPUT_DIRS = {
    "pro1": Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro1_holealign"),
    "pro2": Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro2_holealign"),
    "pro3": Path("/home/hjj747/catheter-defect-inspection/data/processed/processed_images/pro3_holealign"),
}

DEFAULT_OVERLAY_DIRS = {
    "pro1": Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro1_holealign"),
    "pro2": Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro2_holealign"),
    "pro3": Path("/home/hjj747/catheter-defect-inspection/data/processed/overlay_images/pro3_holealign"),
}

DEFAULT_STAGE_DIRS = {
    "pro1": Path("/home/hjj747/catheter-defect-inspection/data/processed/stage_images/pro1_holealign"),
    "pro3": Path("/home/hjj747/catheter-defect-inspection/data/processed/stage_images/pro3_holealign"),
}


def infer_catheter_type(input_dir, template_path=None, template_prefix=None):

    # 입력 경로, 템플릿 경로, 예전 prefix 인자에서 타입 단서를 모아 자동 추론한다.
    candidates = [
        str(input_dir).lower() if input_dir is not None else "",
        str(template_path).lower() if template_path is not None else "",
        str(template_prefix).lower() if template_prefix is not None else "",
    ]

    patterns = {
        "pro1": ("pro1", "pro_1"),
        "pro2": ("pro2", "pro_2"),
        "pro3": ("pro3", "pro_3"),
    }

    for catheter_type, tokens in patterns.items():
        for candidate in candidates:
            if any(token in candidate for token in tokens):
                return catheter_type

    raise ValueError("failed_to_infer_catheter_type: use --catheter-type")


def resolve_output_dir(catheter_type, output_dir, final_output_dir):

    if final_output_dir is not None:
        return final_output_dir
    if output_dir is not None:
        return output_dir
    return DEFAULT_OUTPUT_DIRS[catheter_type]


def resolve_overlay_dir(catheter_type, overlay_dir, overlay_output_dir):

    if overlay_output_dir is not None:
        return overlay_output_dir
    if overlay_dir is not None:
        return overlay_dir
    return DEFAULT_OVERLAY_DIRS[catheter_type]


def resolve_stage_dir(catheter_type, stage_dir, output_dir, final_output_dir):

    if stage_dir is not None:
        return stage_dir

    # 예전 preprocess_source.py 호환: output-dir은 stage, final-output-dir은 최종 결과로 해석한다.
    if catheter_type == "pro3" and output_dir is not None and final_output_dir is not None:
        return output_dir

    return DEFAULT_STAGE_DIRS.get(catheter_type)


def run_preprocess(
    catheter_type,
    input_dir,
    pattern,
    template_path,
    output_dir,
    overlay_dir,
    crop_size,
    alpha,
    scale_adjust,
    save_overlay,
    stage_dir,
    save_stage_images,
):

    if catheter_type == "pro1":
        pro1_holealign.run_preprocess(
            input_dir=input_dir,
            pattern=pattern,
            template_result_dir=template_path,
            output_dir=output_dir,
            overlay_dir=overlay_dir,
            crop_size=crop_size,
            alpha=alpha,
            scale_adjust=scale_adjust,
            save_overlay=save_overlay,
            stage_dir=stage_dir,
            save_stage_images=save_stage_images,
        )
        return

    if catheter_type == "pro2":
        pro2_holealign.run_preprocess(
            input_dir=input_dir,
            pattern=pattern,
            template_result_dir=template_path,
            output_dir=output_dir,
            overlay_dir=overlay_dir,
            crop_size=crop_size,
            alpha=alpha,
            scale_adjust=scale_adjust,
            save_overlay=save_overlay,
            stage_dir=stage_dir,
            save_stage_images=save_stage_images,
        )
        return

    if catheter_type == "pro3":
        pro3_holealign.run_preprocess(
            input_dir=input_dir,
            pattern=pattern,
            template_result_dir=template_path,
            output_dir=output_dir,
            overlay_dir=overlay_dir,
            crop_size=crop_size,
            alpha=alpha,
            scale_adjust=scale_adjust,
            save_overlay=save_overlay,
            stage_dir=stage_dir,
            save_stage_images=save_stage_images,
        )
        return

    raise ValueError(f"unsupported_catheter_type: {catheter_type}")


def main():

    parser = argparse.ArgumentParser(
        description="공용 source 전처리 진입점: 타입별 holealign 스크립트를 import 해서 실행"
    )
    parser.add_argument(
        "--catheter-type",
        type=str,
        choices=["auto", "pro1", "pro2", "pro3"],
        default="auto",
        help="카테터 타입 선택. auto면 입력/템플릿 경로에서 자동 추론",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="입력 폴더 경로",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="파일 패턴(예: *.png 또는 *.BMP,*.bmp,*.png)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=None,
        help="템플릿 입력 경로. pro1/pro2/pro3 모두 전처리 템플릿 결과 폴더",
    )
    parser.add_argument(
        "--template-result-dir",
        type=Path,
        default=None,
        help="이전 CLI 호환용 인자. 타입 추론 보조 용도로만 사용",
    )
    parser.add_argument(
        "--template-prefix",
        type=str,
        default=None,
        help="이전 CLI 호환용 인자. 타입 추론 보조 용도로만 사용",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="최종 전처리 결과 저장 폴더 경로",
    )
    parser.add_argument(
        "--final-output-dir",
        type=Path,
        default=None,
        help="이전 CLI 호환용 alias. 지정 시 output-dir보다 우선",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help="템플릿 오버레이 결과 저장 폴더 경로",
    )
    parser.add_argument(
        "--overlay-output-dir",
        type=Path,
        default=None,
        help="이전 CLI 호환용 alias. 지정 시 overlay-dir보다 우선",
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
        help="템플릿 오버레이 투명도 (0~1)",
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
        default=None,
        help="중간 결과 이미지 저장 폴더 경로",
    )
    parser.add_argument(
        "--no-stage-images",
        action="store_true",
        help="중간 결과 이미지 저장 비활성화",
    )

    args = parser.parse_args()

    hint_input_dir = args.input_dir if args.input_dir is not None else args.template_result_dir
    hint_input_dir = hint_input_dir if hint_input_dir is not None else Path("")

    catheter_type = (
        args.catheter_type
        if args.catheter_type != "auto"
        else infer_catheter_type(
            input_dir=hint_input_dir,
            template_path=args.template,
            template_prefix=args.template_prefix,
        )
    )

    input_dir = args.input_dir if args.input_dir is not None else DEFAULT_INPUT_DIRS[catheter_type]
    pattern = args.pattern if args.pattern is not None else DEFAULT_PATTERNS[catheter_type]
    template_path = args.template if args.template is not None else DEFAULT_TEMPLATE_PATHS[catheter_type]
    output_dir = resolve_output_dir(catheter_type, args.output_dir, args.final_output_dir)
    overlay_dir = resolve_overlay_dir(catheter_type, args.overlay_dir, args.overlay_output_dir)

    # pro3는 예전 CLI의 output-dir/final-output-dir 조합도 stage/final로 받아들인다.
    stage_dir = resolve_stage_dir(catheter_type, args.stage_dir, args.output_dir, args.final_output_dir)
    save_stage_images = stage_dir is not None and not args.no_stage_images

    print("\n=================================== 공용 전처리 디스패처 ==================================")
    print(f"type:      {catheter_type}")
    print(f"input:     {input_dir}")
    print(f"pattern:   {pattern}")
    print(f"template:  {template_path}")
    print(f"output:    {output_dir}")
    print(f"overlay:   {overlay_dir}")
    if stage_dir is not None:
        print(f"stage:     {stage_dir}")
        print(f"save-stage:{save_stage_images}")
    print("===========================================================================================\n")

    run_preprocess(
        catheter_type=catheter_type,
        input_dir=input_dir,
        pattern=pattern,
        template_path=template_path,
        output_dir=output_dir,
        overlay_dir=overlay_dir,
        crop_size=tuple(args.crop_size),
        alpha=args.alpha,
        scale_adjust=args.scale_adjust,
        save_overlay=not args.no_overlay,
        stage_dir=stage_dir,
        save_stage_images=save_stage_images,
    )


if __name__ == "__main__":
    main()
