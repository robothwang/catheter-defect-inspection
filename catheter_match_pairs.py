from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import make_matching_plot, read_image


torch.set_grad_enabled(False)


def normalize_resize(resize_values):
    """match_pairs.py와 같은 방식으로 resize 인자를 정리한다."""
    if len(resize_values) == 2 and resize_values[1] == -1:
        resize_values = resize_values[0:1]

    if len(resize_values) == 2:
        print('Will resize to {}x{} (WxH)'.format(resize_values[0], resize_values[1]))
    elif len(resize_values) == 1 and resize_values[0] > 0:
        print('Will resize max dimension to {}'.format(resize_values[0]))
    elif len(resize_values) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    return resize_values


def build_matching_model(opt, device):
    """SuperPoint + SuperGlue 조합 모델을 생성한다."""
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        },
    }
    return Matching(config).eval().to(device)


def estimate_homography_metrics(mkpts0, mkpts1):
    """정답 pose가 없을 때 사용할 수 있는 대체 평가 지표를 계산한다."""
    metrics = {
        'homography_found': False,
        'homography_inliers': 0,
        'homography_inlier_ratio': 0.0,
        'mean_reprojection_error': np.nan,
        'homography': None,
        'inlier_mask': None,
    }

    if len(mkpts0) < 4 or len(mkpts1) < 4:
        return metrics

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return metrics

    mask = mask.reshape(-1).astype(bool)
    inlier_count = int(mask.sum())
    inlier_ratio = float(inlier_count / max(len(mask), 1))

    projected = cv2.perspectiveTransform(mkpts0.reshape(-1, 1, 2), H).reshape(-1, 2)
    reproj_errors = np.linalg.norm(projected - mkpts1, axis=1)
    mean_error = float(np.mean(reproj_errors[mask])) if inlier_count > 0 else np.nan

    metrics.update({
        'homography_found': True,
        'homography_inliers': inlier_count,
        'homography_inlier_ratio': inlier_ratio,
        'mean_reprojection_error': mean_error,
        'homography': H,
        'inlier_mask': mask,
    })
    return metrics


def save_match_result_npz(save_path, pred, homography_metrics, source_path, template_path):
    """후속 분석용으로 매칭 결과를 npz 파일에 저장한다."""
    payload = {
        'template_path': str(template_path),
        'source_path': str(source_path),
        'keypoints0': pred['keypoints0'],
        'keypoints1': pred['keypoints1'],
        'matches0': pred['matches0'],
        'matching_scores0': pred['matching_scores0'],
        'homography_found': np.array([homography_metrics['homography_found']], dtype=np.uint8),
        'homography_inliers': np.array([homography_metrics['homography_inliers']], dtype=np.int32),
        'homography_inlier_ratio': np.array([homography_metrics['homography_inlier_ratio']], dtype=np.float32),
    }

    if homography_metrics['homography'] is not None:
        payload['homography'] = homography_metrics['homography']
    if homography_metrics['inlier_mask'] is not None:
        payload['homography_inlier_mask'] = homography_metrics['inlier_mask'].astype(np.uint8)
    if not np.isnan(homography_metrics['mean_reprojection_error']):
        payload['mean_reprojection_error'] = np.array([homography_metrics['mean_reprojection_error']], dtype=np.float32)

    np.savez(str(save_path), **payload)


def append_summary_row(csv_path, row, write_header=False):
    """pair별 요약 지표를 CSV에 누적 저장한다."""
    fieldnames = [
        'template_image',
        'source_image',
        'template_keypoints',
        'source_keypoints',
        'matched_keypoints',
        'match_ratio_vs_template',
        'match_ratio_vs_source',
        'mean_confidence',
        'max_confidence',
        'homography_found',
        'homography_inliers',
        'homography_inlier_ratio',
        'mean_reprojection_error',
    ]

    with csv_path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Catheter template-source matching example with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path('/home/hjj747/SuperGluePretrainedNetwork/data/raw/targets/pro_3'),
        help='source 이미지 폴더 경로',
    )
    parser.add_argument(
        '--template-image',
        type=Path,
        default=Path('/home/hjj747/SuperGluePretrainedNetwork/data/raw/templates/pro_3_endpoint.png'),
        help='template 이미지 경로',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.png',
        help='source 이미지 파일 패턴',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/home/hjj747/SuperGluePretrainedNetwork/data/processed/match_results/pro3_preprocessed_template_vs_aligned_gray'),
        help='매칭 결과 저장 폴더',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=-1,
        help='처리할 source 이미지 최대 개수',
    )
    parser.add_argument(
        '--resize',
        type=int,
        nargs='+',
        default=[640, 480],
        help='입력 이미지 resize 설정',
    )
    parser.add_argument(
        '--resize-float',
        action='store_true',
        help='uint8 -> float 변환 뒤 resize 수행',
    )
    parser.add_argument(
        '--superglue',
        choices={'indoor', 'outdoor'},
        default='indoor',
        help='사용할 SuperGlue weight',
    )
    parser.add_argument(
        '--max-keypoints',
        type=int,
        default=1024,
        help='SuperPoint 최대 keypoint 수',
    )
    parser.add_argument(
        '--keypoint-threshold',
        type=float,
        default=0.005,
        help='SuperPoint keypoint threshold',
    )
    parser.add_argument(
        '--nms-radius',
        type=int,
        default=4,
        help='SuperPoint NMS 반경',
    )
    parser.add_argument(
        '--sinkhorn-iterations',
        type=int,
        default=20,
        help='SuperGlue sinkhorn iteration 수',
    )
    parser.add_argument(
        '--match-threshold',
        type=float,
        default=0.2,
        help='SuperGlue match threshold',
    )
    parser.add_argument(
        '--viz',
        action='store_true',
        help='매칭 시각화 이미지 저장',
    )
    parser.add_argument(
        '--show-keypoints',
        action='store_true',
        help='매칭 시각화 시 keypoint도 함께 표시',
    )
    parser.add_argument(
        '--viz-extension',
        type=str,
        default='png',
        choices=['png', 'pdf'],
        help='시각화 파일 확장자',
    )
    parser.add_argument(
        '--fast-viz',
        action='store_true',
        help='OpenCV 기반 빠른 시각화 사용',
    )
    parser.add_argument(
        '--opencv-display',
        action='store_true',
        help='OpenCV로 화면 출력 후 저장',
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='CPU 강제 사용',
    )

    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    opt.resize = normalize_resize(opt.resize)

    if not opt.source_dir.exists():
        raise FileNotFoundError(f'source dir not found: {opt.source_dir}')
    if not opt.template_image.exists():
        raise FileNotFoundError(f'template image not found: {opt.template_image}')

    source_paths = sorted(opt.source_dir.glob(opt.pattern))
    if opt.max_length > -1:
        source_paths = source_paths[:min(len(source_paths), opt.max_length)]
    if not source_paths:
        raise FileNotFoundError(f'no source images found in {opt.source_dir} with pattern {opt.pattern}')

    matches_dir = opt.output_dir / 'matches'
    metrics_dir = opt.output_dir / 'metrics'
    viz_dir = opt.output_dir / 'visualizations'
    matches_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    if opt.viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = metrics_dir / 'summary.csv'
    if summary_csv_path.exists():
        summary_csv_path.unlink()

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print(f'Running inference on device "{device}"')
    matching = build_matching_model(opt, device)

    template_image, template_inp, _ = read_image(
        opt.template_image,
        device,
        opt.resize,
        0,
        opt.resize_float,
    )
    if template_image is None:
        raise RuntimeError(f'failed to read template image: {opt.template_image}')

    print('===========================================================')
    print('Catheter SuperGlue Matching Start')
    print(f'Template : {opt.template_image}')
    print(f'Source   : {opt.source_dir} ({opt.pattern})')
    print(f'Output   : {opt.output_dir}')
    print(f'Pairs    : {len(source_paths)}')
    print('===========================================================')

    for index, source_path in enumerate(source_paths, start=1):
        source_image, source_inp, _ = read_image(
            source_path,
            device,
            opt.resize,
            0,
            opt.resize_float,
        )
        if source_image is None:
            print(f'[{index}/{len(source_paths)}] FAIL {source_path.name} :: image read failed')
            continue

        pred = matching({'image0': template_inp, 'image1': source_inp})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        kpts0 = pred['keypoints0']
        kpts1 = pred['keypoints1']
        matches0 = pred['matches0']
        scores0 = pred['matching_scores0']

        valid = matches0 > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]
        mconf = scores0[valid]

        homography_metrics = estimate_homography_metrics(mkpts0, mkpts1)

        pair_stem = f'{opt.template_image.stem}__{source_path.stem}'
        match_npz_path = matches_dir / f'{pair_stem}_matches.npz'
        save_match_result_npz(match_npz_path, pred, homography_metrics, source_path, opt.template_image)

        mean_conf = float(np.mean(mconf)) if len(mconf) > 0 else 0.0
        max_conf = float(np.max(mconf)) if len(mconf) > 0 else 0.0
        row = {
            'template_image': opt.template_image.name,
            'source_image': source_path.name,
            'template_keypoints': int(len(kpts0)),
            'source_keypoints': int(len(kpts1)),
            'matched_keypoints': int(len(mkpts0)),
            'match_ratio_vs_template': float(len(mkpts0) / max(len(kpts0), 1)),
            'match_ratio_vs_source': float(len(mkpts0) / max(len(kpts1), 1)),
            'mean_confidence': mean_conf,
            'max_confidence': max_conf,
            'homography_found': int(homography_metrics['homography_found']),
            'homography_inliers': int(homography_metrics['homography_inliers']),
            'homography_inlier_ratio': float(homography_metrics['homography_inlier_ratio']),
            'mean_reprojection_error': homography_metrics['mean_reprojection_error'],
        }
        append_summary_row(summary_csv_path, row, write_header=(index == 1))

        if opt.viz:
            color = cm.jet(mconf) if len(mconf) > 0 else np.zeros((0, 4))
            text = [
                'SuperGlue catheter matching',
                f'Template keypoints: {len(kpts0)}',
                f'Source keypoints: {len(kpts1)}',
                f'Matches: {len(mkpts0)}',
                f'H inliers: {homography_metrics["homography_inliers"]}',
            ]
            small_text = [
                f'Template: {opt.template_image.name}',
                f'Source: {source_path.name}',
                f'Mean conf: {mean_conf:.3f}',
                f'Inlier ratio: {homography_metrics["homography_inlier_ratio"]:.3f}',
            ]
            viz_path = viz_dir / f'{pair_stem}_matches.{opt.viz_extension}'
            make_matching_plot(
                template_image,
                source_image,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                viz_path,
                opt.show_keypoints,
                opt.fast_viz,
                opt.opencv_display,
                'Template vs Source',
                small_text,
            )

        print(
            f'[{index}/{len(source_paths)}] OK   {source_path.name:<18} '
            f'matches={len(mkpts0):4d} mean_conf={mean_conf:.3f} '
            f'inliers={homography_metrics["homography_inliers"]:4d} '
            f'inlier_ratio={homography_metrics["homography_inlier_ratio"]:.3f}'
        )

    print('===========================================================')
    print('Catheter SuperGlue Matching Done')
    print(f'Match npz : {matches_dir}')
    print(f'Metrics   : {summary_csv_path}')
    if opt.viz:
        print(f'Viz       : {viz_dir}')
    print('===========================================================')
