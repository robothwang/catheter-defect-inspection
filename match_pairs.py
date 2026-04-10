#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch


from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


if __name__ == '__main__':
    # 이 스크립트의 전체 흐름:
    # 1) 이미지 쌍 목록을 읽는다.
    # 2) 각 이미지 쌍에서 SuperPoint로 keypoint/descriptor를 추출한다.
    # 3) SuperGlue로 두 이미지의 대응 keypoint를 매칭한다.
    # 4) 필요하면 pose 평가와 시각화를 수행한다.
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    # 옵션 간의 의존성을 검사한다.
    # 예를 들어 OpenCV 화면 출력은 시각화 기능이 켜져 있어야 의미가 있다.
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    # resize 인자를 정규화한다.
    # [640, -1] 형태는 [640]과 같은 의미로 취급한다.
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    with open(opt.input_pairs, 'r') as f:
        pairs = [l.split() for l in f.readlines()]

    # 최대 처리 개수를 제한한 경우 일부 pair만 사용한다.
    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    # 같은 순서로 재현 가능하게 섞는다.
    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    # 평가 모드에서는 각 줄에 카메라 정보와 정답 pose까지 포함되어야 한다.
    if opt.eval:
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # SuperPoint + SuperGlue 모델을 로드한다.
    # Matching 클래스 내부에서
    # - SuperPoint: keypoint / descriptor 추출
    # - SuperGlue: 두 이미지 간 descriptor 매칭
    # 을 연결된 파이프라인으로 처리한다.
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)

    # 결과 저장 디렉터리를 준비한다.
    input_dir = Path(opt.input_dir)
    print('Looking for data in directory \"{}\"'.format(input_dir))
    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    timer = AverageTimer(newline=True)
    for i, pair in enumerate(pairs):
        # pair의 앞 2개 값은 이미지 파일명이다.
        # 뒤에는 회전 정보 또는 평가용 카메라/pose 정보가 추가될 수 있다.
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # --cache를 사용하면 이미 계산된 npz/시각화 결과를 재사용한다.
        # 반복 실험 속도를 높이기 위한 장치이다.
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        if opt.cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                do_eval = False
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            timer.update('load_cache')

        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # pair 파일에 회전 정수(EXIF 등)가 있으면 이를 사용한다.
        # 입력 이미지가 저장 단계에서 회전된 경우를 보정하기 위한 정보이다.
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # 이미지 쌍을 읽고 모델 입력 텐서로 변환한다.
        # read_image 내부에서는 로드, grayscale 변환, resize, tensor 변환이 함께 수행된다.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float)
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
        timer.update('load_image')

        if do_match:
            # 실제 이미지 매칭 추론을 수행한다.
            # Matching({'image0', 'image1'}) 호출 한 번으로
            # 1) 각 이미지의 keypoint/descriptor 추출
            # 2) 두 descriptor 집합 간 대응점 매칭
            # 이 연속적으로 실행된다.
            pred = matching({'image0': inp0, 'image1': inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            matches, conf = pred['matches0'], pred['matching_scores0']
            timer.update('matcher')

            # 매칭 결과를 디스크에 저장한다.
            # matches0[i] = j  : image0의 i번째 keypoint가 image1의 j번째 keypoint와 매칭
            # matches0[i] = -1 : 유효한 대응점이 없음을 의미
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            np.savez(str(matches_path), **out_matches)

        # 매칭이 실제로 성립한 keypoint만 추려낸다.
        # kpts0, kpts1은 전체 검출점이고, mkpts0, mkpts1은 대응점만 남긴 결과이다.
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        if do_eval:
            # 평가 모드에서는 정답 카메라 정보와 비교해 기하적 성능을 측정한다.
            assert len(pair) == 38, 'Pair does not have ground truth info'
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # 이미지를 resize했기 때문에 intrinsic도 같은 비율로 보정해야 한다.
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # EXIF 회전이 있었다면 intrinsic / extrinsic도 함께 수정한다.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            # epipolar error를 기준으로 매칭이 기하적으로 올바른지 판단한다.
            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            correct = epi_errs < 5e-4
            num_correct = np.sum(correct)
            precision = np.mean(correct) if len(correct) > 0 else 0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            # 대응점으로부터 상대 pose(R, t)를 복원하고 정답과의 오차를 계산한다.
            thresh = 1.  # In pixels relative to resized image size.
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            if ret is None:
                err_t, err_R = np.inf, np.inf
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # 평가 결과를 npz로 저장한다.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            timer.update('eval')

        if do_viz:
            # 매칭 시각화:
            # 두 이미지의 keypoint와 대응선들을 그려 사람이 바로 결과를 확인할 수 있게 한다.
            # 선의 색은 매칭 confidence를 반영한다.
            color = cm.jet(mconf)
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # 어떤 threshold로 실행했는지 그림 하단에 함께 표시한다.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            timer.update('viz_match')

        if do_viz_eval:
            # 평가 시각화:
            # epipolar error가 작은 매칭과 큰 매칭을 색으로 구분해 보여준다.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # 사용한 threshold 정보도 함께 표시한다.
            k_thresh = matching.superpoint.config['keypoint_threshold']
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            timer.update('viz_eval')

        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    if opt.eval:
        # 모든 pair의 평가 결과를 모아 최종 평균 성능 표를 출력한다.
        # AUC@5/10/20은 pose error 임계값별 누적 성능을 의미한다.
        pose_errors = []
        precisions = []
        matching_scores = []
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            precisions.append(results['precision'])
            matching_scores.append(results['matching_score'])
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.*yy for yy in aucs]
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))
