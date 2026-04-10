# Changelog

이 문서는 프로젝트의 주요 변경 이력을 기록합니다.

## [0.3.2] - 2026-04-11

### Added
- `scripts/preprocess_source.py`에서 `pro1/pro2/pro3` 공통으로 `--stage-dir`, `--no-stage-images` 흐름을 지원하도록 정리
- `pro1/pro2/pro3` source 전처리 단계별 결과 저장 경로 추가
  - `data/processed/stage_images/pro1_holealign`
  - `data/processed/stage_images/pro2_holealign`
  - `data/processed/stage_images/pro3_holealign`

### Changed
- `scripts/preprocess_source.py`를 타입별 `holealign` 스크립트를 import 해서 실행하는 공용 디스패처 구조로 정리
- `scripts/preprocess_pro1_holealign.py`, `scripts/preprocess_pro2_holealign.py`, `scripts/preprocess_pro3_holealign.py`가 모두 전처리된 템플릿 결과 폴더를 직접 읽도록 통일
- `pro1/pro2/pro3` `holealign` 스크립트의 코드 흐름과 주석 스타일을 `new.py` 기준으로 재정리
- `pro1` 단계별 결과를 `2-hole` 구조에 맞춰 저장하도록 정리
  - `07_*_lumen_hole1_mask.png`
  - `08_*_lumen_hole2_mask.png`
  - `09_*_lumen_all_mask.png`
- `pro2` 단계별 결과를 `3-lumen` 구조에 맞춰 정리
  - `07_*_lumen_big_mask.png`
  - `08_*_lumen_small_mask.png`
  - `09_*_lumen_all_mask.png`
  - `16_*_final_crop_600.png`
  - `17_*_template_overlay.png`
- `pro3` 단계별 결과 번호를 `new.py` 흐름 기준으로 유지
  - `10_*_aligned_gray.png`
  - `11_*_aligned_outer_mask.png`
  - `12_*_aligned_section_mask.png`
  - `16_*_final_crop_600.png`
  - `17_*_template_overlay.png`
- `README.MD`를 현재 권장 실행 구조(`preprocess_source.py` + 타입별 `holealign`) 기준으로 갱신

### Fixed
- `preprocess_source.py --catheter-type pro2` 실행 시 `--stage-dir`가 실제 `pro2` 파이프라인으로 전달되지 않던 문제 수정
- `pro1/pro2` source 전처리에서 템플릿 경로는 맞지만 템플릿 처리 구조가 분산돼 있던 부분을 정리
- `pro1/pro2/pro3` 직접 실행과 `preprocess_source.py` 경유 실행 간 출력 구조가 달라지던 부분을 정리

## [0.3.1] - 2026-04-10

### Added
- `scripts/preprocess_pro3_source.py`에 정렬 결과 `aligned_outer_mask` 저장 단계 추가
  - `10_*_aligned_gray.png`
  - `11_*_aligned_outer_mask.png`
  - `12_*_aligned_section_mask.png`

### Changed
- `pro3` source 단계별 출력 번호를 `aligned_outer_mask` 추가에 맞춰 조정
  - `aligned_lumen_big_mask`: `13_*`
  - `aligned_lumen_small_mask`: `14_*`
  - `aligned_lumen_all_mask`: `15_*`
  - 디버그 폴더 내 `final_crop`: `16_*`
  - 디버그 폴더 내 `template_overlay`: `17_*`
- `README.MD`에 `11_*_aligned_outer_mask.png` 산출물 설명 추가
- `data/raw/targets/pro_3` 전체 10장에 대해 `preprocess_pro3_source.py`를 다시 실행해 새 정렬 외곽 마스크 산출물 반영

## [0.3.0] - 2026-04-09

### Added
- `scripts/preprocess_tamplate.py`에 endpoint 템플릿 전처리 파이프라인 정리
- `scripts/preprocess_source.py`에 source 전처리/회전 탐색/최종 크롭 파이프라인 정리
- `scripts/preprocess_pro3_source.py`에 `pro3` source 전용 독립 실행 파이프라인 추가
- 템플릿 전처리 결과 재사용 구조 추가
  - `processed_templates/pro1_endpoint`
  - `processed_templates/pro2_endpoint`
  - `processed_templates/pro3_endpoint`
- `pro3` source 최종 결과물 분리 저장 경로 추가
  - `data/processed/processed_images/pro3_source`
  - `data/processed/overlay_images/pro3_source`
- 템플릿/정렬 마스크 중심점 비교용 오버레이 마커 추가

### Changed
- `preprocess_pro3_holealign.py` 의존 없이 필요한 로직을 `preprocess_pro3_source.py` 내부 함수로 분리
- `preprocess_tamplate.py`의 template lumen mask 생성 방식을 grayscale threshold 기반 탐색에서 `section_mask - outer_mask` 기반 생성으로 단순화
- `pro1/pro2/pro3` endpoint 템플릿 모두에서 lumen mask를 타입별 구조에 맞게 저장하도록 확장
  - `pro1`: `hole1/hole2/all`
  - `pro2`: `big/small/all`
  - `pro3`: `big/small/all`
- `preprocess_source.py`, `preprocess_pro3_source.py` 실행 로그를 `OK/FAIL + geom + hole` 요약 형식으로 정리
- 오버레이 저장 방식을 크롭본 기준에서 템플릿 전체 좌표계 기준으로 변경
- 최종 크롭 생성 시 단면 bbox를 기준으로 자동 축소 후 크롭하도록 변경

### Fixed
- 템플릿 lumen mask가 왜곡된 contour 형태로 생성되던 문제 개선
- 템플릿 mask를 다시 원본 영상 threshold로 찾으면서 생기던 불안정성 개선
- `640x640` 최종 크롭에서 단면 전체가 프레임 밖으로 잘려 보이던 문제 개선
- 오버레이 결과가 크롭/축소본 기준으로 생성되어 템플릿과의 정합 확인이 어려웠던 문제 개선
- 오버레이에서 템플릿 중심과 정렬된 마스크 중심이 일치하는지 바로 확인하기 어려웠던 문제 개선
- 최종 결과 이미지와 오버레이 이미지가 단계별 디버그 결과 폴더에만 저장되던 문제 개선

### Removed
- 템플릿 lumen mask 생성에 사용되던 불필요한 grayscale high-threshold 탐색 의존성 제거
- `preprocess_pro3_holealign.py` import 의존 구조 제거
- 템플릿 전처리 결과를 source 파이프라인에서 다시 계산하던 중복 흐름 제거

## [0.2.0] - 2026-04-01

### Added
- `scripts/preprocess_pro1_holealign.py` 추가
- `scripts/preprocess_pro2_holealign.py` 추가
- `scripts/preprocess_pro3_holealign.py` 추가
- `pro1`용 hole-mask 기반 정렬/오버레이 파이프라인 추가

### Changed
- `.gitignore` 업데이트
- `scripts/preprocess_pro1.py` 수정
- `scripts/preprocess_pro2.py` 수정
- `pro3_holealign` 방식(템플릿 등록 + hole 마스크 정렬)을 `pro1/pro2`에 확장 적용

### Fixed
- `pro1`에서 카테터 대신 배경 무늬가 외곽으로 선택되던 오탐 문제 개선
- `pro1` 정렬 품질 저하(center/scale/rotation 오류) 케이스 개선
- `pro1` 루멘 정렬 점수 저하 케이스 개선

### Removed
- `scripts/overlay_with_template.py` 제거
- `scripts/preprocess_pro1_new.py` 제거
- `scripts/preprocess_pro3.py` 제거

## [0.1.0] - 2026-03-31

### Added
- 초기 전처리 스크립트 구조 구성
- 템플릿 오버레이 기반 시각화 흐름 구성

### Changed
- 데이터셋별(`pro1`, `pro2`, `pro3`) 전처리 전략 분리 시작

### Fixed
- 기본 전처리 실행 경로/출력 경로 정리

### Removed
- 없음
