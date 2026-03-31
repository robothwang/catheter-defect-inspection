# Changelog

이 문서는 프로젝트의 주요 변경 이력을 기록합니다.

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
