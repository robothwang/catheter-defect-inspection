# CLAUDE.md — Catheter Preprocessing

## 하네스: 카테터 전처리 파이프라인 개발

**목표:** 분석-구현-검증 3인 팀이 카테터 전처리 파이프라인의 버그 수정·기능 추가·리팩토링을 안정적으로 수행한다.

**트리거:** 전처리, 파이프라인, 카테터, 마스크, 정렬, holealign, pro1/pro2/pro3, IoU, 버그, 수정, 새 타입 추가, 코드 구현 관련 요청 시 `catheter-preprocess` 스킬을 사용하라. 단순 코드 질문은 직접 응답 가능.

GID, 특성 방향(characteristic orientation), 회전 정규화 전처리 요청 시 `gid-preprocess` 스킬을 사용하라.

데이터셋 구축, 라벨링, train/val/test 분할 요청 시 `dataset-builder` 스킬을 사용하라.

분류 실험, 모델 학습, CNN/ORN/RotEqNet, accuracy 측정, 전처리 비교 실험, 연구 파이프라인 관련 요청 시 `research-pipeline` 스킬을 사용하라.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-04-23 | 초기 하네스 구성 | 전체 | 카테터 전처리 파이프라인 개발 자동화 |
| 2026-04-24 | 연구 파이프라인 하네스 추가 | 전체 | 루멘 분류 전처리 비교 실험 자동화 (Raw/Holealign/GID × CNN/ORN/RotEqNet) |
