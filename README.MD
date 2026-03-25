# Catheter Defect Inspection Preprocessing

카테터 단면 이미지 전처리 및 템플릿 오버레이 확인용 스크립트 모음입니다.

## 프로그램 소개
이 프로젝트는 카테터 단면 이미지를 학습/분석에 활용하기 쉬운 형태로 정렬하고,
템플릿 기준으로 결과를 빠르게 점검할 수 있도록 만든 전처리 도구입니다.

민감할 수 있는 세부 파라미터나 내부 기준값은 문서에 상세히 기재하지 않고,
실행 흐름과 사용 목적 중심으로만 안내합니다.

## 사용된 방식(요약)
- 이미지 기본 전처리(그레이스케일 변환 등) 수행
- 단면 영역 중심 정렬 및 회전 보정
- 필요한 영역만 crop하여 결과 이미지 저장
- 템플릿 오버레이로 정렬 상태 시각적 확인

`pro1`, `pro2`, `pro3`는 단면 형태가 달라 동일 파이프라인 틀을 유지하되,
각 타입에 맞는 정렬 로직을 개별 스크립트에서 적용합니다.

## 구조
- `scripts/preprocess_pro1.py`: pro1 전처리(기존 방식)
- `scripts/preprocess_pro1_new.py`: pro1 템플릿 정합 실험 버전
- `scripts/preprocess_pro2.py`: pro2 전처리(템플릿 정합)
- `scripts/preprocess_pro3.py`: pro3 전처리(큰/작은 lumen 분리 정합)
- `scripts/overlay_with_template.py`: 템플릿 오버레이 확인

## 환경
- Python 3.10+
- OpenCV (`cv2`), NumPy
- (권장) `conda` 환경: `yolo26`

## 실행 예시
```bash
conda run -n yolo26 python scripts/preprocess_pro1.py
conda run -n yolo26 python scripts/preprocess_pro2.py
conda run -n yolo26 python scripts/preprocess_pro3.py
```

## 출력 경로
- 전처리 결과: `data/processed/processed_images/<type>`
- 오버레이 결과: `data/processed/overlay_images/<type>`

## 참고 이미지 출처
- 카테터 단면 결함 검사 이해를 돕기 위한 이미지 출처:
  [https://www.mtnews.net/news/articleView.html?idxno=16856](https://www.mtnews.net/news/articleView.html?idxno=16856)

## 참고
- 대용량 데이터(`data/raw`, `data/processed`)와 가중치(`weights`)는 `.gitignore`로 제외되어 있습니다.
