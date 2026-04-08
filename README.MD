# Catheter Defect Inspection Preprocessing
![cross section of the infusion catheter](./img/catheter_01.png)

카테터 단면 이미지 전처리 및 템플릿 오버레이 확인용 스크립트 모음이며, 연구 중 입니다.

## 프로그램 소개
이 프로젝트는 카테터 단면 이미지를 학습/분석에 활용하기 쉬운 형태로 정렬하고,
템플릿 기준으로 결과를 빠르게 점검할 수 있도록 만든 전처리 도구입니다.

민감할 수 있는 세부 파라미터나 내부 기준값은 문서에 상세히 기재하지 않고,
실행 흐름과 사용 목적 중심으로만 안내합니다.

## 최근 변경 사항 요약
- 템플릿 전처리와 source 전처리를 별도 스크립트로 분리했습니다.
- 템플릿 lumen mask 생성 방식을 `section_mask - outer_mask` 기반으로 단순화했습니다.
- `pro1`, `pro2`, `pro3` endpoint 템플릿 모두에서 타입별 lumen mask를 저장하도록 확장했습니다.
- `pro3` source 파이프라인에서 회전 정렬, 최종 crop, 템플릿 오버레이를 독립적으로 재현할 수 있게 정리했습니다.
- `pro3` 오버레이에는 템플릿 중심과 정렬된 마스크 중심을 서로 다른 마커로 표시해 정합 상태를 바로 확인할 수 있습니다.

## 사용된 방식(요약)
- 템플릿 endpoint 이미지에서 외곽/section/lumen mask를 생성
- source 이미지를 템플릿 좌표계에 맞게 scale + shift
- source lumen mask와 template lumen mask의 IoU 기반으로 최적 회전각 탐색
- 단면 전체가 crop 안에 들어오도록 자동 축소 후 최종 결과 저장
- 템플릿 전체 좌표계 위에 오버레이하여 정렬 상태 시각적 확인

`pro1`, `pro2`, `pro3`는 단면 형태가 달라 동일 파이프라인 틀을 유지하되,
각 타입에 맞는 정렬 로직을 개별 스크립트에서 적용합니다.

## 주요 스크립트
- `scripts/preprocess_template.py`
  - `pro1/pro2/pro3` endpoint 템플릿 전처리
  - grayscale, binary, outer mask, section mask, lumen mask 저장
- `scripts/preprocess_pro3_source.py`
  - `pro3` source 이미지 전처리
  - template 좌표계 배치, lumen 검출, IoU 기반 회전 정렬, 최종 crop, 오버레이 생성
- `scripts/preprocess_pro1_holealign.py`
  - `pro1`용 hole-mask 기반 정렬 파이프라인
- `scripts/preprocess_pro2_holealign.py`
  - `pro2`용 hole-mask 기반 정렬 파이프라인
- `scripts/preprocess_pro3_holealign.py`
  - 기존 `pro3` holealign 기준 파이프라인

## 템플릿 전처리 출력
- 템플릿 전처리 결과는 `data/processed/processed_templates/<type>_endpoint` 에 저장됩니다.
- 공통 저장 항목
  - `01_*_grayscale.png`
  - `02_*_binary.png`
  - `03_*_main_component.png`
  - `04_*_outer_mask.png`
  - `05_*_section_mask.png`
- lumen mask 저장 항목
  - `pro1`: `06_*_lumen_hole1_mask.png`, `07_*_lumen_hole2_mask.png`, `08_*_lumen_all_mask.png`
  - `pro2`: `06_*_lumen_big_mask.png`, `07_*_lumen_small_mask.png`, `08_*_lumen_all_mask.png`
  - `pro3`: `06_*_lumen_big_mask.png`, `07_*_lumen_small_mask.png`, `08_*_lumen_all_mask.png`

## pro3 Source 전처리 출력
- 단계별 디버그/중간 결과
  - `data/processed/processed_pro3_source/<image_name>`
- 최종 전처리 이미지
  - `data/processed/processed_images/pro3_source`
- 템플릿 오버레이 이미지
  - `data/processed/overlay_images/pro3_source`

대표 산출물:
- `15_*_final_crop_640.png`: 최종 전처리 결과
- `16_*_template_overlay.png`: 템플릿 전체 기준 오버레이 결과
  - 템플릿 중심: 빨간 `cross`
  - 정렬된 마스크 중심: 초록 `diamond`

## 환경
|속성|버전|
|------|---|
|**OS**|Ubuntu 24.04.4 LTS (nobel)|
|**Python**|3.10 +|
|**CUDA**|12.6|
|**cuDNN**|9.14.0.64|
|**OpenCV**|`cv2`|
|**NumPy**|2.4.3|
|**venv**|`miniforge3` or `conda`|

## 실행 예시
```bash
conda run -n yolo26 python scripts/preprocess_template.py
conda run -n yolo26 python scripts/preprocess_pro1_holealign.py
conda run -n yolo26 python scripts/preprocess_pro2_holealign.py
conda run -n yolo26 python scripts/preprocess_pro3_holealign.py
conda run -n yolo26 python scripts/preprocess_pro3_source.py --alpha 0.55
```

## `preprocess_pro3_source.py` 실행 인자 예시
```bash
conda run -n yolo26 python scripts/preprocess_pro3_source.py \
  --pattern "*.png" \
  --crop-size 640 640 \
  --alpha 0.55
```

주요 인자:
- `--input-dir`: source 이미지 폴더
- `--template-result-dir`: 전처리된 템플릿 결과 폴더
- `--output-dir`: 단계별 중간 결과 저장 폴더
- `--final-output-dir`: 최종 전처리 이미지 저장 폴더
- `--overlay-output-dir`: 템플릿 오버레이 이미지 저장 폴더
- `--crop-size`: 최종 crop 크기
- `--alpha`: 오버레이 투명도

## 해결된 주요 문제
- 템플릿 lumen mask가 grayscale threshold 의존으로 왜곡되던 문제를 `section - outer` 기반 생성으로 정리
- 템플릿 전처리 결과를 source 파이프라인에서 재계산하던 중복 제거
- `640x640` crop에서 단면 일부만 보이던 문제를 자동 축소 후 crop으로 개선
- 오버레이가 crop/축소본 기준으로 생성되던 문제를 템플릿 전체 기준 오버레이로 수정
- 템플릿 중심과 정렬된 마스크 중심이 일치하는지 한눈에 확인할 수 있도록 마커 추가

## 참고 이미지 출처
- 카테터 단면 결함 검사 이해를 돕기 위한 이미지 출처:
  [https://www.mtnews.net/news/articleView.html?idxno=16856](https://www.mtnews.net/news/articleView.html?idxno=16856)

## 참고
- 대용량 데이터(`data/raw`, `data/processed`)와 가중치(`weights`)는 `.gitignore`로 제외되어 있습니다.

## License
This project is licensed under the MIT License. See [LICENSE](./LICENSE).
