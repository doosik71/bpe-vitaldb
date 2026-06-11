# generate-overview.py

`scripts/generate-overview.py`는 `data/models` 아래에 저장된 모델 구조/평가 결과를 수집해서
모델 간 성능 비교 그래프를 생성하는 스크립트다.

이 문서는 아래 내용을 설명한다.

- 스크립트의 목적
- 입력 데이터 구조
- 내부 처리 흐름
- 생성되는 그래프 종류
- CLI 사용법
- 출력 파일 규칙
- 주의사항과 해석 팁

## 1. 목적

이 스크립트는 여러 모델의 다음 정보를 한 번에 비교하기 위한 overview 그래프를 생성한다.

- trainable parameter count
- SBP/DBP metric (`MAE`, `ME`, `SD`, `RMSE`)
- inference time (`ms / sample`)

현재 출력은 두 계열로 나뉜다.

1. `plot_*`
   - scatter 기반 비교 그래프
   - `png`와 `html` 출력 지원
2. `bar_*`
   - 모델 순위형 수직 bar 그래프
   - `png`만 출력

## 2. 입력 데이터 구조

기본 입력 루트는 `data/models`다.

각 모델 디렉터리는 대략 아래 구조를 가진다.

```text
data/models/
├── acfa/
│   ├── struct.txt
│   └── eval_results.json
├── ae_lstm/
│   ├── struct.txt
│   └── eval_results.json
└── ...
```

### 2.1 `struct.txt`

이 파일에서 아래 형식의 줄을 찾아 trainable parameter count를 파싱한다.

```text
Trainable params: 1,234,567
```

스크립트는 정규식 `Trainable params:\s*([\d,]+)`로 값을 추출한다.

### 2.2 `eval_results.json`

평가 결과 JSON은 최소한 아래 구조를 포함해야 한다.

```json
{
  "sbp": {
    "mae": 8.12,
    "me": -0.31,
    "sd": 10.42,
    "rmse": 10.43
  },
  "dbp": {
    "mae": 5.27,
    "me": 0.14,
    "sd": 6.81,
    "rmse": 6.81
  },
  "avg_ms_per_sample": 0.93
}
```

`avg_ms_per_sample`는 없을 수 있다. 이 경우 inference time 그래프에서만 제외된다.

## 3. 제외 규칙

현재 `EXCLUDE_MODELS`에 포함된 모델은 그래프 생성 대상에서 제외된다.

현재 기본 제외 모델:

- `naive`

또한 아래 경우는 자동으로 skip된다.

- 모델 디렉터리가 아님
- `struct.txt` 없음
- parameter count 파싱 실패
- `eval_results.json` 없음

이 경우 스크립트는 경고를 출력하고 다음 모델로 진행한다.

## 4. 내부 처리 흐름

스크립트의 큰 흐름은 아래와 같다.

1. CLI 인자 파싱
2. `models_dir` 하위 모델 디렉터리 순회
3. `struct.txt`에서 parameter count 파싱
4. `eval_results.json`에서 SBP/DBP metric 및 inference time 로드
5. 로드된 레코드 목록으로 그래프 생성
6. `output_dir`에 파일 저장

핵심 함수 역할은 아래와 같다.

### 4.1 데이터 로딩

- `parse_args()`
  - `--models-dir`, `--output-dir`, `--format` 파싱
- `_parse_param_count(struct_path)`
  - `struct.txt`에서 trainable parameter count 추출
- `load_model_data(models_dir)`
  - 전체 모델을 순회하여 내부 레코드 리스트 생성

생성되는 내부 레코드 예시는 아래와 같다.

```python
{
    "model": "acfa",
    "n_params": 1234567,
    "sbp": {...},
    "dbp": {...},
    "avg_ms_per_sample": 0.93,
}
```

### 4.2 공통 보조 함수

- `_param_formatter()`
  - parameter count를 `15K`, `2.18M` 같은 표시 형식으로 변환
- `_plotly_tick_values()`
  - HTML plot의 x축 tick 값/레이블 생성
- `_annotate()`
  - PNG scatter plot에서 각 점 옆에 모델명을 표시
- `_sorted_bar_records()`
  - SBP/DBP metric 값 기준으로 bar 그래프 정렬
- `_sorted_inference_records()`
  - inference time 값 기준으로 bar 그래프 정렬
- `_bar_colors()`
  - bar 그래프 색상 목록 생성
- `_write_html()`
  - Plotly figure spec을 standalone HTML 파일로 저장

### 4.3 HTML figure builder

- `_build_metric_html_figure()`
  - `plot_mae.html`, `plot_me.html`, `plot_sd.html`, `plot_rmse.html`용 figure spec 생성
- `_build_inference_time_html_figure()`
  - `plot_inference_time.html`용 figure spec 생성

이 HTML 출력은 Plotly.js CDN을 로드하는 standalone HTML이다.

## 5. 생성되는 그래프 종류

## 5.1 Scatter overview PNG/HTML

### `plot_mae.png`, `plot_mae.html`
### `plot_me.png`, `plot_me.html`
### `plot_sd.png`, `plot_sd.html`
### `plot_rmse.png`, `plot_rmse.html`

구성:

- 하나의 파일에 좌우 2개 subplot
- 왼쪽: `SBP`
- 오른쪽: `DBP`
- x축: trainable parameter count (log scale)
- y축: 해당 metric 값

특징:

- PNG는 점과 모델명 annotation을 함께 그림
- HTML은 확대/축소, pan, hover, legend 토글 지원
- HTML은 모델명 label trace를 별도로 두고 `Labels On/Off` 버튼으로 제어
- HTML legend는 우측 세로 패널로 배치되어 그래프와 겹치지 않도록 구성됨

### `plot_inference_time.png`, `plot_inference_time.html`

구성:

- 단일 subplot
- x축: trainable parameter count (log scale)
- y축: `avg_ms_per_sample`

특징:

- PNG는 점과 모델명 annotation을 함께 그림
- HTML은 확대/축소, pan, hover, legend 토글 지원
- HTML은 metric 그래프와 동일하게 label trace와 `Labels On/Off` 버튼을 제공
- legend는 우측 세로 패널에 위치함

## 5.2 Ranking bar PNG

### `bar_mae.png`
### `bar_me.png`
### `bar_sd.png`
### `bar_rmse.png`

구성:

- 하나의 파일에 좌우 2개 subplot
- 왼쪽: `SBP`
- 오른쪽: `DBP`
- x축: 모델 이름
- y축: metric 값
- 각 subplot은 해당 값 기준 오름차순 정렬

주의:

- `SBP` 정렬 순서와 `DBP` 정렬 순서는 서로 다를 수 있다.
- 즉, 같은 파일 안에서도 좌/우 subplot의 모델 순서가 달라질 수 있다.

### `bar_inference_time.png`

구성:

- 단일 subplot
- x축: 모델 이름
- y축: `avg_ms_per_sample`
- inference time 값 기준 오름차순 정렬

## 6. CLI 사용법

기본 실행:

```bash
uv run python scripts/generate-overview.py
```

기본 wrapper 사용:

```bash
bin/generate-overview
```

모델 디렉터리 변경:

```bash
uv run python scripts/generate-overview.py --models-dir data/models
```

출력 디렉터리 변경:

```bash
uv run python scripts/generate-overview.py --output-dir images
```

HTML만 생성:

```bash
uv run python scripts/generate-overview.py --format html
```

PNG만 생성:

```bash
uv run python scripts/generate-overview.py --format png
```

둘 다 생성(기본값):

```bash
uv run python scripts/generate-overview.py --format both
```

## 7. 출력 파일 규칙

### 7.1 Scatter overview

```text
images/plot_mae.png
images/plot_mae.html
images/plot_me.png
images/plot_me.html
images/plot_sd.png
images/plot_sd.html
images/plot_rmse.png
images/plot_rmse.html
images/plot_inference_time.png
images/plot_inference_time.html
```

### 7.2 Ranking bar

```text
images/bar_mae.png
images/bar_me.png
images/bar_sd.png
images/bar_rmse.png
images/bar_inference_time.png
```

## 8. `--format` 동작

`--format`은 `plot_*` 계열에만 영향을 준다.

- `png`
  - `plot_*.png`와 `bar_*.png` 생성
  - HTML은 생성하지 않음
- `html`
  - `plot_*.html`만 생성
  - `bar_*.png`는 생성하지 않음
- `both`
  - `plot_*.png`, `plot_*.html`, `bar_*.png` 모두 생성

즉, bar 그래프는 현재 PNG 전용이다.

## 9. HTML 그래프 사용법

HTML 파일은 브라우저에서 열면 바로 상호작용 가능하다.

지원 기능:

- 마우스 drag 확대
- wheel zoom
- pan
- reset axes
- hover tooltip
- legend 클릭으로 특정 모델 표시/숨김
- `Labels On/Off` 버튼으로 모델명 표시 토글

### 주의: Plotly CDN 사용

현재 HTML은 아래 CDN을 로드한다.

```text
https://cdn.plot.ly/plotly-2.35.2.min.js
```

따라서 브라우저가 외부 CDN에 접근할 수 있어야 HTML 그래프가 정상 동작한다.
오프라인 완전 독립 HTML은 현재 구현 범위에 포함되지 않는다.

## 10. 해석 팁

### 10.1 Scatter overview 해석

- 왼쪽 아래로 갈수록 metric 관점에서 유리한 모델일 가능성이 높다.
- 단, `ME`는 0에 가까운 값이 더 바람직하므로 절대적으로 낮을수록 좋다고 해석하면 안 된다.
- parameter count는 log scale이므로 축 간 간격은 절대 차이가 아니라 상대 규모 차이를 반영한다.

### 10.2 Bar ranking 해석

- `MAE`, `SD`, `RMSE`, `inference time`은 일반적으로 낮을수록 유리하다.
- `ME`는 signed metric이므로 단순 오름차순 정렬이 곧 품질 순위를 뜻하지는 않는다.
  - 0에 가까운 모델을 따로 보고 싶다면 bar 값의 절대값 기준 재정렬이 필요하다.

## 11. 구현 제약 및 현재 가정

현재 스크립트는 아래를 가정한다.

- `struct.txt` 안에 `Trainable params:` 줄이 존재한다.
- `eval_results.json` 구조는 `sbp`, `dbp`, `avg_ms_per_sample` 키를 사용한다.
- 모델 비교 색상은 `matplotlib.cm.tab20`을 순환 사용한다.
- 모델 수가 많아도 HTML legend는 우측 세로 패널로 분리해 겹침을 줄인다.

## 12. 관련 파일

- 스크립트: [scripts/generate-overview.py](../scripts/generate-overview.py)
- 실행 wrapper (POSIX): [bin/generate-overview](../bin/generate-overview)
- 실행 wrapper (Windows): [bin/generate-overview.bat](../bin/generate-overview.bat)
