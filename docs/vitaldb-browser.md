# `vitaldb-browser.py` 사용 및 개발 설명

작성일: 2026-06-08  
관련 코드: [scripts/vitaldb-browser.py](/home/doosik/work/bpe-vitaldb/scripts/vitaldb-browser.py:1)  
관련 문서: [README.md](/home/doosik/work/bpe-vitaldb/README.md:1)

---

## 1. 목적

`scripts/vitaldb-browser.py`는 원본 VitalDB `.vital` 파일을 직접 탐색하는 파형 브라우저다.

이 스크립트의 역할은 다음과 같다.

- 다운로드한 `.vital` 파일 목록을 보여준다.
- 케이스별 임상 메타데이터를 함께 표시한다.
- PPG, ABP, ECG II, numeric BP/HR 트랙을 한 창에서 탐색하게 한다.
- 시간 슬라이더와 키보드 단축키로 긴 수술 기록을 빠르게 이동하게 한다.
- 트랙 정보 창으로 전체 track inventory를 확인하게 한다.

즉, 이 도구는 전처리 이전의 원본 수술 신호를 검토하는 1차 브라우저다.

---

## 2. 입력과 출력

### 입력

- 원본 데이터 디렉터리: 기본값 `data/vitaldb`
- 파일 형식: `*.vital`

스크립트가 직접 또는 간접적으로 사용하는 데이터는 다음과 같다.

- 로컬 `.vital` 파일
- VitalDB 공개 clinical API
- VitalDB 공개 track index API

### 출력

- 별도 파일 저장 없음
- Tkinter + Matplotlib 기반 GUI 창
- 선택한 케이스의 waveform / numeric track 시각화

---

## 3. 실행 방법

### 기본 실행

```bash
uv run python scripts/vitaldb-browser.py
```

### 특정 케이스를 바로 열기

```bash
uv run python scripts/vitaldb-browser.py --case 1
```

### 데이터 디렉터리 변경

```bash
uv run python scripts/vitaldb-browser.py --data-dir data/vitaldb
```

옵션 설명:

- `--data-dir`
  - `.vital` 파일 디렉터리
  - 기본값: `data/vitaldb`
- `--case`
  - 시작 시 바로 열 케이스 ID
  - 예: `1`

---

## 4. UI 구성

### 왼쪽 패널

- 검색 입력창
- PPG / ABP 범례
- 정렬 가능한 케이스 목록
  - `Case`
  - `Dur.`
  - `Age`
  - `Sex`
  - `Size`
  - `Operation`
- 목록 상태 요약
  - 현재 필터 결과 수
  - PPG 포함 케이스 수
  - ABP 포함 케이스 수

### 오른쪽 패널

- Matplotlib 파형 캔버스
- 하단 시간 이동 바
  - `<< 60s`
  - `< 10s`
  - 슬라이더
  - `10s >`
  - `60s >>`
  - `Track Info`
  - 현재 시간 구간 표시

### 하단 상태 바

- 현재 케이스 ID
- 로드된 트랙 목록
- 길이
- 단축키 안내
- 에러 / 경고 메시지

---

## 5. 표시 대상 트랙

코드는 다음 트랙 정의를 기본으로 갖고 있다.

### waveform tracks

- `SNUADC/PLETH` → `PPG`
- `SNUADC/ART` → `ABP`
- `SNUADC/ECG_II` → `ECG II`

### numeric tracks

- `Solar8000/ART_SBP`
- `Solar8000/ART_DBP`
- `Solar8000/ART_MBP`
- `Solar8000/HR`

waveform은 500 Hz 기준으로 읽고, numeric은 1초 간격으로 읽는다.

실제 표시되는 것은 "해당 케이스에 존재하는 트랙만"이다.

---

## 6. 데이터 로딩 구조

### 6.1 파일 목록 수집

`list_vital_files(data_dir)`는 `*.vital` 파일을 모아 숫자 stem 기준으로 정렬한다.

### 6.2 임상 메타데이터 수집

`fetch_clinical_map(files)`는 VitalDB API에서 다음 필드를 가져온다.

- `caseid`
- `age`
- `sex`
- `opname`
- `caseend`

이 정보는 왼쪽 목록의 duration, age, sex, operation 컬럼을 채우는 데 쓰인다.

API 호출 실패 시 빈 dict를 반환하므로, 브라우저는 메타데이터 없이도 동작한다.

### 6.3 트랙 존재 여부 수집

`fetch_track_flags(files)`는 공개 `trks` 인덱스를 읽어
각 케이스의 `PPG`, `ABP` 존재 여부를 조사한다.

반환 형식:

```text
{caseid: (has_ppg, has_abp)}
```

이 결과는 목록 행의 색상 강조에 사용된다.

### 6.4 실제 케이스 데이터 로딩

`load_vital(path)`는 `VitalFile`을 열고,
표시 대상 track 중 실제로 존재하는 것만 골라 읽는다.

구현 방식:

- waveform track → `interval=1 / SRATE`
- numeric track → `interval=1`

그리고 `{track_name: ndarray}` dict로 변환해 돌려준다.

---

## 7. 목록 표시와 검색/정렬

### 검색

검색창은 다음 필드에 부분 일치를 적용한다.

- case ID
- operation name
- age
- sex

### 정렬

각 컬럼 헤더를 클릭하면 `_sort_by()`가 실행된다.

지원 정렬 키:

- case
- duration
- age
- sex
- size
- opname

같은 컬럼을 다시 누르면 오름차순 / 내림차순이 토글된다.

현재 정렬 컬럼 헤더에는 `▲` 또는 `▼`가 붙는다.

### 행 강조 색상

트랙 존재 여부에 따라 행이 다르게 표시된다.

- `none`
  - 일반 색상
- `ppg`
  - 녹색 계열 텍스트
- `abp`
  - 연한 붉은 배경
- `ppg_abp`
  - 녹색 텍스트 + 연한 붉은 배경

즉, 학습에 유용한 PPG/ABP 보유 케이스를 목록에서 빨리 찾을 수 있다.

---

## 8. 케이스 로딩과 플롯 구성

### 8.1 케이스 선택

사용자가 목록에서 케이스를 선택하면 `_load_case()`가 실행된다.

이 함수는:

- `.vital` 파일을 열고
- 표시 가능한 트랙을 읽고
- duration을 계산하고
- 현재 시간 시작점 `_t0`를 `0`으로 초기화하고
- Matplotlib figure 레이아웃을 새로 구성한다.

표시 가능한 트랙이 하나도 없으면 경고 메시지만 남기고 종료한다.

### 8.2 duration 계산

`duration_sec(data)`는:

- waveform 트랙이 있으면 waveform 길이를 우선 사용하고
- 없으면 numeric 트랙 길이를 사용한다.

즉, 가능한 한 파형 길이를 기준으로 전체 시간축을 잡는다.

### 8.3 figure 재구성

`_rebuild_figure()`는 케이스마다 figure를 처음부터 다시 만든다.

레이아웃 규칙:

- waveform track 하나당 1행
- numeric track가 하나라도 있으면 맨 아래에 numeric 통합 패널 1행 추가

예를 들어 PPG, ABP, ECG II, numeric BP/HR가 모두 있으면:

- PPG 축
- ABP 축
- ECG II 축
- numeric 축

순서로 배치된다.

### 8.4 임상 제목

figure 상단 제목에는 가능하면 다음 임상 정보가 붙는다.

- case ID
- age / sex
- height / weight
- operation name
- anesthesia type

이 정보는 케이스 로딩 시점에 VitalDB clinical API를 다시 호출해 가져온다.
실패하면 단순히 `Case {id}`만 표시한다.

---

## 9. 파형 그리기 방식

### 9.1 시간 창

브라우저는 전체 수술 기록 중 일부 시간 구간만 보여준다.

- 표시 창 길이: `WINDOW_SEC = 30`
- 기본 이동 단위: `STEP_SEC = 10`

즉, 한 번에 30초를 보고, 좌우 이동은 기본적으로 10초씩 한다.

### 9.2 waveform slice

`_wave_slice(name)`는 현재 시작 시각 `_t0` 기준으로:

- 시작 샘플 `i0 = int(_t0 * 500)`
- 끝 샘플 `i1 = i0 + 30초 * 500`

를 계산해 waveform 일부만 잘라낸다.

### 9.3 numeric slice

`_num_slice(name)`는 numeric track를 1초 단위로 잘라낸다.

즉:

- 시작 인덱스 `int(_t0)`
- 끝 인덱스 `int(_t0) + 30`

를 사용한다.

### 9.4 y축 범위

waveform 패널은 표시 구간의 유효값에 대해 `1%`, `99%` 백분위수를 계산하고,
거기에 약간의 margin을 더해 y축 범위를 잡는다.

이 방식의 목적은:

- 큰 이상치 하나에 스케일이 망가지지 않게 하고
- 파형 형태가 눈에 잘 들어오게 하기 위함이다.

### 9.5 numeric 패널

numeric track는 하나의 하단 축에 모두 겹쳐 그린다.

특징:

- 점 + 선 플롯
- 우상단 범례
- numeric 공통 y축

---

## 10. 내비게이션

### 버튼

- `<< 60s`
- `< 10s`
- `10s >`
- `60s >>`

### 슬라이더

슬라이더는 현재 `_t0`를 초 단위로 조절한다.

슬라이더 최대값은 대략:

```text
max(duration - WINDOW_SEC, 1)
```

이다.

### 키보드 단축키

- `←` / `→`: 10초 이동
- `Ctrl+←` / `Ctrl+→`: 60초 이동

### 시간 표시

하단에는 다음 형식으로 현재 창을 표시한다.

```text
HH:MM:SS ~ HH:MM:SS / 총길이
```

---

## 11. Track Info 창

`Track Info` 버튼은 별도 `Toplevel` 창을 띄워
현재 케이스의 전체 트랙 목록을 보여준다.

표시 컬럼:

- Track Name
- Type
- Unit
- Records

여기서 `Type`은:

- `srate > 0`이면 `XXX Hz`
- 아니면 `numeric`

으로 표시한다.

하단에는 전체 트랙 수와 waveform/numeric 개수 요약도 나온다.

이 창은 "왜 어떤 트랙은 안 보이는가"를 확인할 때 유용하다.

---

## 12. 함수별 설명

### `list_vital_files()`

로컬 `.vital` 파일을 수집한다.

### `fetch_clinical_map()`

목록용 임상 메타데이터를 API에서 가져온다.

### `fetch_track_flags()`

PPG/ABP 존재 여부를 API 인덱스로부터 가져온다.

### `load_vital()`

현재 케이스에서 표시 가능한 트랙들을 실제 배열로 읽는다.

### `duration_sec()`

표시 길이를 계산한다.

### `VitalDBBrowser._refresh_list()`

검색, 정렬, 목록 렌더링, 상태 라벨 갱신을 담당한다.

### `VitalDBBrowser._load_case()`

케이스 로드와 상태 초기화를 담당한다.

### `VitalDBBrowser._rebuild_figure()`

현재 케이스의 트랙 구성에 맞춰 figure layout을 다시 만든다.

### `VitalDBBrowser._draw()`

현재 시간창의 waveform / numeric 데이터를 실제로 그린다.

### `VitalDBBrowser._show_track_info()`

현재 파일의 전체 트랙 인벤토리를 별도 창으로 보여준다.

---

## 13. 상태 표시와 사용자 피드백

상태 바는 다음 정보를 보여준다.

- 로딩 중 메시지
- 로딩 실패 에러
- 현재 케이스와 트랙 목록
- 전체 길이
- 방향키 도움말

입력 디렉터리에 `.vital` 파일이 하나도 없으면,
Tk messagebox로 에러를 띄우고 종료한다.

`--case`로 지정한 파일이 없으면 표준 에러에 메시지를 출력하고 종료한다.

---

## 14. 개발 시 알아둘 제약과 주의점

### 14.1 API 의존성이 있다

clinical metadata와 track flag는 네트워크 기반 API 호출에 의존한다.
호출 실패 시 브라우저 자체는 동작하지만, 목록 정보와 강조 표시가 줄어든다.

### 14.2 표시 대상 트랙은 고정 정의다

현재 브라우저는 `TRACK_DEFS`에 들어 있는 트랙만 표시한다.
다른 유용한 waveform이 있어도 자동으로 추가되지는 않는다.

### 14.3 waveform sample rate를 500 Hz로 가정한다

코드는 `SRATE = 500`을 전제로 waveform interval과 slicing을 계산한다.
다른 샘플링 주파수 트랙을 일반화해 처리하는 구조는 아니다.

### 14.4 numeric track는 하나의 축에 모두 겹친다

값 범위가 다른 numeric 지표가 많아지면 가독성이 떨어질 수 있다.
현재는 단순성과 overview를 우선한 설계다.

### 14.5 케이스 로드 시 figure를 재생성한다

구현이 단순한 대신, 매우 많은 트랙을 가진 케이스에서 전환 비용이 약간 있을 수 있다.

---

## 15. 검증 포인트

문서화 기준으로 점검할 때는 아래를 우선 보면 된다.

- `uv run python scripts/vitaldb-browser.py --help`가 문서와 일치하는가
- `.vital` 파일이 있으면 목록이 뜨는가
- 검색과 컬럼 정렬이 정상 동작하는가
- 케이스 선택 시 파형과 numeric 패널이 표시되는가
- `← → Ctrl+← Ctrl+→` 이동이 정상 동작하는가
- `Track Info` 창이 현재 케이스의 트랙 목록을 보여주는가
- `--case <id>`로 시작 시 해당 케이스가 자동으로 열리는가

---

## 16. 요약

`vitaldb-browser.py`는 원본 VitalDB 수술 신호를 탐색하는 기초 브라우저다.

현재 구현의 핵심 특징은 다음과 같다.

- 케이스 목록, 임상 메타데이터, 트랙 존재 여부를 한 창에서 보여준다.
- PPG, ABP, ECG II, numeric BP/HR를 시간창 기반으로 탐색할 수 있다.
- 검색, 정렬, 슬라이더, 단축키로 긴 수술 데이터를 빠르게 훑을 수 있다.
- `Track Info` 창으로 현재 파일의 전체 트랙 구성을 확인할 수 있다.

전처리나 모델링 이전에 "원본 데이터가 어떤 모습인지"를 이해하고,
학습 가능한 케이스를 골라보는 출발점 역할을 하는 도구다.
