# Odom2Pos

`data/Odom.tum` odometry trajectory와 `data/pose_GT_by_mocap.tum` mocap ground-truth pose trajectory를 비교하고 시각화하기 위한 초기 분석 프로젝트입니다. 이후 단계에서는 odom trajectory를 pseudo-GT trajectory로 변환하는 매핑 모델을 학습하거나 추정하는 방향으로 확장할 수 있습니다.

## TUM Trajectory Format

TUM trajectory 파일은 한 줄이 하나의 pose를 나타내는 텍스트 포맷입니다. 일반적인 컬럼 순서는 다음과 같습니다.

```text
timestamp tx ty tz qx qy qz qw
```

- `timestamp`: Unix epoch 기반 시간. 초 단위 부동소수점 값입니다.
- `tx ty tz`: position translation. 보통 미터 단위입니다.
- `qx qy qz qw`: orientation quaternion. TUM 포맷은 quaternion을 `x, y, z, w` 순서로 저장합니다.

이 프로젝트의 두 파일도 동일한 8컬럼 구조입니다. `Odom.tum`은 2D odometry 성격이 강해 `z`, `qx`, `qy`가 대부분 0이고 yaw가 `qz`, `qw`에 담겨 있습니다. `pose_GT_by_mocap.tum`은 mocap에서 얻은 3D pose라 `z`, roll, pitch 성분도 포함되어 있습니다.

## Setup

레포 최상위 디렉토리에서 가상환경을 만들고 dependency를 설치합니다.

```bash
bash scripts/setup_venv.sh
source venv/bin/activate
```

직접 실행하려면 다음 커맨드와 동일합니다.

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Visualization

다음 스크립트가 `data` 아래의 두 TUM 파일을 읽고 결과 그래프와 요약 파일을 `results`에 저장합니다.

```bash
bash scripts/run_visualization.sh
```

생성되는 주요 결과:

- `results/raw/`: 정합 전 원본 odom과 mocap GT 비교 그래프
  - `trajectory_xy.png`: XY trajectory 비교
  - `position_vs_time.png`: x, y, z position time-series 비교
  - `yaw_vs_time.png`: quaternion에서 계산한 yaw time-series 비교
  - `time_aligned_comparison.png`: 공통 시간 구간에서 GT timestamp 기준으로 odom을 보간한 비교
- `results/initial_alignment/`: 초기 pose 기준 2D offset 정합 후 비교 그래프
  - `trajectory_xy.png`: 공통 시간 구간 XY trajectory 비교
  - `position_vs_time.png`: 초기 정합 후 x, y, z position time-series 비교
  - `yaw_vs_time.png`: 초기 정합 후 yaw time-series 비교
  - `time_aligned_comparison.png`: 초기 정합 후 공통 시간 구간 timestamp-aligned 비교
- `results/diagnostics/sampling_intervals.png`: 각 파일의 sampling interval 비교
- `results/summaries/`: TUM 데이터 통계와 초기 정합 summary
  - `tum_summary.json`, `tum_summary.txt`
  - `initial_alignment_summary.json`, `initial_alignment_summary.txt`

초기 정합은 공통 시간 구간의 시작 시점에서 odom pose가 GT pose와 일치하도록 다음 2D transform을 자동 계산합니다. 이 단계에서는 `x`, `y`, `yaw`만 정합하며 `z`는 odom 원본 값을 유지합니다.

```text
x' = cos(theta) * x - sin(theta) * y + offset_x
y' = sin(theta) * x + cos(theta) * y + offset_y
yaw' = yaw + offset_theta
```

## JSONL Export

TUM trajectory를 2D velocity 학습용 JSONL로 변환하려면 다음 스크립트를 실행합니다.

```bash
bash scripts/run_export_jsonl.sh
```

생성되는 파일:

- `data/Odom.jsonl`
- `data/pose_GT_by_mocap.jsonl`

각 JSONL 파일은 한 줄에 하나의 velocity sample을 담고, key는 다음 네 개만 사용합니다.

```json
{"timestamp":1769620191.2758064,"vx":0.0,"vy":0.0,"vtheta":0.0}
```

변환 규칙:

- 원본 TUM의 `timestamp`는 그대로 유지합니다.
- `tx`, `ty`만 사용하고 `tz`는 버립니다.
- quaternion에서는 z축 회전 성분인 yaw만 추출하고 roll, pitch 성분은 버립니다.
- 각 파일의 첫 pose frame에서 `x`, `y`, `theta` trajectory를 계산한 뒤 시간 미분해 `vx`, `vy`, `vtheta`를 저장합니다.
- `vtheta`는 rad/s 단위입니다.
- Odom처럼 중복 timestamp가 있는 경우 unique timestamp trajectory에서 velocity를 계산한 뒤 원래 timestamp row에 같은 시각의 velocity를 다시 매핑합니다.

## Temporal Alignment

두 velocity JSONL trajectory 사이의 timestamp calibration을 NCC로 추정하고 보정하려면 다음 스크립트를 실행합니다.

```bash
bash scripts/run_temporal_alignment.sh
```

NCC는 `speed = sqrt(vx^2 + vy^2)`와 `vtheta` feature를 사용합니다. 기본 lag 후보 범위는 `-5초 ~ +5초`이고, 각 후보 lag마다 짧은 sliding window가 아니라 전체 overlap 구간의 NCC score를 계산합니다.

생성되는 파일:

- `data/Odom_temporal_aligned.jsonl`
- `data/pose_GT_by_mocap_temporal_aligned.jsonl`
- `results/summaries/temporal_alignment_summary.json`
- `results/summaries/temporal_alignment_summary.txt`
- `results/diagnostics/temporal_ncc.png`

추정 offset의 convention은 다음과 같습니다.

```text
gt_motion(t + gt_time_offset_sec) ~= odom_motion(t)
```
