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

- `results/trajectory_xy.png`: odom과 mocap GT의 XY trajectory 비교
- `results/position_vs_time.png`: x, y, z position time-series 비교
- `results/yaw_vs_time.png`: quaternion에서 계산한 yaw time-series 비교
- `results/sampling_intervals.png`: 각 파일의 sampling interval 비교
- `results/aligned_time_comparison.png`: 공통 시간 구간에서 GT timestamp 기준으로 odom을 보간한 비교
- `results/tum_summary.json`, `results/tum_summary.txt`: 행 수, 시간 범위, 중복 timestamp, sampling 통계
