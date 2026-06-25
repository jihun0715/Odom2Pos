# Temporal Offset Estimation Plan Using NCC

## Goal

`data/Odom.jsonl`과 `data/pose_GT_by_mocap.jsonl` 사이의 temporal offset을 normalized cross correlation, NCC로 추정한다. 추정된 offset을 이용해 두 trajectory를 같은 시간축에 정렬한 뒤, 이후 odom-to-GT mapping 모델 학습에 사용할 수 있는 보정 JSONL을 `data` 디렉토리에 저장한다.

## Input

- `data/Odom.jsonl`
- `data/pose_GT_by_mocap.jsonl`

각 파일은 다음 네 key를 가진 JSONL이다.

```json
{"timestamp":1769620191.2758064,"odom_x":0.0,"odom_y":0.0,"odom_theta":0.0}
```

`odom_theta`는 radian이고, 각 파일의 첫 pose가 `(0, 0, 0)`이 되도록 이미 origin-normalized 되어 있다.

## Output

구현 시 기본 출력은 다음처럼 둔다.

- `data/Odom_temporal_aligned.jsonl`
- `data/pose_GT_by_mocap_temporal_aligned.jsonl`

두 출력 파일은 같은 timestamp grid를 공유한다. 각 줄의 key는 기존과 동일하게 유지한다.

```json
{"timestamp":0.0,"odom_x":...,"odom_y":...,"odom_theta":...}
```

출력 timestamp는 temporal alignment 이후의 공통 상대 시간으로 둔다. 즉 첫 공통 sample이 `0.0`이 되도록 저장한다. 원본 epoch timestamp가 필요할 수 있으므로, 추정 결과 summary에는 원본 common window도 같이 기록한다.

추정 결과와 품질 지표는 다음 파일에 저장한다.

- `results/summaries/temporal_alignment_summary.json`
- `results/summaries/temporal_alignment_summary.txt`
- 선택적으로 `results/diagnostics/temporal_ncc.png`

## Offset Convention

추정값을 `gt_time_offset_sec`로 정의한다.

```text
gt_motion(t + gt_time_offset_sec) ~= odom_motion(t)
```

해석:

- `gt_time_offset_sec > 0`: GT signal을 더 미래 timestamp에서 읽어야 odom과 맞는다. 즉 GT가 odom보다 늦게 기록된 것처럼 보인다.
- `gt_time_offset_sec < 0`: GT signal을 더 과거 timestamp에서 읽어야 odom과 맞는다. 즉 GT가 odom보다 빠르게 기록된 것처럼 보인다.

JSONL 출력에서는 이 offset을 적용한 뒤 두 signal의 overlap 구간만 사용한다.

## Why Use Motion Features

절대 `x`, `y`, `theta` trajectory에 직접 NCC를 적용하면 다음 문제가 있다.

- 두 파일의 origin normalization 시점이 서로 다르다.
- spatial offset 또는 residual rotation이 있으면 absolute pose correlation이 왜곡될 수 있다.
- trajectory가 천천히 변하거나 구간별로 정지하면 absolute pose NCC peak가 둔해질 수 있다.

따라서 NCC는 pose 자체가 아니라 motion feature에서 수행한다.

기본 feature:

- linear speed: `speed = sqrt((dx/dt)^2 + (dy/dt)^2)`
- yaw rate: `omega = d(odom_theta)/dt`

이 두 값은 2D frame의 translation/rotation origin에 덜 민감하고, 시간 지연을 찾는 데 더 직접적인 신호다.

## Processing Steps

1. Load JSONL

   - `timestamp`, `odom_x`, `odom_y`, `odom_theta`를 float array로 읽는다.
   - timestamp 순서로 정렬한다.
   - 같은 timestamp가 중복된 경우 마지막 row를 사용한다. Odom 원본에는 중복 timestamp가 많으므로 NCC 전에 반드시 처리한다.

2. Build Uniform Time Grid

   - 두 파일의 median sample interval을 계산한다.
   - 기본 resampling interval은 `max(median_dt_odom, median_dt_gt)`로 둔다.
   - 현재 데이터 기준으로는 Odom이 약 25 ms, GT가 약 10 ms이므로 기본 grid는 약 25 ms가 된다.
   - CLI option으로 `--dt`, `--max-lag-sec`를 override할 수 있게 한다.

3. Interpolate Pose

   - uniform grid에 `odom_x`, `odom_y`, `odom_theta`를 선형 보간한다.
   - `odom_theta`는 이미 unwrap된 값이라고 보고 그대로 보간한다.
   - 보간은 각 파일의 원본 timestamp 범위 내부에서만 수행한다.

4. Compute Motion Features

   - central difference 또는 `np.gradient`로 `vx`, `vy`, `omega`를 계산한다.
   - `speed = sqrt(vx^2 + vy^2)`를 만든다.
   - 정지 구간의 노이즈를 줄이기 위해 필요하면 작은 moving average smoothing을 적용한다. 초기 구현에서는 dependency를 늘리지 않기 위해 `numpy.convolve` 기반 box filter만 사용한다.

5. Normalize Features

   - 각 feature channel에서 평균을 빼고 표준편차로 나눈다.
   - 표준편차가 너무 작은 channel은 NCC에서 제외한다.
   - 기본 channel은 `speed`와 `omega`이며, channel별 NCC score를 평균한다.

6. Search Lag

   - `lag` 범위는 기본 `[-5.0 sec, +5.0 sec]`로 둔다.
   - lag step은 uniform grid interval `dt`와 같다.
   - 각 lag에서 두 feature signal의 overlap sample만 사용한다.
   - overlap sample 수가 너무 적으면 해당 lag는 버린다.
   - score는 overlap 구간의 normalized dot product로 계산한다.

7. Sub-sample Peak Refinement

   - 최고 NCC score를 가진 lag index를 찾는다.
   - peak 전후 3점을 이용해 parabolic interpolation을 적용한다.
   - refined offset을 초 단위 `gt_time_offset_sec`로 저장한다.

8. Generate Temporally Aligned JSONL

   - 추정된 offset을 GT timestamp에 적용한 뒤 공통 overlap window를 계산한다.
   - 두 trajectory를 같은 relative timestamp grid로 resampling한다.
   - 출력 timestamp는 `0.0`부터 시작하는 aligned relative time으로 둔다.
   - 출력 row key는 `timestamp`, `odom_x`, `odom_y`, `odom_theta`만 유지한다.

9. Save Summary and Diagnostics

   Summary에 저장할 값:

   - estimated `gt_time_offset_sec`
   - NCC peak score
   - search range and dt
   - number of overlap samples
   - common window before and after alignment
   - channels used for NCC

   Diagnostic plot:

   - NCC score versus lag
   - offset 적용 전후 `speed`, `omega` overlay

## Implementation Files

예정 파일:

- `src/estimate_temporal_offset_ncc.py`
- `scripts/run_temporal_alignment.sh`

기존 dependency인 `numpy`와 `matplotlib`만 사용한다. 추가 dependency는 필요하지 않을 것으로 본다.

## Validation Criteria

구현 후 다음을 확인한다.

- `data/Odom_temporal_aligned.jsonl`과 `data/pose_GT_by_mocap_temporal_aligned.jsonl`의 row 수가 같다.
- 두 출력 파일의 `timestamp` column이 동일하다.
- 첫 timestamp가 `0.0`이다.
- NCC peak가 search boundary에 붙어 있지 않다. boundary에 붙으면 `--max-lag-sec`를 늘려 재실행한다.
- offset 적용 후 speed/yaw-rate overlay가 적용 전보다 더 잘 맞는다.

## Risks

- 주행 중 정지 구간이 많으면 speed 기반 NCC가 약해질 수 있다.
- yaw rate가 거의 없는 구간이 길면 omega channel도 정보량이 부족하다.
- Odom 중복 timestamp가 많기 때문에 중복 제거 정책이 결과에 영향을 줄 수 있다.
- NCC는 global constant time offset만 추정한다. sensor clock drift처럼 시간이 지날수록 offset이 변하는 문제는 별도 모델이 필요하다.
