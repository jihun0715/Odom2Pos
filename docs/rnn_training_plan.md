# RNN Training Plan

## Goal

`data/Odom_temporal_aligned.jsonl`을 학습 input으로, `data/pose_GT_by_mocap_temporal_aligned.jsonl`을 GT target으로 사용해 odom velocity를 mocap GT velocity로 매핑하는 RNN 모델을 학습한다.

입력과 출력 JSONL schema는 동일하다.

```json
{"timestamp":0.0,"vx":0.7441520842396535,"vy":0.20772345741912607,"vtheta":-0.23887816162272735}
```

모델은 CPU 환경에서 PyTorch로 학습한다.

## Dependency

`requirements.txt`에 다음 dependency를 추가한다.

```text
torch
```

CPU-only 환경이므로 코드에서는 device 기본값을 `cpu`로 둔다. 사용자가 추후 GPU 환경을 붙이더라도 `config.py`에서 device를 바꿀 수 있게 한다.

## Planned Files

예정 구현 파일:

- `src/config.py`
- `src/train_rnn.py`
- `src/model.py`
- `src/losses.py`
- `scripts/run_train_rnn.sh`

예정 산출물:

- `results/models/rnn_odom_to_gt.pt`
- `results/models/rnn_odom_to_gt_config.json`
- `results/training/train_history.json`
- `results/training/loss_curve.png`
- `results/training/prediction_preview.png`

## Config

`src/config.py`에서 주요 학습 파라미터를 관리한다. 시작 기본값은 다음처럼 둔다.

```python
INPUT_PATH = "data/Odom_temporal_aligned.jsonl"
TARGET_PATH = "data/pose_GT_by_mocap_temporal_aligned.jsonl"

FEATURE_KEYS = ("vx", "vy", "vtheta")
TARGET_KEYS = ("vx", "vy", "vtheta")

DEVICE = "cpu"
SEED = 42

HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.0

SEQ_LEN = 128
STRIDE = 16
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

LOSS_TYPE = "berhu"
BERHU_C = 0.2
THETA_LOSS_WEIGHT = 1.0
```

학습 데이터가 한 trajectory이므로 random split 대신 시간 순서를 유지한 train/validation split을 사용한다.

## Data Loading

1. `Odom_temporal_aligned.jsonl`과 `pose_GT_by_mocap_temporal_aligned.jsonl`을 읽는다.
2. 두 파일의 row 수와 timestamp column이 같은지 assert한다.
3. feature matrix `X = [vx, vy, vtheta]`, target matrix `Y = [vx, vy, vtheta]`를 만든다.
4. train 구간 통계로 input과 target을 각각 normalize한다.
5. 연속 sequence window를 만든다.

Sequence 생성:

```text
X[i : i + SEQ_LEN] -> Y[i : i + SEQ_LEN]
```

`STRIDE` 간격으로 window를 만든다. validation window는 train/validation boundary를 넘지 않도록 한다.

## Model

시작 모델은 기본 PyTorch `nn.RNN`을 사용한다.

```text
input_dim = 3
hidden_size = 32
num_layers = 2
output_dim = 3
```

구조:

```text
RNN(input_dim, hidden_size, num_layers, batch_first=True)
Linear(hidden_size, output_dim)
```

후보 확장:

- vanilla RNN이 불안정하면 GRU로 교체할 수 있게 `RNN_TYPE = "rnn" | "gru"` option을 둔다.
- 초기 구현에서는 요구사항대로 RNN을 기본값으로 둔다.

## Loss

Outlier에 강건한 BerHu loss를 기본으로 사용한다.

BerHu:

```text
if |e| <= c:
    loss = |e|
else:
    loss = (e^2 + c^2) / (2c)
```

여기서 `c`는 고정값 `BERHU_C` 또는 batch error의 `0.2 * max(|e|)` 중 하나를 선택할 수 있다. 초기 구현은 config가 단순한 고정 `BERHU_C`를 사용한다.

### Theta Handling

현재 target은 pose angle `theta`가 아니라 yaw rate `vtheta`이다. 따라서 엄밀히는 angle wrap이 필요하지 않다. 하지만 이후 schema가 `theta`로 되돌아가거나 누적 yaw를 직접 예측하는 실험이 가능하므로 loss 함수에는 angle residual wrap helper를 넣어둔다.

```python
wrapped_error = atan2(sin(pred_theta - target_theta), cos(pred_theta - target_theta))
```

초기 velocity schema에서는 `vtheta` residual에는 wrap을 적용하지 않는다. 대신 config에 다음 option을 둔다.

```python
WRAP_THETA_RESIDUAL = False
THETA_INDEX = 2
```

만약 target이 angle인 실험을 하게 되면 `WRAP_THETA_RESIDUAL = True`로 바꾼다.

## Optimization

Optimizer:

```text
Adam(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
```

Gradient clipping:

```text
clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
```

Validation loss가 가장 낮은 checkpoint를 저장한다.

## Metrics

매 epoch 저장할 값:

- train loss
- validation loss
- validation MAE for `vx`
- validation MAE for `vy`
- validation MAE for `vtheta`
- validation RMSE for each output channel

저장 위치:

- `results/training/train_history.json`
- `results/training/loss_curve.png`

## Inference Preview

학습 후 validation 구간 일부에 대해 prediction preview plot을 저장한다.

Plot:

- `vx`: prediction vs GT
- `vy`: prediction vs GT
- `vtheta`: prediction vs GT

저장 위치:

- `results/training/prediction_preview.png`

## Validation Criteria

구현 후 다음을 확인한다.

- `torch`가 import된다.
- `scripts/run_train_rnn.sh`가 CPU에서 실행된다.
- train/validation loss가 `NaN` 없이 기록된다.
- best checkpoint가 `results/models/rnn_odom_to_gt.pt`에 저장된다.
- prediction preview plot이 생성된다.
- config 값 변경만으로 `hidden_size`, `num_layers`, `seq_len`, `learning_rate`, `loss_type`을 바꿀 수 있다.

## Risks

- 단일 trajectory만 있으므로 validation split은 분포가 train과 다를 수 있다.
- velocity만 입력하면 누적 pose drift를 직접 제어하지 못한다.
- vanilla RNN은 긴 sequence에서 gradient 문제가 생길 수 있다. 기본은 요구사항대로 RNN으로 두되, config로 GRU 전환 여지를 둔다.
- `vtheta`는 angle이 아니라 angular velocity이므로 wrap을 무조건 적용하면 오히려 잘못된 loss가 된다.
