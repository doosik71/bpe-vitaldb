# 배치/쉘 명령어 사용 예시

## VitalDB 다운로드

```bash
bin/download-vitaldb --help
```

## 데이터셋 구축

```bash
bin/construct-dataset
uv run python scripts/construct-dataset-v1.py
uv run python scripts/construct-dataset-v2.py
```

## 데이터셋 분석

```bash
bin/dataset-statistic
bin/dataset-statistic --dataset-dir data/dataset-v1
bin/dataset-statistic --dataset-dir data/dataset-v2
```

## 모델 학습

```bash
bin/train-model --model acfa
bin/train-model --model ae_lstm
bin/train-model --model bpnet_cf
bin/train-model --model cnn_bilstm_at
bin/train-model --model conv_reg
bin/train-model --model conv_reg_at
bin/train-model --model conv_reg_ds
bin/train-model --model minception
bin/train-model --model mtae
bin/train-model --model mtae_tr
bin/train-model --model naive
bin/train-model --model pctn
bin/train-model --model resnet1d
bin/train-model --model resnet1d_micro
bin/train-model --model resnet1d_mini
bin/train-model --model resnet1d_tiny
bin/train-model --model st_resnet
bin/train-model --model xresnet1d

bin/train-all-model
bin/train-all-model --dataset-dir data/dataset-v1 --models-dir data/models-v1
bin/train-all-model --dataset-dir data/dataset-v2 --models-dir data/models-v2
```

## 모델 구조 출력

```bash
bin/print-model --model acfa              > data/models/acfa/struct.txt
bin/print-model --model ae_lstm           > data/models/ae_lstm/struct.txt
bin/print-model --model bpnet_cf           > data/models/bpnet_cf/struct.txt
bin/print-model --model cnn_bilstm_at     > data/models/cnn_bilstm_at/struct.txt
bin/print-model --model conv_reg          > data/models/conv_reg/struct.txt
bin/print-model --model conv_reg_at       > data/models/conv_reg_at/struct.txt
bin/print-model --model conv_reg_ds       > data/models/conv_reg_ds/struct.txt
bin/print-model --model minception        > data/models/minception/struct.txt
bin/print-model --model mtae              > data/models/mtae/struct.txt
bin/print-model --model mtae_tr           > data/models/mtae_tr/struct.txt
bin/print-model --model naive             > data/models/naive/struct.txt
bin/print-model --model pctn              > data/models/pctn/struct.txt
bin/print-model --model resnet1d          > data/models/resnet1d/struct.txt
bin/print-model --model resnet1d_micro    > data/models/resnet1d_micro/struct.txt
bin/print-model --model resnet1d_mini     > data/models/resnet1d_mini/struct.txt
bin/print-model --model resnet1d_tiny     > data/models/resnet1d_tiny/struct.txt
bin/print-model --model st_resnet         > data/models/st_resnet/struct.txt
bin/print-model --model xresnet1d         > data/models/xresnet1d/struct.txt

bin/print-model --model acfa              > data/models-v1/acfa/struct.txt
bin/print-model --model ae_lstm           > data/models-v1/ae_lstm/struct.txt
bin/print-model --model bpnet_cf          > data/models-v1/bpnet_cf/struct.txt
bin/print-model --model cnn_bilstm_at     > data/models-v1/cnn_bilstm_at/struct.txt
bin/print-model --model conv_reg          > data/models-v1/conv_reg/struct.txt
bin/print-model --model conv_reg_at       > data/models-v1/conv_reg_at/struct.txt
bin/print-model --model conv_reg_ds       > data/models-v1/conv_reg_ds/struct.txt
bin/print-model --model minception        > data/models-v1/minception/struct.txt
bin/print-model --model mtae              > data/models-v1/mtae/struct.txt
bin/print-model --model mtae_tr           > data/models-v1/mtae_tr/struct.txt
bin/print-model --model naive             > data/models-v1/naive/struct.txt
bin/print-model --model pctn              > data/models-v1/pctn/struct.txt
bin/print-model --model resnet1d          > data/models-v1/resnet1d/struct.txt
bin/print-model --model resnet1d_micro    > data/models-v1/resnet1d_micro/struct.txt
bin/print-model --model resnet1d_mini     > data/models-v1/resnet1d_mini/struct.txt
bin/print-model --model resnet1d_tiny     > data/models-v1/resnet1d_tiny/struct.txt
bin/print-model --model st_resnet         > data/models-v1/st_resnet/struct.txt
bin/print-model --model xresnet1d         > data/models-v1/xresnet1d/struct.txt

bin/print-model --model acfa              > data/models-v2/acfa/struct.txt
bin/print-model --model ae_lstm           > data/models-v2/ae_lstm/struct.txt
bin/print-model --model bpnet_cf          > data/models-v2/bpnet_cf/struct.txt
bin/print-model --model cnn_bilstm_at     > data/models-v2/cnn_bilstm_at/struct.txt
bin/print-model --model conv_reg          > data/models-v2/conv_reg/struct.txt
bin/print-model --model conv_reg_at       > data/models-v2/conv_reg_at/struct.txt
bin/print-model --model conv_reg_ds       > data/models-v2/conv_reg_ds/struct.txt
bin/print-model --model minception        > data/models-v2/minception/struct.txt
bin/print-model --model mtae              > data/models-v2/mtae/struct.txt
bin/print-model --model mtae_tr           > data/models-v2/mtae_tr/struct.txt
bin/print-model --model naive             > data/models-v2/naive/struct.txt
bin/print-model --model pctn              > data/models-v2/pctn/struct.txt
bin/print-model --model resnet1d          > data/models-v2/resnet1d/struct.txt
bin/print-model --model resnet1d_micro    > data/models-v2/resnet1d_micro/struct.txt
bin/print-model --model resnet1d_mini     > data/models-v2/resnet1d_mini/struct.txt
bin/print-model --model resnet1d_tiny     > data/models-v2/resnet1d_tiny/struct.txt
bin/print-model --model st_resnet         > data/models-v2/st_resnet/struct.txt
bin/print-model --model xresnet1d         > data/models-v2/xresnet1d/struct.txt

bin/print-all-model
bin/print-all-model --models-dir data/models-v1
bin/print-all-model --models-dir data/models-v2
```

## 모델 Training Status

```bash
bin/generate-train-status  data/models/acfa
bin/generate-train-status  data/models/ae_lstm
bin/generate-train-status  data/models/bpnet_cf
bin/generate-train-status  data/models/cnn_bilstm_at
bin/generate-train-status  data/models/conv_reg
bin/generate-train-status  data/models/conv_reg_at
bin/generate-train-status  data/models/conv_reg_ds
bin/generate-train-status  data/models/minception
bin/generate-train-status  data/models/mtae
bin/generate-train-status  data/models/mtae_tr
bin/generate-train-status  data/models/naive
bin/generate-train-status  data/models/pctn
bin/generate-train-status  data/models/resnet1d
bin/generate-train-status  data/models/resnet1d_micro
bin/generate-train-status  data/models/resnet1d_mini
bin/generate-train-status  data/models/resnet1d_tiny
bin/generate-train-status  data/models/st_resnet
bin/generate-train-status  data/models/xresnet1d

bin/generate-train-status  data/models-v1/acfa
bin/generate-train-status  data/models-v1/ae_lstm
bin/generate-train-status  data/models-v1/bpnet_cf
bin/generate-train-status  data/models-v1/cnn_bilstm_at
bin/generate-train-status  data/models-v1/conv_reg
bin/generate-train-status  data/models-v1/conv_reg_at
bin/generate-train-status  data/models-v1/conv_reg_ds
bin/generate-train-status  data/models-v1/minception
bin/generate-train-status  data/models-v1/mtae
bin/generate-train-status  data/models-v1/mtae_tr
bin/generate-train-status  data/models-v1/naive
bin/generate-train-status  data/models-v1/pctn
bin/generate-train-status  data/models-v1/resnet1d
bin/generate-train-status  data/models-v1/resnet1d_micro
bin/generate-train-status  data/models-v1/resnet1d_mini
bin/generate-train-status  data/models-v1/resnet1d_tiny
bin/generate-train-status  data/models-v1/st_resnet
bin/generate-train-status  data/models-v1/xresnet1d

bin/generate-train-status  data/models-v2/acfa
bin/generate-train-status  data/models-v2/ae_lstm
bin/generate-train-status  data/models-v2/bpnet_cf
bin/generate-train-status  data/models-v2/cnn_bilstm_at
bin/generate-train-status  data/models-v2/conv_reg
bin/generate-train-status  data/models-v2/conv_reg_at
bin/generate-train-status  data/models-v2/conv_reg_ds
bin/generate-train-status  data/models-v2/minception
bin/generate-train-status  data/models-v2/mtae
bin/generate-train-status  data/models-v2/mtae_tr
bin/generate-train-status  data/models-v2/naive
bin/generate-train-status  data/models-v2/pctn
bin/generate-train-status  data/models-v2/resnet1d
bin/generate-train-status  data/models-v2/resnet1d_micro
bin/generate-train-status  data/models-v2/resnet1d_mini
bin/generate-train-status  data/models-v2/resnet1d_tiny
bin/generate-train-status  data/models-v2/st_resnet
bin/generate-train-status  data/models-v2/xresnet1d

bin/generate-all-train-status
bin/generate-all-train-status --models-dir data/models-v1
bin/generate-all-train-status --models-dir data/models-v2
```

## 모델 평가

```bash
bin/eval-model data/models/acfa
bin/eval-model data/models/ae_lstm
bin/eval-model data/models/bpnet_cf
bin/eval-model data/models/cnn_bilstm_at
bin/eval-model data/models/conv_reg
bin/eval-model data/models/conv_reg_at
bin/eval-model data/models/conv_reg_ds
bin/eval-model data/models/minception
bin/eval-model data/models/mtae
bin/eval-model data/models/mtae_tr
bin/eval-model data/models/naive
bin/eval-model data/models/pctn
bin/eval-model data/models/resnet1d
bin/eval-model data/models/resnet1d_micro
bin/eval-model data/models/resnet1d_mini
bin/eval-model data/models/resnet1d_tiny
bin/eval-model data/models/st_resnet
bin/eval-model data/models/xresnet1d

bin/eval-all-model --dataset-dir data/dataset-v1 --models-dir data/models-v1
bin/eval-all-model --dataset-dir data/dataset-v2 --models-dir data/models-v2
```

## 평가 결과 수집

```bash
bin/collect-result

bin/collect-result --models-dir data/models-v1 --results-dir data/results-v1
bin/collect-result --models-dir data/models-v2 --results-dir data/results-v2

bin/generate-overview
bin/generate-overview --models-dir data/models-v1 --results-dir data/results-v1
bin/generate-overview --models-dir data/models-v2 --results-dir data/results-v2
```
