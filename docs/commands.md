# 배치/쉘 명령어 사용 예시

## 데이터셋 구축

```bash
bin/construct-dataset
```

## 모델 학습

```bash
bin/train-model --model acfa
bin/train-model --model ae_lstm
bin/train-model --model cnn_bilstm_at
bin/train-model --model conv_reg
bin/train-model --model conv_reg_at
bin/train-model --model conv_reg_ds
bin/train-model --model conv_reg_nas
bin/train-model --model minception
bin/train-model --model mtae
bin/train-model --model mtae_tr
bin/train-model --model naive
bin/train-model --model pulse_resnet1d
bin/train-model --model pulsew_resnet1d
bin/train-model --model pulsewo_resnet1d
bin/train-model --model pulsewoq_resnet1d
bin/train-model --model resnet1d
bin/train-model --model resnet1d_micro
bin/train-model --model resnet1d_mini
bin/train-model --model resnet1d_tiny
bin/train-model --model st_resnet
bin/train-model --model xresnet1d
```

## 모델 구조 출력

```bash
bin/print-model --model acfa              > data/models/acfa/struct.txt
bin/print-model --model ae_lstm           > data/models/ae_lstm/struct.txt
bin/print-model --model cnn_bilstm_at     > data/models/cnn_bilstm_at/struct.txt
bin/print-model --model conv_reg          > data/models/conv_reg/struct.txt
bin/print-model --model conv_reg_at       > data/models/conv_reg_at/struct.txt
bin/print-model --model conv_reg_ds       > data/models/conv_reg_ds/struct.txt
bin/print-model --model conv_reg_nas       > data/models/conv_reg_nas/struct.txt
bin/print-model --model minception        > data/models/minception/struct.txt
bin/print-model --model mtae              > data/models/mtae/struct.txt
bin/print-model --model mtae_tr           > data/models/mtae_tr/struct.txt
bin/print-model --model naive             > data/models/naive/struct.txt
bin/print-model --model pulse_resnet1d    > data/models/pulse_resnet1d/struct.txt
bin/print-model --model pulsew_resnet1d   > data/models/pulsew_resnet1d/struct.txt
bin/print-model --model pulsewo_resnet1d  > data/models/pulsewo_resnet1d/struct.txt
bin/print-model --model pulsewoq_resnet1d > data/models/pulsewoq_resnet1d/struct.txt
bin/print-model --model resnet1d          > data/models/resnet1d/struct.txt
bin/print-model --model resnet1d_micro    > data/models/resnet1d_micro/struct.txt
bin/print-model --model resnet1d_mini     > data/models/resnet1d_mini/struct.txt
bin/print-model --model resnet1d_tiny     > data/models/resnet1d_tiny/struct.txt
bin/print-model --model st_resnet         > data/models/st_resnet/struct.txt
bin/print-model --model xresnet1d         > data/models/xresnet1d/struct.txt
```

## 모델 Training Status

```bash
bin/train-status  data/models/acfa
bin/train-status  data/models/ae_lstm
bin/train-status  data/models/cnn_bilstm_at
bin/train-status  data/models/conv_reg
bin/train-status  data/models/conv_reg_at
bin/train-status  data/models/conv_reg_ds
bin/train-status  data/models/conv_reg_nas
bin/train-status  data/models/minception
bin/train-status  data/models/mtae
bin/train-status  data/models/mtae_tr
bin/train-status  data/models/naive
bin/train-status  data/models/pulse_resnet1d
bin/train-status  data/models/pulsew_resnet1d
bin/train-status  data/models/pulsewo_resnet1d
bin/train-status  data/models/pulsewoq_resnet1d
bin/train-status  data/models/resnet1d
bin/train-status  data/models/resnet1d_micro
bin/train-status  data/models/resnet1d_mini
bin/train-status  data/models/resnet1d_tiny
bin/train-status  data/models/st_resnet
bin/train-status  data/models/xresnet1d
```

## 모델 평가

```bash
bin/eval-model data/models/acfa
bin/eval-model data/models/ae_lstm
bin/eval-model data/models/cnn_bilstm_at
bin/eval-model data/models/conv_reg
bin/eval-model data/models/conv_reg_at
bin/eval-model data/models/conv_reg_ds
bin/eval-model data/models/conv_reg_nas
bin/eval-model data/models/minception
bin/eval-model data/models/mtae
bin/eval-model data/models/mtae_tr
bin/eval-model data/models/naive
bin/eval-model data/models/pulse_resnet1d
bin/eval-model data/models/pulsew_resnet1d
bin/eval-model data/models/pulsewo_resnet1d
bin/eval-model data/models/resnet1d
bin/eval-model data/models/resnet1d_micro
bin/eval-model data/models/resnet1d_mini
bin/eval-model data/models/resnet1d_tiny
bin/eval-model data/models/st_resnet
bin/eval-model data/models/xresnet1d
bin/eval-model-pulsewoq data/models/pulsewoq_resnet1d
```

## 평가 결과 수집

```bash
bin/collect-result
bin/generate-overview
```
