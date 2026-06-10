# 배치/쉘 명령어 사용 예시

## 데이터셋 구축

```bash
bin/construct-dataset
```

## 모델 학습

```bash
bin/train-model --device cuda:0 --model acfa              ; \
bin/train-model --device cuda:0 --model ae_lstm           ; \
bin/train-model --device cuda:0 --model cnn_bilstm_at

bin/train-model --device cuda:1 --model minception        ; \
bin/train-model --device cuda:1 --model mtae_tr           ; \
bin/train-model --device cuda:1 --model mtae              ; \
bin/train-model --device cuda:1 --model naive             ; \
bin/train-model --device cuda:1 --model pulse_resnet1d    ; \
bin/train-model --device cuda:1 --model pulsew_resnet1d

bin/train-model --device cuda:2 --model pulsewo_resnet1d  ; \
bin/train-model --device cuda:2 --model pulsewoq_resnet1d ; \
bin/train-model --device cuda:2 --model resnet1d_micro    ; \
bin/train-model --device cuda:2 --model resnet1d_mini

bin/train-model --device cuda:3 --model resnet1d_tiny     ; \
bin/train-model --device cuda:3 --model resnet1d          ; \
bin/train-model --device cuda:3 --model st_resnet         ; \
bin/train-model --device cuda:3 --model xresnet1d
```

## 모델 구조 출력

```bash
bin/print-model --model acfa              > data/models/acfa.txt
bin/print-model --model ae_lstm           > data/models/ae_lstm.txt
bin/print-model --model cnn_bilstm_at     > data/models/cnn_bilstm_at.txt
bin/print-model --model minception        > data/models/minception.txt
bin/print-model --model mtae              > data/models/mtae.txt
bin/print-model --model mtae_tr           > data/models/mtae_tr.txt
bin/print-model --model naive             > data/models/naive.txt
bin/print-model --model pulse_resnet1d    > data/models/pulse_resnet1d.txt
bin/print-model --model pulsew_resnet1d   > data/models/pulsew_resnet1d.txt
bin/print-model --model pulsewo_resnet1d  > data/models/pulsewo_resnet1d.txt
bin/print-model --model pulsewoq_resnet1d > data/models/pulsewoq_resnet1d.txt
bin/print-model --model resnet1d          > data/models/resnet1d.txt
bin/print-model --model resnet1d_micro    > data/models/resnet1d_micro.txt
bin/print-model --model resnet1d_mini     > data/models/resnet1d_mini.txt
bin/print-model --model resnet1d_tiny     > data/models/resnet1d_tiny.txt
bin/print-model --model st_resnet         > data/models/st_resnet.txt
bin/print-model --model xresnet1d         > data/models/xresnet1d.txt
```

## 모델 Training Status

```bash
bin/train-status  data/models/acfa
bin/train-status  data/models/ae_lstm
bin/train-status  data/models/cnn_bilstm_at
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
```
