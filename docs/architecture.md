# Architecture

```
image ──► YOLO26 (1 class: "text") ──► boxes ──► crop (+5 % padding) ──► CRNN ──► text
                                                              │
                         ResNet (stride (2,1) in stages 3-4) ─┤ (B,C,H',T)  mean over H'
                         Linear 512 + ReLU + Dropout          │
                         BiLSTM/BiGRU x3  ► LayerNorm ► Linear(37) ► log-softmax  (T,B,37)
                         CTC loss (blank = 0), greedy decoding
```

## Design decisions (and where they differ from the course notebook)

| Topic | Course notebook | This repo | Reason |
|---|---|---|---|
| CTC indices | blank = index of `-` (sorted position 1), padding 0, chars up to 37 with a 37-way output | **blank = 0**, chars 1..36, 37-way output | In the notebook the last character (`z`) has index 37, outside the 37 outputs |
| Inference decode | drops blanks only | **collapses repeats, then drops blanks** | The notebook turns `hello` into `helo` |
| Data split | split after cropping; val/test folders swapped in `save_data` | **split by source image**, once | Crops of one photo in train and test leak information |
| Image file names | basename only | `folder__stem` | Avoids name collisions across folders |
| Label filter | `isalnum()` (accepts accents) + odd char checks | regex `[0-9a-z]+` | Matches the stated 0-9 / a-z vocabulary |
| Detector | Pretrained YOLO26 model (`yolo26m.pt`) | Fine-tuned one-class text detector; inference uses the checkpoint configured in `configs/pipeline.yaml` | YOLO26 is the detector used throughout the current project |
| Backbone | text says ResNet101, training code ResNet152, inference code ResNet101 | single `backbone` config key, default ResNet34 | Train and inference cannot disagree; weights are stored with their config |
| Time steps | 4x14 map -> 14 steps | stride (2,1) in stages 3-4 -> **53 steps** | Longest ICDAR word is 14 chars; CTC needs `T >= L + repeats` |
| Precision | `@autocast` on forward, CTC in fp16 | AMP for the network, **CTC in fp32** | CTC is numerically fragile in fp16 |
| Metrics | loss only | + word accuracy, CER, detection P/R/mAP, **end-to-end F1** | Loss does not tell you whether it reads correctly |

## Evaluation protocol

* **Detector**: Ultralytics `val` on the held-out test scenes (precision, recall, mAP50, mAP50-95).
* **Recogniser**: loss, exact-match accuracy and character error rate on crops of held-out scenes.
* **End-to-end**: a predicted region is a true positive when IoU >= 0.5 with a ground-truth word
  *and* the transcription matches (case-insensitive); greedy one-to-one matching.
  Reported as precision / recall / F1, plus detection-only P/R/F1 for comparison.
