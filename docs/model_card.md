# Model card

**Intended use**: reading short words (digits and Latin letters) in photographs, e.g. signs and
labels. Educational / portfolio project.

**Training data**: ICDAR 2003 Robust Reading, *SceneTrialTrain* (250 photos; 1,097 words pass the
`[0-9a-z]+` filter; 1,088 crops survive the size/brightness filter). Split by scene 70/20/10.
The detector used for this project is Ultralytics YOLO26 (`yolo26m.pt`), fine-tuned to detect one
class, `text`. Recognition uses a CRNN with a ResNet34 backbone and BiLSTM.

**Not suitable for**: documents, handwriting, non-Latin scripts, punctuation or mixed case
(labels are lower-cased and restricted to 36 characters), safety-critical use.

**Known limitations**
* Tiny training set: expect weak generalisation outside ICDAR-style photos; the detector sees about
  170 training images.
* Words containing punctuation/accents were removed from training, so the model cannot output them.
* Upper/lower case is not distinguished.
* Confidence scores are heuristic (mean max-probability of non-blank frames), not calibrated.

**Reported results**: one completed training run on the held-out ICDAR 2003 test split. These
figures describe this run only; they are not a performance guarantee and may vary with data,
software versions, and training settings.

| Metric | Split | Value |
|---|---|---|
| YOLO26 detector precision | test | 0.886 |
| YOLO26 detector recall | test | 0.840 |
| YOLO26 detector mAP50 | test | 0.860 |
| YOLO26 detector mAP50-95 | test | 0.670 |
| CRNN recogniser word accuracy | test | 0.523 |
| CRNN recogniser character error rate (CER) | test | 0.127 |
| End-to-end F1 @ IoU 0.5 | test | 0.433 |
