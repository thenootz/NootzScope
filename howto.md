# Usefull CLIs

## Trainig your model

In order to train your model

```bash

python tools/train_model.py \ 
  --lot data/Lot1 \
  --lot data/Lot2 \
  --out models/model.joblib \
  --tune-threshold \
  --verbose

```

## Submission

In order to create the subbmistion zip

```bash

python tools/make_submission_zip.py \
  --student-folder IURASCU_Danut \
  --main src/main.py \
  --model models/model.joblib \
  --include-src-helpers \
  --include-requirements \
  --out submissions/project.zip
  ```

## Local evaluation

```bash

python tools/evaluate_local.py --lot data/Lot2 --model models/model.joblib

[OVERALL] Using threshold=0.48 (scores=proba)
  acc=0.9999 prec=0.9998 rec=1.0000 f1=0.9999
Confusion matrix (labels: 0=clean, 1=spam):
  TN=4732  FP=1
  FN=0  TP=5360

```

```bash

python tools/evaluate_local.py --lot data/Lot1 --model models/model.joblib

[OVERALL] Using threshold=0.48 (scores=proba)
  acc=0.9997 prec=0.9994 rec=1.0000 f1=0.9997
Confusion matrix (labels: 0=clean, 1=spam):
  TN=2880  FP=2
  FN=0  TP=3349
```

## Running on Lot3

```bash

unzip project.zip
cd IURASCU_Danut
python main.py -scan ../data/Lot3 output.txt
```
