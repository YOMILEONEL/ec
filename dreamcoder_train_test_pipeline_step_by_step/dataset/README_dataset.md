# Dataset folder

Place your PBE JSON datasets here.

Expected default files for the train/test workflow:

```text
T=2_train.json
T=2_test.json
```

The repository includes only small/example files when available. If `T=2_train.json` is missing, copy it from your local dataset folder into this directory.

Expected JSON format per task:

```json
{
  "program": "LIST|MAP,*2,0|REVERSE,1",
  "examples": [
    {"inputs": [[1,2,3]], "output": [6,4,2]}
  ]
}
```
