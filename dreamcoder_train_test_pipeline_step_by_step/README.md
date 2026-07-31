# DreamCoder Train/Test Pipeline Robust

Diese Pipeline führt eine saubere DreamCoder-Auswertung mit getrenntem Trainings- und Testdatensatz durch.

Der Ablauf ist:

```text
Trainingsdatensatz → DreamCoder-Training/Suche → Testdatensatz → Export → Normalisierung → Metriken
```

Die Pipeline wurde gegenüber der vorherigen Version robuster gemacht. Der wichtigste Fix betrifft `step04`: DreamCoder gibt manchmal im Terminal `HIT test_task_...` aus, speichert diese Test-HITs aber nicht zuverlässig an der Stelle im Ergebnisobjekt, an der die Pipeline zuerst gesucht hat. Deshalb durchsucht die neue Version mehrere mögliche Frontier-Strukturen und verwendet zusätzlich einen kontrollierten stdout-Fallback.

---

## 1. Ordnerstruktur

```text
dreamcoder_train_test_pipeline_robust/
├── dataset/
│   ├── README_dataset.md
│   ├── T=1_train.json
│   ├── T=1_test.json
│   └── T=2_test.json
├── scripts/
│   ├── step01_validate_datasets.py
│   ├── step02_convert_train_test_tasks.py
│   ├── step03_create_dreamcoder_task_pickles.py
│   ├── step04_run_dreamcoder_train_test.py
│   ├── step05_detect_operations.py
│   ├── step06_normalize_programs.py
│   ├── step07_calculate_metrics.py
│   └── step08_summarize_results.py
├── outputs/
├── run_all.sh
├── README.md
└── DreamCoder_Train_Test_Workflow.ipynb
```

`outputs/` wird automatisch beim Ausführen gefüllt.

---

## 2. Voraussetzungen

Die Pipeline erwartet ein funktionierendes DreamCoder-Repository, zum Beispiel:

```bash
/mnt/c/BA/ec
```

Die Conda-Umgebung muss aktiv sein:

```bash
cd /mnt/c/BA/ec
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dreamcoder
python --version
```

Erwartet wird ungefähr:

```text
Python 3.8.20
```

DreamCoder sollte grundsätzlich funktionieren:

```bash
python bin/list.py --help
```

---

## 3. Datensatz vorbereiten

Die Pipeline erwartet standardmäßig:

```text
dataset/T=2_train.json
dataset/T=2_test.json
```

In der ZIP kann `T=2_train.json` fehlen, weil diese Datei groß sein kann. Kopiere sie dann in den Pipeline-Ordner:

```bash
cd /mnt/c/BA/ec/dreamcoder_train_test_pipeline_robust
cp /mnt/c/BA/deepcoder/dataset/T=2_train.json dataset/T=2_train.json
```

Danach prüfen:

```bash
ls dataset
```

Du solltest sehen:

```text
T=2_train.json
T=2_test.json
```

Wichtig: Die JSON-Dateien enthalten PBE-Listenaufgaben mit Referenzprogrammen und Input-Output-Beispielen. Die Aufgaben werden nicht direkt roh an DreamCoder gegeben, sondern zuerst in DreamCoder-`Task`-Objekte konvertiert.

---

## 4. Empfohlener erster Testlauf

Starte zuerst einen kleinen, schnellen Lauf. Damit prüfst du, ob alle Schritte funktionieren:

```bash
cd /mnt/c/BA/ec/dreamcoder_train_test_pipeline_robust

MAX_TRAIN_TASKS=100 \
TRAIN_DATASET_FILE=dataset/T=2_train.json \
TEST_DATASET_FILE=dataset/T=2_test.json \
DREAMCODER_REPO_ROOT=/mnt/c/BA/ec \
DREAMCODER_TIMEOUT=10 \
DREAMCODER_TESTING_TIMEOUT=10 \
DREAMCODER_ITERATIONS=1 \
DREAMCODER_FRONTIER_SIZE=10 \
DREAMCODER_USE_RECOGNITION=false \
DREAMCODER_NO_CONSOLIDATION=true \
DREAMCODER_CPUS=1 \
bash run_all.sh
```

Dieser Lauf ist nur ein Funktionstest. Er soll nicht als finale Modellleistung interpretiert werden.

---

## 5. Sinnvollerer Zwischenlauf

Wenn der kleine Lauf funktioniert, starte einen mittleren Lauf:

```bash
cd /mnt/c/BA/ec/dreamcoder_train_test_pipeline_robust

MAX_TRAIN_TASKS=300 \
TRAIN_DATASET_FILE=dataset/T=2_train.json \
TEST_DATASET_FILE=dataset/T=2_test.json \
DREAMCODER_REPO_ROOT=/mnt/c/BA/ec \
DREAMCODER_TIMEOUT=15 \
DREAMCODER_TESTING_TIMEOUT=15 \
DREAMCODER_ITERATIONS=2 \
DREAMCODER_FRONTIER_SIZE=30 \
DREAMCODER_USE_RECOGNITION=true \
DREAMCODER_NO_CONSOLIDATION=true \
DREAMCODER_CPUS=1 \
bash run_all.sh
```

Dieser Lauf nutzt:

```text
300 Trainingsaufgaben
99 Testaufgaben
2 Iterationen
Recognition Model aktiviert
Grammar-Konsolidierung deaktiviert
15 Sekunden Enumeration pro Phase
15 Sekunden Testing Timeout
```

`DREAMCODER_NO_CONSOLIDATION=true` ist stabiler, weil dafür kein funktionierender Rust-Compressor nötig ist.

---

## 6. Größerer Lauf

Erst wenn die kleinen Läufe stabil sind:

```bash
cd /mnt/c/BA/ec/dreamcoder_train_test_pipeline_robust

MAX_TRAIN_TASKS=0 \
TRAIN_DATASET_FILE=dataset/T=2_train.json \
TEST_DATASET_FILE=dataset/T=2_test.json \
DREAMCODER_REPO_ROOT=/mnt/c/BA/ec \
DREAMCODER_TIMEOUT=30 \
DREAMCODER_TESTING_TIMEOUT=30 \
DREAMCODER_ITERATIONS=3 \
DREAMCODER_FRONTIER_SIZE=50 \
DREAMCODER_USE_RECOGNITION=true \
DREAMCODER_NO_CONSOLIDATION=true \
DREAMCODER_CPUS=1 \
bash run_all.sh
```

`MAX_TRAIN_TASKS=0` bedeutet: alle nutzbaren Trainingsaufgaben verwenden.

---

## 7. Wichtige Parameter

| Parameter | Bedeutung |
|---|---|
| `TRAIN_DATASET_FILE` | Trainingsdatensatz relativ zum Pipeline-Ordner |
| `TEST_DATASET_FILE` | Testdatensatz relativ zum Pipeline-Ordner |
| `DREAMCODER_REPO_ROOT` | Pfad zum DreamCoder-Repository |
| `DREAMCODER_TIMEOUT` | Suchzeit für Trainingsaufgaben |
| `DREAMCODER_TESTING_TIMEOUT` | Suchzeit für Testaufgaben |
| `DREAMCODER_ITERATIONS` | Anzahl der DreamCoder-Iterationen |
| `DREAMCODER_FRONTIER_SIZE` | maximale Frontier-Größe |
| `DREAMCODER_USE_RECOGNITION` | Recognition Model aktivieren/deaktivieren |
| `DREAMCODER_NO_CONSOLIDATION` | Grammar-Konsolidierung deaktivieren/aktivieren |
| `DREAMCODER_CPUS` | Anzahl CPUs, die an DreamCoder übergeben wird |
| `MAX_TRAIN_TASKS` | Anzahl Trainingsaufgaben; `0` bedeutet alle |

---

## 8. Was macht jeder Schritt?

### Step 01: `step01_validate_datasets.py`

Prüft Trainings- und Testdatensatz.

Erfasst unter anderem:

```text
Anzahl Aufgaben
Programmlängen
Beispiele pro Aufgabe
Input-Arity
Output-Typen
Null-Outputs
fehlende Programme
fehlende Beispiele
```

Ergebnisse:

```text
outputs/<RUN_KEY>/step01_validate_datasets/step01_dataset_summary.json
outputs/<RUN_KEY>/step01_validate_datasets/step01_dataset_summary.txt
```

---

### Step 02: `step02_convert_train_test_tasks.py`

Konvertiert die JSON-Aufgaben in ein neutrales Zwischenformat.

Dabei werden:

```text
Aufgaben mit nur null-Outputs übersprungen
Task-Namen erzeugt
Request-Typen bestimmt
Referenzprogramme gespeichert
Input-Output-Beispiele gespeichert
Train/Test getrennt exportiert
```

Ergebnisse:

```text
outputs/<RUN_KEY>/step02_convert_train_test/step02_train_tasks.json
outputs/<RUN_KEY>/step02_convert_train_test/step02_test_tasks.json
outputs/<RUN_KEY>/step02_convert_train_test/step02_train_tasks.csv
outputs/<RUN_KEY>/step02_convert_train_test/step02_test_tasks.csv
outputs/<RUN_KEY>/step02_convert_train_test/step02_skipped_tasks.json
outputs/<RUN_KEY>/step02_convert_train_test/step02_conversion_summary.json
```

---

### Step 03: `step03_create_dreamcoder_task_pickles.py`

Erzeugt echte DreamCoder-`Task`-Objekte aus dem Zwischenformat.

Dabei werden die Input-Output-Beispiele in das Format gebracht, das DreamCoder erwartet.

Ergebnisse:

```text
outputs/<RUN_KEY>/step03_create_task_pickles/step03_train_tasks.pkl
outputs/<RUN_KEY>/step03_create_task_pickles/step03_test_tasks.pkl
outputs/<RUN_KEY>/step03_create_task_pickles/step03_task_pickle_summary.txt
```

---

### Step 04: `step04_run_dreamcoder_train_test.py`

Das ist der wichtigste Schritt.

Er macht:

```text
DreamCoder-Primitives laden
Grammar erzeugen
Trainingsaufgaben laden
Testaufgaben laden
ecIterator starten
DreamCoder auf Trainingsaufgaben laufen lassen
Held-out Testaufgaben evaluieren
Train- und Testlösungen exportieren
```

Robustheitsänderungen in dieser Version:

```text
1. Frontiers werden nicht nur direkt über das Task-Objekt gesucht.
2. Zusätzlich wird über Task-Namen gesucht.
3. Mehrere mögliche Ergebnisattribute werden durchsucht:
   allFrontiers, frontiers, testingFrontiers, testFrontiers,
   heldoutFrontiers, frontiersOverTime usw.
4. DreamCoder-stdout wird live gespeichert.
5. HIT-Zeilen aus stdout werden geparst.
6. Wenn DreamCoder im stdout Test-HITs meldet, aber die CSV leer bleibt,
   werden diese HITs kontrolliert in step04_test_results.csv übernommen.
```

Wichtige Ergebnisdateien:

```text
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_stdout.log
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_stdout_hits.csv
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_stdout_hits.json
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_train_results.csv
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_test_results.csv
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_train_test_summary.json
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_train_frontier_diagnostics.json
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_test_frontier_diagnostics.json
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_dreamcoder_train_test_result.pkl
```

Die wichtigste Datei ist:

```text
outputs/<RUN_KEY>/step04_run_dreamcoder/step04_test_results.csv
```

Dort stehen die Testlösungen. Die Spalte `solution_source` zeigt, woher eine Lösung kommt:

```text
frontier          = Lösung wurde direkt aus dem DreamCoder-Ergebnisobjekt gelesen
stdout_fallback   = Lösung wurde aus einer DreamCoder-HIT-Zeile im stdout übernommen
leer              = ungelöste Aufgabe
```

---

### Step 05: `step05_detect_operations.py`

Prüft, welche Operationen in Referenzprogrammen und DreamCoder-Lösungen vorkommen.

Dieser Schritt ist eine Qualitätskontrolle vor der Normalisierung.

Ergebnisse:

```text
outputs/<RUN_KEY>/step05_detect_operations/step05_operation_check.csv
outputs/<RUN_KEY>/step05_detect_operations/step05_reference_operations.csv
outputs/<RUN_KEY>/step05_detect_operations/step05_solution_tokens.csv
outputs/<RUN_KEY>/step05_detect_operations/step05_unknown_solution_tokens.csv
outputs/<RUN_KEY>/step05_detect_operations/step05_operation_check_summary.txt
```

---

### Step 06: `step06_normalize_programs.py`

Normalisiert Referenzprogramme und DreamCoder-Lösungen auf gemeinsame Operationstokens.

Beispiel:

```text
Referenzprogramm: MAP,*2
normalisiert:     MAP_MULT2
```

Diese Version korrigiert außerdem den früheren Fehler, dass ungelöste Aufgaben als triviale Lösungen gezählt wurden. Jetzt gilt:

```text
trivial_solution = 1 nur wenn solved = true und die Lösung wirklich trivial ist
```

Ergebnisse:

```text
outputs/<RUN_KEY>/step06_normalize_programs/step06_normalized_test_programs.csv
outputs/<RUN_KEY>/step06_normalize_programs/step06_normalization_summary.json
```

---

### Step 07: `step07_calculate_metrics.py`

Berechnet Accuracy und Programmmetriken auf den Testdaten.

Berechnet werden:

```text
Accuracy
normalized_operation_score
normalized_position_score
normalized_order_score
normalized_edit_score
abstract_operation_score
abstract_position_score
abstract_order_score
abstract_edit_score
```

Accuracy wird aus der Spalte `solved` berechnet.

Ergebnisse:

```text
outputs/<RUN_KEY>/step07_calculate_metrics/step07_test_results_with_metrics.csv
outputs/<RUN_KEY>/step07_calculate_metrics/step07_metrics_summary.json
```

---

### Step 08: `step08_summarize_results.py`

Erzeugt die finale Zusammenfassung.

Ergebnisse:

```text
outputs/<RUN_KEY>/step08_summarize_results/step08_summary.txt
outputs/<RUN_KEY>/step08_summarize_results/dreamcoder_test_results_complete.csv
outputs/<RUN_KEY>/step08_summarize_results/dreamcoder_test_solved_tasks.csv
outputs/<RUN_KEY>/step08_summarize_results/dreamcoder_test_unsolved_tasks.csv
```

Die wichtigste Datei für die schnelle Kontrolle ist:

```text
outputs/<RUN_KEY>/step08_summarize_results/step08_summary.txt
```

---

## 9. Ergebnis schnell anzeigen

Nach einem Lauf:

```bash
cd /mnt/c/BA/ec/dreamcoder_train_test_pipeline_robust
LATEST=$(ls -td outputs/* | head -1)
cat "$LATEST/step08_summarize_results/step08_summary.txt"
```

Testlösungen anzeigen:

```bash
cat "$LATEST/step08_summarize_results/dreamcoder_test_solved_tasks.csv"
```

Prüfen, ob stdout-Fallback verwendet wurde:

```bash
cat "$LATEST/step04_run_dreamcoder/step04_train_test_summary.json"
```

Oder nur die stdout-HITs:

```bash
cat "$LATEST/step04_run_dreamcoder/step04_stdout_hits.csv"
```

---

## 10. Wie du die Ergebnisse interpretierst

Wenn `step04_stdout.log` zum Beispiel zeigt:

```text
Hits 11/99 testing tasks
```

und die Pipeline jetzt ebenfalls `Solved test tasks: 11` schreibt, dann wurde der frühere Exportfehler korrigiert.

Wenn `solution_source = stdout_fallback` vorkommt, heißt das:

```text
DreamCoder hat die Lösung im Terminal ausgegeben,
die Frontier-Struktur im Ergebnisobjekt enthielt sie aber nicht an der erwarteten Stelle.
```

Das ist methodisch transparent, weil die Quelle jeder Lösung in der CSV markiert wird.

---

## 11. Wichtige wissenschaftliche Formulierung

Für die Bachelorarbeit kannst du dazu schreiben:

> Die DreamCoder-Auswertung verwendet getrennte Trainings- und Testaufgaben. Die JSON-Aufgaben werden zunächst in DreamCoder-Task-Objekte überführt. DreamCoder wird anschließend auf den Trainingsaufgaben ausgeführt und mit `testingTimeout` auf den Testaufgaben evaluiert. Da DreamCoder Test-HITs in einigen Fällen im stdout ausgibt, diese aber nicht zuverlässig in der Frontier-Struktur des Ergebnisobjekts speichert, protokolliert die Pipeline die Quelle jeder Lösung. Lösungen werden bevorzugt aus den Frontiers gelesen; falls dort keine Lösung vorhanden ist, aber eine eindeutige HIT-Zeile im stdout existiert, wird diese Lösung als `stdout_fallback` markiert übernommen.

---

## 12. Typische Probleme

### Problem: `T=2_train.json` fehlt

Lösung:

```bash
cp /mnt/c/BA/deepcoder/dataset/T=2_train.json dataset/T=2_train.json
```

### Problem: Lauf dauert sehr lange

Nimm zuerst:

```bash
MAX_TRAIN_TASKS=100
DREAMCODER_TIMEOUT=10
DREAMCODER_TESTING_TIMEOUT=10
DREAMCODER_ITERATIONS=1
```

### Problem: `Accuracy = 0`, obwohl stdout HITs zeigt

In dieser robusten Version sollte das nicht mehr passieren. Prüfe:

```bash
cat "$LATEST/step04_run_dreamcoder/step04_stdout_hits.csv"
cat "$LATEST/step04_run_dreamcoder/step04_test_results.csv"
```

Wenn HITs vorhanden sind, sollten die entsprechenden Aufgaben in `step04_test_results.csv` als `solved=True` stehen.

