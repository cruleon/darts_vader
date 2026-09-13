# 🎯 DartVision

Sistema di computer vision per il punteggio automatico a freccette (501), da una
**singola webcam**, con calibrazione **automatica** tramite marker ArUco e una
dashboard live per punteggio, storico e heatmap degli impatti.

Progetto di portfolio pensato per essere letto quanto per essere eseguito: ogni
scelta progettuale è motivata più sotto, non solo implementata.

![Dashboard live](docs/images/dashboard-overview.jpg)

## Indice

- [Cosa fa](#cosa-fa)
- [Architettura](#architettura)
- [Perché queste scelte progettuali](#perché-queste-scelte-progettuali)
- [Struttura del repository](#struttura-del-repository)
- [Setup](#setup)
- [Configurazione fisica del bersaglio](#configurazione-fisica-del-bersaglio)
- [Stampare e posizionare i marker ArUco](#stampare-e-posizionare-i-marker-aruco)
- [Utilizzo](#utilizzo)
- [Test](#test)
- [Limiti noti](#limiti-noti)
- [Roadmap](#roadmap)

## Cosa fa

1. Legge un video (file mp4) o una **webcam live**.
2. Rileva 4+ marker ArUco posizionati attorno al bersaglio e calcola
   l'homography che raddrizza la vista in un cerchio perfetto — **ricalcolata
   ad ogni frame**, quindi robusta a una webcam che cambia posizione tra una
   sessione e l'altra.
3. Rileva quando una nuova freccetta compare (differenza tra frame) e ne
   isola la punta, convertendola in coordinate sul piano raddrizzato — con
   due detector intercambiabili: un'euristica geometrica classica
   (`FrameDiffDetector`) o un modello ML di keypoint detection
   (`MLTipDetector`), più robusto su freccette ravvicinate/sovrapposte
   (vedi [Perché queste scelte progettuali](#perché-queste-scelte-progettuali)).
4. Converte la coordinata in settore/anello/punteggio e applica le regole
   della modalità 501 (bust, chiusura su doppio o bullseye interno,
   distanza dal target di chiusura).
5. Salva ogni freccetta in SQLite.
6. Mostra tutto in una dashboard Streamlit live: punteggio corrente, medie,
   residuo, storico, heatmap interattiva sul bersaglio, statistiche di fine
   partita.

![Statistiche di fine partita](docs/images/dashboard-end-of-game.jpg)

## Architettura

Il sistema è stato costruito **un livello alla volta, dal segnale verso
l'alto**, verificando ciascuno prima di passare al successivo — lo stesso
ordine in cui va letto:

```mermaid
flowchart LR
    subgraph L1["1 · input + calibrazione"]
        A["FrameSource<br/>VideoFileSource / WebcamSource"]
        B["ArucoDetector"]
        C["Calibrator<br/>homography"]
        A --> B --> C
    end

    subgraph L2["2 · detection (ImpactDetector)"]
        M["motion_regions<br/>(regione cambiata, condiviso)"]
        D["FrameDiffDetector<br/>+ tip_extraction"]
        N["MLTipDetector<br/>+ YOLO-pose"]
        M --> D
        M --> N
    end

    subgraph L3["3 · scoring"]
        E["score_point<br/>x,y -&gt; settore/anello"]
        F["Game501<br/>bust · chiusura · distanza"]
        E --> F
    end

    subgraph L4["4 · persistenza"]
        G[("SQLite")]
    end

    subgraph L5["5 · dashboard"]
        H["Streamlit + Plotly"]
    end

    A -->|frame grezzi| M
    C -->|homography| D
    C -->|homography| N
    D -->|"Impact (x,y mm)"| E
    N -->|"Impact (x,y mm)"| E
    F -->|ThrowRecord| G
    G --> H
```

`FrameDiffDetector` e `MLTipDetector` sono **intercambiabili** dietro
`ImpactDetector` (flag `--detector {frame-diff,ml}` negli script): nessun
livello a valle sa quale dei due sia in uso.

`pipeline.runner` è l'**unico** modulo che conosce tutti i livelli: li
orchestra, ma nessuno di essi conosce gli altri. In pratica:

| Modulo | Responsabilità | Non sa nulla di |
|---|---|---|
| `input` | Fornire frame, da un file o (in futuro) una webcam | calibrazione, detection, scoring |
| `calibration` | Rilevare i marker, calcolare/ricalcolare l'homography | detection, scoring, persistenza |
| `detection` | Trovare il punto d'impatto sul piano raddrizzato | numerazione dei settori, regole 501 |
| `scoring` | (x,y) → settore/anello/punteggio; regole 501 | OpenCV, SQLite, Streamlit |
| `persistence` | Salvare/rileggere partite e tiri | come un punteggio viene calcolato |
| `pipeline` | Collegare i livelli sopra in un ciclo | — (l'unico che li conosce tutti) |
| `dashboard` | Visualizzare cosa c'è nel database | CV, detection — legge **solo** dal DB |

Questa separazione non è decorativa: è il motivo per cui, ad esempio, ho
potuto costruire e collaudare a fondo `scoring` con dati completamente finti
(nessuna dipendenza da OpenCV), e perché la dashboard non deve cambiare di
una riga quando in futuro cambierà l'algoritmo di detection.

## Perché queste scelte progettuali

**Calibrazione via ArUco, ricalcolata ogni frame.** Il vincolo di partenza è
un setup mobile: la webcam può spostarsi tra una sessione e l'altra. Una
calibrazione salvata su disco si romperebbe al primo spostamento. Rilevando
i marker e ricalcolando l'homography ad ogni frame, il sistema si autoregola
sempre — il prezzo è dover gestire con cura il caso "marker insufficienti"
(vedi sotto), non un crash.

**Fallback esplicito, mai un'eccezione.** `Calibrator.calibrate()` ritorna
sempre un `CalibrationResult` con `success` e un messaggio diagnostico,
anche quando i marker rilevati sono troppo pochi. Il chiamante scarta il
frame e continua: in un flusso video continuo, un singolo frame mal
illuminato non deve mai interrompere la sessione.

**Detection dietro un'interfaccia astratta (`ImpactDetector`).** L'algoritmo
di detection è la parte più delicata e più probabile da rifare (un domani
con un modello ML). Tenerlo dietro un'interfaccia con firma fissa
(due frame + homography → `Impact` sul piano o `None`) significa poter
sostituire `FrameDiffDetector` senza toccare scoring, persistenza o
dashboard — e poterlo testare con frame finti, senza hardware.

**Diff sui frame grezzi, non su quelli raddrizzati.** Ri-raddrizzare ogni
frame con `warpPerspective` prima di confrontarlo col precedente introduce
un micro-jitter: l'homography viene ricalcolata ogni frame dai marker
rilevati, con un minimo di rumore di sub-pixel, per cui l'intera immagine
raddrizzata "vibra" leggermente anche a telecamera ferma — abbastanza da
generare falsi positivi nella differenza tra frame. Si confrontano invece i
frame della camera così come sono (stabili, se la camera non si muove), e
solo i contorni candidati che superano la prima soglia grezza vengono
proiettati sul piano raddrizzato — dove le soglie di area (config) restano
espresse in unità fisiche stabili, indipendenti da quanto la webcam sia
vicina o lontana dal bersaglio.

**Detection ML ibrida, non "un modello su ogni frame".** `MLTipDetector`
riusa la stessa maschera "regione cambiata" di `FrameDiffDetector`
(estratta in `detection/motion_regions.py`) come proposta economica di
ROI sui frame grezzi, e affida a un modello YOLO-pose (1 keypoint = la
punta) solo il compito di localizzare la punta con precisione dentro
ciascun ritaglio candidato, raddrizzato sul piano. Le alternative scartate:
bbox-only richiederebbe comunque un'euristica geometrica a valle per
isolare la punta dentro il box (stesso problema di fondo su freccette
ravvicinate/a mezzaluna); la segmentazione istanza è più pesante per un
guadagno che non serve, perché a valle serve comunque un punto, non
un'area. Il vantaggio del keypoint diretto: ogni freccetta rilevata ha la
propria punta nello stesso forward pass, quindi le freccette sovrapposte
non vengono più fuse in un unico blob come nel frame-diff puro. E
facendo girare il modello solo sui rari eventi "qualcosa è cambiato", su
piccoli ritagli (non sul frame intero, non su ogni frame), resta
sostenibile anche su CPU.

**Dataset di training 100% sintetico.** Nessun dato reale annotato esiste
ancora (nessun setup fisico su cui raccoglierlo — stesso vincolo che ha
già rimandato la webcam). Il dataset per YOLO-pose viene quindi generato
interamente in `dartvision.synthetic` (`scripts/generate_pose_dataset.py`),
disegnando freccette stilizzate (fusto+punta+alette, non un semplice
cerchio) sul piano raddrizzato, con domain randomization (luce, rumore,
blur, compressione, scala) per ridurre il gap tra sintetico e reale. Il
training vero e proprio (`scripts/train_pose_model.py`) è pensato per
girare su una macchina con GPU, non su questa: è uno script standalone,
non parte della pipeline principale.

**La punta è il punto più vicino al centro.** Quando una freccetta si
conficca, la regione rilevata dal frame-diff include punta, fusto e alette.
L'euristica adottata — indicata dalle specifiche del progetto — è scegliere
il punto della sagoma rilevata più vicino al centro del bersaglio: fusto e
alette sporgono verso l'esterno, la punta è l'estremo più "interno". È
isolata in una funzione a sé (`tip_extraction.select_tip_point`) proprio
per poter essere sostituita con un'euristica diversa senza toccare il resto
del detector.

**Il bust è un concetto di turno, non di singola freccetta.** Punto
facile da sbagliare: se la terza freccetta di un turno manda in bust, anche
i punti delle prime due — validi al momento del lancio — vengono annullati,
e il residuo torna a quello di inizio turno. `Game501` lo implementa
esplicitamente, con un test dedicato proprio per questo caso.

**Ordine di costruzione bottom-up.** Ogni livello è stato verificato prima
di costruire il successivo: la calibrazione su un video sintetico con
marker veri (Livello 1) prima ancora di scrivere una riga di detection; la
detection verificata con blob disegnati a mano prima di collegarla allo
scoring; lo scoring testato con coordinate pure, senza nessuna dipendenza
da OpenCV, prima di collegarlo alla persistenza. Questo ha fatto emergere
presto i problemi (es. la geometria dei marker mal dimensionata nel
Livello 1, o la sensibilità del frame-diff a freccette molto ravvicinate —
vedi [Limiti noti](#limiti-noti)) invece che a integrazione ultimata.

## Struttura del repository

```
config/
  board_config.yaml          # UNICA fonte di geometria fisica: raggi anelli,
                              # posizione marker, soglie di detection (classica e ML)
docs/
  images/                     # screenshot per questo README
scripts/
  generate_aruco_markers.py   # genera i PNG dei marker da stampare
  generate_synthetic_video.py # video mp4 sintetico per sviluppo/test (dev tool)
  generate_pose_dataset.py    # dataset sintetico YOLO-pose per addestrare MLTipDetector
  import_deepdarts_dataset.py # converte il dataset reale DeepDarts nello stesso formato
  train_pose_model.py         # training standalone (da eseguire su una macchina con GPU)
  process_video.py            # pipeline completa: video -> DB (--detector frame-diff|ml)
  run_live.py                 # pipeline completa: webcam live -> DB (Ctrl+C per fermare)
  seed_demo_data.py           # popola il DB con dati demo (bypassa la CV, per la dashboard)
  run_on_video.py             # solo Livello 1: diagnostica di calibrazione
  run_dashboard.py            # avvia la dashboard Streamlit
src/dartvision/
  config.py                   # loader tipizzato di board_config.yaml
  input/                      # Livello 1a: FrameSource, VideoFileSource, WebcamSource
  calibration/                # Livello 1b: ArucoDetector, Calibrator/homography
  detection/                  # Livello 2: ImpactDetector, motion_regions (condiviso),
                               # FrameDiffDetector, tip_extraction, MLTipDetector
  synthetic/                  # rendering bersaglio/freccette per test e dataset ML
  scoring/                    # Livello 3: geometria polare, punteggio, regole 501
  persistence/                # Livello 4: modelli, schema SQLite, repository
  pipeline/                   # orchestratore: unisce i livelli 1-4
  analytics.py                 # statistiche di fine partita (logica pura)
  dashboard/                   # Livello 5: app Streamlit, figura Plotly, componenti
tests/                         # pytest (uv run pytest -q per il conteggio aggiornato)
```

## Setup

Richiede Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # installa tutte le dipendenze in .venv
uv run pytest -q   # verifica che tutto funzioni: tutti i test dovrebbero passare
```

Nessun'altra configurazione è necessaria per esplorare il progetto con i
dati dimostrativi già inclusi. Solo per usare `MLTipDetector` (inferenza)
o `scripts/train_pose_model.py` (training) serve l'estensione opzionale
`ml` (porta con sé `ultralytics`/`torch`, non installata di default):

```bash
uv sync --extra ml
```

## Configurazione fisica del bersaglio

Tutta la geometria fisica vive in `config/board_config.yaml`, mai nel
codice: raggi degli anelli, ordine e offset dei settori, dizionario ArUco,
posizione di ciascun marker rispetto al centro del bersaglio, soglie di
detection. Per adattare il sistema a un bersaglio o setup diverso si
modifica solo questo file. `dartvision.config.load_config()` lo valida
(raggi crescenti, nessun settore duplicato, marker sufficienti, geometria
coerente col piano raddrizzato) e fallisce subito con un messaggio chiaro
se qualcosa non torna.

## Stampare e posizionare i marker ArUco

```bash
uv run python scripts/generate_aruco_markers.py
```

Genera in `docs/markers/` un PNG per marker, dimensionato in DPI per
ottenere esattamente il lato fisico configurato (`marker_length_mm`).
Dopo averli stampati e posizionati attorno al bersaglio:

1. Misura la distanza reale del **centro** di ciascun marker dal centro del
   bersaglio (bullseye).
2. Aggiorna i campi `x_mm`/`y_mm` di quel marker in `board_config.yaml`.
3. Verifica con `scripts/run_on_video.py` (sotto) che la calibrazione
   raddrizzi il bersaglio in un cerchio corretto.

## Utilizzo

### Elaborare un video registrato

```bash
uv run python scripts/process_video.py --video path/al/video.mp4
uv run python scripts/process_video.py --video path/al/video.mp4 --detector ml
```

Esegue l'intera pipeline (calibrazione → detection → scoring →
persistenza) e salva la partita in `data/dartvision.db`, stampando un
riepilogo turno per turno. `--detector` sceglie tra `frame-diff`
(default, sempre disponibile) e `ml` (richiede un modello addestrato, vedi
[Detection ML](#perché-queste-scelte-progettuali) e
`scripts/train_pose_model.py`).

### Webcam live

```bash
uv run python scripts/run_live.py --player1 Alice --player2 Bob
```

Stessa pipeline, ma da webcam invece che da file: gira finché non si preme
Ctrl+C, che chiude in modo pulito l'eventuale turno in corso e salva la
partita (stessa logica di fine-sorgente usata per i video). Accetta anche
`--detector {frame-diff,ml}` e `--device-index` per scegliere la webcam.

### Dashboard

```bash
uv run python scripts/run_dashboard.py
```

Apre la dashboard su `http://localhost:8501`, che legge live da
`data/dartvision.db` (aggiornamento automatico ogni 2 secondi via
`st.fragment`, senza ricaricare l'intera pagina).

Se il database è vuoto, popolalo con dati dimostrativi (bypassa la
computer vision, usa direttamente i livelli di scoring/persistenza già
testati):

```bash
uv run python scripts/seed_demo_data.py                 # una partita completa
uv run python scripts/seed_demo_data.py --max-turns 3    # una partita "in corso"
```

### Solo calibrazione (diagnostica Livello 1)

```bash
uv run python scripts/run_on_video.py --video path/al/video.mp4 \
    --output-rectified out_rectified.mp4 --output-annotated out_annotated.mp4
```

Utile per isolare problemi di calibrazione (marker non rilevati, homography
instabile) senza il resto della pipeline.

### Video sintetico (per sviluppo, senza hardware)

```bash
uv run python scripts/generate_synthetic_video.py --dart=0,0 --dart=20,-140
```

Genera un mp4 con bersaglio, marker e freccette finte proiettati con
un'unica homography plausibile, più un `.ground_truth.json` con le
coordinate vere — usato per sviluppare e validare i Livelli 1-2 prima di
avere un setup fisico.

### Dataset e training per `MLTipDetector`

```bash
uv run python scripts/generate_pose_dataset.py --num-samples 5000
```

Genera un dataset sintetico in formato YOLO-pose (`data/pose_dataset/` di
default): freccette stilizzate (fusto+punta+alette) disegnate sul piano
raddrizzato, con posizioni/angoli casuali, freccette ravvicinate/
sovrapposte incluse di proposito, e domain randomization (luce, rumore,
blur, compressione, scala) per ridurre il gap sintetico/reale.

Il training vero e proprio va eseguito su una macchina con GPU (non
questa):

```bash
uv sync --extra ml   # sulla macchina GPU
uv run python scripts/train_pose_model.py --data data/pose_dataset/dataset.yaml --device 0
```

Copia poi `best.pt` nel path indicato da `ml_detection.model_path` in
`config/board_config.yaml` (default `models/dart_tip_pose.pt`) per usare
`--detector ml`.

#### Dati reali (opzionale): dataset DeepDarts

Oltre al dataset sintetico, `scripts/import_deepdarts_dataset.py` converte
il dataset pubblico **DeepDarts** (McNally, CVSports 2021 — 16.050 foto
reali, 32.027 freccette annotate) nello stesso formato YOLO-pose, cosi' i
due dataset si possono mescolare in training. Funziona sfruttando il
fatto che DeepDarts annota 4 punti di calibrazione a confini di settore
standard del bersaglio (5/20, 17/3, 8/11, 13/6, sul bordo esterno
dell'anello doppio) — posizioni fisiche note, le stesse specifiche BDO
gia' usate in `config/board_config.yaml` — permettendo di calcolare
un'omografia diretta verso il nostro piano raddrizzato e riproiettarci
sopra anche le punte reali (vedi `dartvision.synthetic.deepdarts_import`,
verificato empiricamente per allineamento).

Il dataset **non è incluso** in questo repository (licenza CC-BY, richiede
un account IEEE DataPort gratuito):

1. Scarica `labels_pkl.zip` e `cropped_images.zip` da
   [ieee-dataport.org/open-access/deepdarts-dataset](https://ieee-dataport.org/open-access/deepdarts-dataset)
   ed estraili (es. in `training_data_IEEE/`).
2. Converti:
   ```bash
   uv run python scripts/import_deepdarts_dataset.py \
       --labels-pkl training_data_IEEE/labels_pkl/labels.pkl \
       --images-dir training_data_IEEE/cropped_images/800
   ```
3. Passa entrambi i `dataset.yaml` (sintetico + DeepDarts) a
   `scripts/train_pose_model.py --data` (Ultralytics accetta più percorsi
   in `train:`/`val:`).

Citazione: William McNally, "DeepDarts Dataset", IEEE Dataport, 2021,
doi:10.21227/05e7-xs69 ([paper](https://arxiv.org/abs/2105.09880),
[codice](https://github.com/wmcnally/deep-darts)).

## Test

```bash
uv run pytest -q
```

128 test. Priorità data alla logica pura (geometria, punteggio, regole 501,
distanza di chiusura, statistiche): non richiedono OpenCV né un database,
girano in meno di 5 secondi. Calibrazione, detection, persistenza e
pipeline hanno invece test con dati sintetici (marker finti, frame
disegnati a mano, database temporanei) — nessun test richiede un video, un
DB, una webcam o un GPU reali. `MLTipDetector` è testato iniettando un
modello finto scriptato (`model=...` nel costruttore): nessun test
richiede `ultralytics` installato o un modello realmente addestrato — la
validazione con un modello vero, dopo un training reale, resta manuale
(es. con uno script diagnostico stile `run_on_video.py`).

## Limiti noti

**Il frame-diff fatica con freccette molto ravvicinate (`FrameDiffDetector`).**
Nel costruire i dati dimostrativi ho verificato che quando due impatti
atterrano a pochi millimetri l'uno dall'altro in tempi ravvicinati, la
compressione video (mp4/H.264) introduce artefatti che deformano la
regione "nuova" rilevata in una mezzaluna invece che in un cerchio pulito,
confondendo l'euristica della punta. È il motivo per cui
`scripts/seed_demo_data.py` genera i dati dimostrativi della dashboard
direttamente dai Livelli 3-4 (già validati), mentre `process_video.py`
resta il percorso reale via computer vision — accurato quando gli impatti
sono ragionevolmente separati, come verificato nei test di integrazione.
`MLTipDetector` è stato introdotto proprio per attenuare questo limite
(ogni freccetta rilevata ha la propria punta, anche se sovrapposta ad
altre), ma vedi il punto successivo sul suo limite attuale.

**`MLTipDetector` non è mai stato validato sul nostro setup fisico reale.**
Il dataset di training può includere dati sintetici
(`scripts/generate_pose_dataset.py`) e/o il dataset reale pubblico
DeepDarts (`scripts/import_deepdarts_dataset.py`, vedi
[Utilizzo](#utilizzo)) — ma quest'ultimo è stato girato su bersagli e
webcam diversi dai nostri. Un fine-tuning su un piccolo set raccolto con
il **nostro** setup fisico (marker, webcam, bersaglio specifici), una
volta disponibile, resta lavoro futuro (vedi [Roadmap](#roadmap)).

**Un solo impatto per confronto tra frame, in entrambi i detector.**
`ImpactDetector.detect()` ritorna al più un `Impact` per coppia di frame,
per costruzione (sia in `FrameDiffDetector` che in `MLTipDetector`, che pure
può rilevare più freccette in un frame ma ne restituisce solo la più
confidente): due freccette che atterrano nello stesso identico frame
verrebbero quindi rilevate come una sola. In pratica il framerate della
camera rende l'evento raro, ma è un limite noto, non un edge case gestito.

**Nessun rilevamento di fine turno anticipata.** Un turno viene chiuso al
terzo impatto rilevato, o alla fine del video se ne restano meno di tre in
sospeso. Per un file registrato questo non è un problema pratico (i turni
sono quasi sempre completi), ma su uno stream live continuo non c'è ancora
modo di riconoscere che un turno è finito dopo 1-2 freccette (es. un
checkout precoce) senza aspettare la terza. Punto naturale in cui la
futura estensione a webcam live dovrà aggiungere un'euristica di confine
turno (timeout di inattività, o la mano che entra a ritirare le
freccette).

## Roadmap

- **Fine-tuning di `MLTipDetector` sul nostro setup fisico**: raccogliere
  un piccolo set di frame reali con la nostra webcam/bersaglio (ora che
  entrambi esistono) ed eseguire un fine-tuning sul checkpoint
  sintetico+DeepDarts, per colpire il gap documentato in
  [Limiti noti](#limiti-noti).
- **Euristica di confine turno per stream continui** (webcam live: nessun
  modo oggi di riconoscere un turno chiuso a 1-2 freccette senza aspettare
  la terza — vedi sopra).
- **Estensione multi-keypoint**: oggi `MLTipDetector` predice un solo
  keypoint (la punta); un secondo keypoint su fusto/aletta potrebbe
  aiutare a distinguere l'orientamento e a filtrare falsi positivi.
- Modello di sfondo multi-frame per `FrameDiffDetector` (es. media mobile
  o MOG2), per migliorare la robustezza a impatti ravvicinati e a
  mani/ombre in movimento sul percorso classico.
