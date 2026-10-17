# CT2Eco

Dataset sintetico di video ecografici da CT (TotalSegmentator) con supervisione geometrica e anatomica 3D in stile D4RT.
La rete riceve solo i frame; le etichette vengono dalla CT e dalle pose della sonda.

## Setup

```bash
mamba create -n ct2eco -c conda-forge python=3.11 numpy scipy nibabel matplotlib pyarrow pyyaml pytest
mamba run -n ct2eco pip install torch --index-url https://download.pytorch.org/whl/cu124
pytest -q          # include un test end-to-end su fantocci sintetici
```

## Uso

```bash
python scripts/build_dataset.py --totalseg-dir data/raw/totalseg_small --out data/pilot_v0   # 20 CT x 8 sweep x 32 frame
python scripts/check_dataset.py data/pilot_v0 --overlays
```

Il build riprende da dove si è interrotto (sweep completi = presenza di `sweep.json`).

```python
from ct2eco.queries import QueryDataset
train = QueryDataset("data/pilot_v0", "train", queries_per_video=256)   # query campionate online
val = QueryDataset("data/pilot_v0", "val", fixed=True)                  # query fisse da queries/val.parquet
```

## Modello US-D4RT v0.1 (`ct2eco/model.py`)

```bash
python scripts/train.py --data data/pilot_v0 --out runs/v0_1      # log in runs/v0_1/log.jsonl, best.pt / last.pt
```

- **Encoder:** CNN da zero con stride 16 → griglia 14×14 = 196 token per frame, dimensione 384 (stessa griglia di un ViT-B/16).
  I frame vengono ridimensionati a 224×224 dentro il modello.
- **Positional encoding:** Fourier delle coordinate in mm del centro di ogni patch (dalla calibrazione) + Fourier dell'indice di frame.
- **Transformer multi-vista:** 4 blocchi, alternati frame-wise / globale (VGGT), 6 teste. Output `[B, T, 196, 384]`.
- **Decoder:** query `(u, v, t_s, t_r)` = Fourier(mm) + Fourier(frame) + feature locale interpolata dal frame sorgente.
  4 blocchi di sola cross-attention, quindi le query sono indipendenti. Output `(x, y, z)` in mm nel frame sonda `t_r`.
- **Loss:** SmoothL1 su coordinate / 50 mm (beta 1 mm). Train su clip di 16 frame; val su sweep interi (32 frame) con query fisse.
- **Metriche:** errore 3D medio (tutte / in-slice / fuori slice), errore |z|, e la baseline "sonda ferma" (`val_identity_mm`).

## Struttura

```
ct2eco/
  geometry.py      convenzioni dei frame, p_t = T_t^-1 T_s p_s
  case.py          soggetto TotalSegmentator -> CT + segmentazione multilabel RAS; lookup label/CT/SDF
  sweep.py         traiettorie della sonda sulla cute anteriore
  render.py        slicing + speckle 3D fisso + simulatore SonoGym
  sonogym_conv.py  simulatore convoluzionale SonoGym (vendored, MIT)
  queries.py       campionamento stratificato, etichette, torch Dataset
scripts/           build_dataset.py, check_dataset.py
```

Dataset prodotto:

```
<out>/metadata/label_map.json, splits.json      split per caso CT
<out>/cases/<id>/ct.nii.gz, segmentation.nii.gz  RAS; 0 aria, 1 corpo non segmentato, 2.. strutture
<out>/sweeps/<id>_sweepKK/frames.npy (F,H,W) uint8, labels.npy (F,H,W), poses.npy (F,4,4), sweep.json
<out>/queries/{train,val,test}.parquet
```

## Convenzioni

- Frame sonda (mm): `x` laterale (colonne), `y` profondità (righe), `z` = x × y elevazione. Piano immagine `z = 0`,
  origine al centro del trasduttore, pixel `(u, v)` -> `x = (u - (W-1)/2)·dx`, `y = v·dy`.
- `poses[t]` = T_t, sonda -> mondo (mm RAS della CT canonica).
- Sonda lineare, 200×150 px a 0.4 mm (default SonoGym model-based), spessore geometrico della sezione 1 mm.

## Campi delle query

| campo | significato |
| --- | --- |
| `source_frame`, `target_frame`, `source_u`, `source_v` | query `(u, v, t_s, t_t)` |
| `target_x_mm`, `target_y_mm`, `target_z_mm` | punto nel frame sonda target; `z` = distanza firmata dal piano |
| `target_u`, `target_v` | proiezione ortogonale in pixel target |
| `target_in_fov` | proiezione dentro il rettangolo immagine |
| `target_in_slice` | in FOV e `|z| <= spessore/2` |
| `anatomical_label` | label CT nel punto (id di `label_map.json`) |
| `sdf_mm` | distanza firmata (negativa dentro) da ciascuna struttura di `queries.SDF_STRUCTURES`, NaN se assente |
| `stratum` | `interface` 50%, `inside` 25%, `other` 25% |
| `quality_flags` | bit 1: punto fuori dal volume CT |

## Simulazione

`render.py` riusa il simulatore convoluzionale di SonoGym senza Isaac Sim. Le pose sono nostre, quindi esatte.
Lo speckle (T0, T1, Vl) è un rumore deterministico legato a celle 3D del mondo: gli stessi scatterer per tutto lo sweep,
nuovi scatterer e jitter ±20% su attenuazione e intensità di scattering per ogni sweep.
Modalità CT come nei task SonoGym (`if_use_ct=True`, HU in [-300, 3000] -> 0..255); `--no-ct` per usare solo le label.

Limiti noti: anatomia statica (nessuna deformazione), visibilità solo geometrica (non acustica),
sweep solo da cute anteriore, speckle quantizzato a celle di 0.4 mm (decorrelazione a gradini fuori piano).
