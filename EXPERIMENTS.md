# Registro esperimenti

> **Archivio storico (US-D4RT v0.x).** Questo registro conserva i risultati del modello iniziale. Per Local Stream L1/L2b/L3 e i Gate T0/T1, consultare [docs/RESULTS.md](docs/RESULTS.md) e [docs/DECISIONS.md](docs/DECISIONS.md). Non confrontare direttamente metriche e supporti dei due protocolli.

Errori medi in mm sul **val** (3 CT mai viste), query fisse nei primi 16 frame di ogni sweep. Train: 14 CT, 112 sweep, 3000 step, batch 4.
Baseline "ferma" = la sonda non si è mossa (punto sorgente con z = 0). |z| = errore su |z| (ignora il verso).

## Storico prima della sessione autonoma

| Run | Dati | Cosa cambia | val tutte | val z | val \|z\| | note |
| --- | --- | --- | --- | --- | --- | --- |
| v0 | pilot_v0 | traiettorie monotone | 2.12 | 1.45 | – | = baseline cieca (2.05/1.43): impara solo la velocità media |
| v1_alltrain | pilot_v1 | traiettorie casuali | 3.10 | 2.47 | – | = baseline ferma (3.05/2.48) |
| ruq_alltrain | pilot_ruq | solo fegato/colecisti | 3.01 | 2.39 | – | = baseline (2.96/2.41): la regione non cambia nulla |
| v1_signinv | pilot_v1 | loss invariante al segno | 3.39 | 2.79 | **1.63** | \|z\| -34% vs ferma (2.48): l'entità si impara e generalizza |
| v1_curriculum | pilot_v1 | signinv -> loss col segno | 3.12 | 2.46 | 2.37 | dimentica \|z\|, verso non appreso (sign_ok fit 0.47) |
| v1_control | pilot_v1 | v1_alltrain -> +3000 step col segno | 3.13 | 2.47 | 2.34 | = curriculum: il curriculum non aiuta; sign_ok 0.71 in entrambi = bias del val |

Diagnosi di v1_signinv (offline): x 1.07 vs ferma 1.03, y 0.84 vs 0.78 -> **nessun movimento nel piano appreso**.
\|z\| per distanza tra frame: guadagno solo +29% a |dt| = 1 (0.80 vs 1.12), +42% a |dt| 9-15.
Ipotesi: i frame si confrontano solo tra token da ~4x6 mm, la decorrelazione dello speckle (pixel) si perde.

## Sessione autonoma

### Confronto con D4RT (arXiv 2512.08924, verificato da subagent)
Differenze, in ordine di probabile impatto sui nostri fallimenti:
1. **Patch RGB 9x9 a risoluzione nativa dal frame sorgente nella query** (Tab. 7: senza, ATE ~2x). Noi: solo feature di token (~4x6 mm). -> sospettato principale per x,y.
2. Scala: ViT-g pretrained VideoMAE, 500k step, 2048 query/clip, clip da 48 frame (Tab. 11: senza pretraining AbsRel 0.74 vs 0.30). Noi: CNN da zero, 3k step, 256 query.
3. Loss pesata per confidenza per query (c*L - 0.2 log c), Tab. 8: componente più critico per la posa.
4. Normalizzazione sign*log(1+|x|) e loss ausiliarie (2D target, **spostamento**, visibilità, normali). Noi: posizione assoluta / 50 mm.
5. Tubelet 2x16x16 (fusione temporale nel patch embedding) ~ nostra early fusion.
6. Il segno di z non ha analogo in D4RT (la prospettiva rende osservabile la direzione, il fascio simmetrico no).

### Esperimenti

**E1 — early fusion [t-1,t,t+1] + loss invariante al segno, pilot_v1** (`runs/e1_fusion_signinv`)
Ipotesi: confrontare frame adiacenti a livello pixel migliora |z| sui passi piccoli e x,y.
Risultato (val): all 3.37, all* 2.77, |z| 1.66, |z|dt1 0.81, x 1.07, y 0.85 — identico a v1_signinv (|z| 1.63, dt1 0.80).
Fit (train): x 1.12 vs ferma 1.08, y 0.88 vs 0.80 -> **x,y non appresi nemmeno sui dati di training**: non è generalizzazione, manca informazione/capacità.
Decisione: early fusion non è il collo di bottiglia. Tenuta (non peggiora, accelera |z| a metà training). Prossimi: E2 fascio 2 mm, E3 patch locale 9x9 (D4RT).

**E2 — E1 + fascio elevazionale FWHM 2 mm, pilot_v2 (traiettorie identiche a v1)** (`runs/e2_beam2_fusion_signinv`)
Ipotesi: decorrelazione dello speckle più graduale -> |z| migliore sui passi piccoli.
Risultato (val): all* 2.75, |z| 1.64, |z|dt1 0.80, x 1.07, y 0.82 — identico a E1.
Curva: fit |z| in plateau da step ~1500 (1.58 -> 1.55), fit ~ val: niente overfitting, plateau.
Decisione: lo spessore non è il collo di bottiglia (lo tengo: più realistico). Sospetto nuovo: SmoothL1 con beta 1 mm è quadratica
sotto 1 mm -> quasi nessun gradiente per spostamenti nel piano < 1 mm (60% delle query). D4RT usa L1. -> E4.

**E3 — E2 + patch locale 9x9 a risoluzione nativa (D4RT)** (`runs/e3_beam2_fusion_patch9_signinv`)
Ipotesi: la query non ha identità d'aspetto fine -> x,y non appresi.
Risultato (val): all* 2.76, |z| 1.65, |z|dt1 0.81, x 1.07, y 0.83 — identico a E1/E2 (verificato: patch attiva, pesi addestrati).
Decisione: tre modifiche all'input (fusione, fascio, patch) -> stesso plateau. Il collo di bottiglia non è l'informazione in ingresso
ma loss/ottimizzazione. -> E4: uscita come spostamento + L1 (beta 0.05 mm), stessa config di E3.

**Diagnosi D1 — E3 valutato con frame alterati** (val, |z| / x / y; ferma 2.48 / 1.03 / 0.78)
| frame | \|z\| | x | y |
| --- | --- | --- | --- |
| veri | 1.64 | 1.06 | 0.82 |
| ordine temporale mescolato | 1.65 | 1.20 | 0.92 |
| da un altro sweep | 1.64 | 1.11 | 0.91 |
| tutti zero | 1.64 | 1.06 | 0.84 |
**Correzione importante:** il modello NON usa le immagini, nemmeno per |z|. Il guadagno |z| (1.64 vs 2.48) è un prior cieco:
E[|z| | Δt, profondità] imparato dalla query (con la loss invariante al segno predire la media di |z| batte "ferma" = 0).
Anche l'overfit su 1 sweep (0.3 mm) non prova l'uso delle immagini: (u,v,t_s,t_r) -> p si memorizza senza frame.
Aggiunta la metrica `blind_z_mag` (|z| medio per Δt dal train) come baseline corretta per |z|.
Prossimo: test di sanità S1 — solo traslazione laterale nel piano a velocità costante casuale per sweep (-2..2 mm/frame):
la velocità è osservabile solo dalle immagini (l'anatomia scorre). Se il modello non impara questo, il problema è di base.

**E4 — E3 + uscita come spostamento + loss quasi L1 (beta 0.05 mm)** (`runs/e4_residual_l1`)
Ipotesi: SmoothL1 beta 1 mm non dà gradiente sui piccoli spostamenti nel piano.
Risultato (val): stesso plateau di E1-E3 (|z| ~1.66, x ~1.09, y ~0.79). Decisione: la formulazione della loss non è il collo di bottiglia;
il modello ignora le immagini (D1). -> S1 test di sanità.

**S1 — sanità: solo traslazione laterale, velocità costante casuale per sweep (`data/sanity_lateral`)** (`runs/s1_lateral`)
Risultato: val x 2.03 vs ferma 2.00, fit x 2.43 vs 2.39 -> **fallisce**. Dati verificati: lo spostamento dell'immagine tra frame
(phase correlation, ~2 px/frame) coincide in segno ed entità con le pose.

**S2 — overfit di 2 sweep della stessa CT con velocità opposte** (`runs/s2_two_sweeps*`)
Le query dei due sweep sono indistinguibili: per abbassare la loss il modello DEVE usare le immagini.
Risultato: fit x 2.02 vs ferma 1.94 con lr 3e-4, 1e-4, 3e-5 -> **fallisce**, non memorizza nemmeno quale sweep sta vedendo.

**Test sintetici sul solo modello (video casuali, nessuna ecografia)**
| indizio nel video | target | risultato |
| --- | --- | --- |
| luminosità media = c | x = c·Δt | fallisce (atteso: GroupNorm dell'encoder cancella la luminosità globale) |
| larghezza di una banda chiara = c | x = c·Δt | fallisce (2.50 vs 2.43 predicendo 0) |
| larghezza banda = c | x = 5c (senza Δt) | **0.54 mm a step 600, poi collassa a 5.06 a step 900** |
Conclusione: il modello PUÒ leggere l'immagine, ma (1) il training è instabile (picchi di loss che cancellano quanto appreso,
visti anche in S2) e (2) il prodotto contenuto x Δt non viene appreso in 1200 step.
