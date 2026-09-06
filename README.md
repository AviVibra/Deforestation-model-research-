# Deforestation-model-research-
# Can a 93% Satellite AI Really See Deforestation?

A benchmark-vs-reality study. I trained a transfer-learned ResNet50 to 93%
accuracy on EuroSAT, deployed it on real Sentinel-2 imagery of Rondônia,
Brazil (2018 vs 2024), and graded its deforestation detections against the
Hansen Global Forest Change record.

## Result

**The 93% benchmark model detected only 19% of real deforestation.**
- Precision 0.60, recall 0.19 (20 flagged, 63 real, 12 caught)
- Mean confidence collapsed 0.95 → 0.50 on arrival in the Amazon
- Error taxonomy: **100% of the 51 misses share one cause** — the domain
  gap stopped the model from recognizing tropical forest as Forest in 2018,
  so the Forest→not-Forest change rule could never fire
- All 8 false alarms were correctly labeled Forest and several show visible
  partial clearing — likely sub-threshold disagreements with Hansen, not errors

## Pipeline

EuroSAT (27k images) → transfer-learned ResNet50 (93% test, baseline CNN 63%)
→ Sentinel-2 cloud-filtered median composites via Google Earth Engine
(2018: 12 images, 2024: 28) → 17×17 grid of 640m patches classified both
years → Forest→not-Forest change rule → graded against Hansen loss (2019–24,
>10% threshold).

## Files

| File | What it is |
|---|---|
| `research2_paper.pdf` | Full write-up with figures |
| `notebook.ipynb` | All code: training, exports, tiling, grading, taxonomy |
| `predict.py` | Standalone prediction script with usage instructions |
| `eurosat_resnet50.pt` | Trained model weights |
| `amazon_2018.tif` / `amazon_2024.tif` | The Sentinel-2 composites |
| `hansen_lossyear.tif` | Hansen ground-truth export |
| `map18.npy` / `map24.npy` | The model's land-cover maps |

## How to use the model

Open `predict.py` — full instructions are written as comments at the top,
including where to put the files and how to predict on your own region.
Runs on a free Google Colab GPU.

## Honesty note

Real data throughout: Sentinel-2 imagery, the Hansen Global Forest Change
record, and one real region (Rondônia). The headline number (19% recall) is
a deliberately honest failure measurement — the study was designed to
quantify the benchmark-vs-reality gap, not to build a working detector.
AI assistance was used for debugging, verification, and language editing;
all experiments, decisions, and findings are my own.

## Related

Part 2 of a two-study arc on benchmark-vs-reality collapse.
Part 1 (tabular robustness study): [link to your first repo]
