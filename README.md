# SemiHastakshar

This repository is being reorganized for public release.

Code for **"SemiHastakshar: Generalizable Indic Handwritten OCR through
Semi-Supervised Learning"** (Lalitha Evani, Ajoy Mondal, Jawahar C. V. —
CVIT, IIIT Hyderabad), [ICVGIP 2025](https://doi.org/10.1145/3774521.3774605).
See the [project page](https://lalithaevani.github.io/SemiHastakshar-page/)
for the abstract, method overview, and results.

## Datasets

- **[Indic-HW-Wild](https://huggingface.co/datasets/evanilalitha/IIIT-INDIC-HW-WILD)**
  — the paper's own unlabeled corpus (~2.45M word images, nine Indic
  scripts, collected from the web). Published on Hugging Face.
- **[IIIT-INDIC-HW-WORDS](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words)**
  (Gongidi & Jawahar, ICDAR 2021) — the in-domain labeled benchmark used to
  train/evaluate the baseline.
- **[IIIT-Indic-HW-UC](https://cvit.iiit.ac.in/usodi/ucciohd.php)**
  ("Unconstrained Camera", 13 Indic languages) — the out-of-domain,
  real-world handwriting test set where SemiHastakshar's gains over the
  labeled-only baseline matter most.
