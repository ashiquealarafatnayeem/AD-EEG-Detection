# Detecting Alzheimer's Disease from Resting-State EEG

**EEE 312 (January 2026), Digital Signal Processing I Laboratory, BUET · Section C2, Group 01**

A complete digital signal processing pipeline that turns a 19-channel, eyes-closed resting EEG recording into a probability of Alzheimer's disease (AD), tested only on people the model has never seen.

| Member | Student ID |
|---|---|
| Md. Ashique Al Arafat Nayeem | 2206190 |
| Arib Salehin | 2206186 |
| H M Hamim Al Muhit | 2206169 |
| Shahriar Nafis Rabbi | 2206184 |

## Main results

| Result | Value |
|---|---|
| AD vs healthy (CN), AUC, nested cross-validation | **0.811** [95 % CI 0.717–0.896] |
| Permutation test (200 label shuffles) | p = 0.005 |
| Balanced accuracy / sensitivity / specificity | 0.76 / 0.73 / 0.80 |
| Theta/alpha ratios only (7 features) | AUC 0.827 |
| Robustness criteria (fixed before the runs) | 5 of 5 met |
| FTD vs CN / AD vs FTD | AUC 0.728 / 0.578 |

## Pipeline

1. **Filtering:** IIR Butterworth band-pass 0.5–45 Hz (order 22, 11 second-order sections) and a 50 Hz notch, zero phase (forward–backward). An FIR Kaiser design (7253 taps) is included for comparison.
2. **Preprocessing:** crop, resample to 250 Hz, faulty-electrode repair, common-average reference, ICA blink removal, 4 s epochs, 150 µV rejection.
3. **Spectral estimation:** our own Welch estimator (Hann window, 4 s segments, 0.25 Hz resolution), validated against SciPy.
4. **Features:** 94 regional features (band powers, theta/alpha and slow/fast ratios, spectral shape, Hjorth parameters, alpha peak).
5. **Classification:** L2 logistic regression with nested, subject-wise cross-validation, bootstrap confidence intervals and a permutation test.

## Repository contents

```
src/            all code: core modules and the numbered analysis scripts (run in order)
figures/        every figure produced by the scripts
results/        small result tables (features, predictions, summaries)
literature.md
requirements.txt
```

Core modules: `filters.py` (filter design), `spectral.py` (Welch estimator), `preprocess.py` (cleaning), `features.py` (features), `ml.py` (models and validation), `eegio.py` (data access). Demonstration tool: `44_live_demo.py`.

## How to run

**1. Install** (Python 3.10 or newer):

```
python -m venv .venv
.venv\Scripts\Activate.ps1          (Windows)   or   source .venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
```

**2. Get the data.** The EEG data are public and are not stored in this repository. Download them from OpenNeuro into `data/raw/`:

- ds004504 (eyes-closed resting EEG): https://openneuro.org/datasets/ds004504
- ds006036 (eyes-open photic EEG, used for alpha reactivity): https://openneuro.org/datasets/ds006036

**3. Run the scripts in numerical order** from the project folder, for example `python src\12_batch_preprocess.py`. Each script prints a checkpoint line first; if it reports FAIL, fix the problem before continuing. The main result is printed by `27_classify.py`, the permutation test by `28_permutation_test.py`.

**4. Score one person (demonstration):**

```
python src\44_live_demo.py --prepare      (once, caches group spectra)
python src\44_live_demo.py --suggest      (lists confidently scored people)
python src\44_live_demo.py sub-041        (scores one person)
```

The model that scores a person is trained on everyone else, never on that person. The output is for research and teaching only, not a medical diagnosis.

## Data source and licence

Data: A. Miltiadous et al., "A dataset of scalp EEG recordings of Alzheimer's disease, frontotemporal dementia and healthy subjects from routine EEG," *Data*, 8(6):95, 2023 (OpenNeuro ds004504, CC0), and A. Ntetska et al., *Data*, 10(5):64, 2025 (OpenNeuro ds006036, CC0).

Note: participant numbers do not match between ds004504 and ds006036; see Section 5.3.3 of the report. Cross-dataset links in this project use EEG-based pairing.
