# ⚡ Decentralized Smart Grid Stability Prediction

> Pattern Recognition & Machine Learning — Universitat Politècnica de Catalunya  
> Author · Mohammad Alikhani Najafabadi • mohammad.najafabadi@estudiantat.upc.edu

This repository contains code, data, and experiments for predicting stability in a **Decentral Smart Grid Control (DSGC)** system using machine-learning and deep-learning models (SVM, Gradient Boosting, and Artificial Neural Networks).  
Our best model (ANN) reaches **94 % accuracy** on a 60 k-sample dataset derived from 4-node DSGC simulations.

---

## 📚 Table of Contents
1. [Background](#background)
2. [DSGC Mathematical Model](#dsgc-mathematical-model)
3. [Dataset](#dataset)
4. [Exploratory Data Analysis](#exploratory-data-analysis)
5. [Methodology](#methodology)
6. [Results](#results)
7. [How to Run](#how-to-run)
8. [References](#references)

---

## 🧠 Background
Smart grids are evolving from unidirectional energy delivery to **bidirectional networks with “prosumers.”**  
Monitoring grid **frequency** is key: excess generation raises frequency; shortages lower it. The DSGC framework links **real-time pricing** to local frequency measurements, but the original differential-equation model relies on simplifying assumptions (“fixed-inputs” & “equality” issues).  
We replace that fragile analytical model with data-driven predictors.

---

## ⚙️ DSGC Mathematical Model

Core power-balance (producer / consumer $j$):

$$
p^{\text{source}} = \tfrac12 M_j \,\frac{d}{dt}(\delta'_j)^2
                 + \kappa_j \sum_{k=1}^{N} K^{\max}_{jk}\sin(\delta_k-\delta_j)
$$

Phase dynamics (**synchronous machine**; eq 3 in the paper):

$$
\frac{d^{2}\delta_j}{dt^{2}}
  = P_j
  - \alpha_j\frac{d\delta_j}{dt}
  + \sum_{k=1}^{N} K_{jk}\sin(\delta_k-\delta_j)
$$

With reaction-time delay $\tau_j$ and price elasticity $\gamma_j$ the full model becomes (eq 4):

$$
\frac{d^{2}\delta_j}{dt^{2}}
  = P_j
  - \alpha_j\dot{\delta}_j
  + \sum_{k}K_{jk}\sin(\theta_k-\theta_j)
  \;-\;
  \gamma_j\,\frac{d}{dt}\bigl[\theta_j(t-\tau_j)-\theta_j(t-\tau_j-T_j)\bigr].
$$

A negative maximum eigenvalue of the Jacobian ⇒ **stable** grid.

---

## 📊 Dataset

| Field(s) | Description | Range |
|----------|-------------|-------|
| `tau1`…`tau4` | Reaction times | 0.5 – 10 s |
| `p1`…`p4` | Nominal power (producer positive, consumers negative) | –2 … +6 |
| `g1`…`g4` | Price elasticity | 0.05 – 1.0 |
| `stabf` | **Target** — binary {`stable`, `unstable`} | 36 % / 64 % |

*Original 10 000 simulated runs → **6× augmentation** by exploiting star-grid symmetry ⇒ **60 000 samples** (54 k train / 6 k test).*

---

## 🔍 Exploratory Data Analysis

* Key class imbalance: **64 % unstable**.  
* Strong negative correlation ($-0.83$) between feature magnitude and stability.  
* No two features were colinear enough to remove.  

![Correlation matrix](./img/cov.png)

---

## 🛠️ Methodology

### 1 · Pre-processing
* **StandardScaler** fitted on training fold then applied everywhere (no leakage).  
* Stratified train / test split preserves class ratios.

### 2 · Models & Hyper-tuning

| Model | Key search grid | CV folds |
|-------|-----------------|----------|
| **SVM** | kernel ∈ {linear, poly, rbf}; C ∈ {0.1, 1, 10}; γ, degree | 5 |
| **XGBoost** | `n_estimators`, `max_depth`, `eta`, `subsample`, … | 5 |
| **ANN** | 12 → 288 → 288 → 24 → 12 → 1 (sigmoid) &nbsp;•&nbsp; Nadam • 50 epochs | 10 |

RandomizedGridSearch ⟶ best hyper-set per model.

---

## 🧪 Results

| Model | Best CV Acc (%) | Test Acc (%) |
|-------|-----------------|--------------|
| SVM (RBF kernel) | 94.0 | 93.9 |
| XGBoost | 93.8 | 93.6 |
| ANN | **94.6** | **94.4** |

*ANN confusion matrix*  
![ANN Confusion](./img/32.png)

---

## 📈 Sample Figures

| Caption | Image |
|---------|-------|
| 4-node star DSGC topology | ![Star grid](./img/1.png) |
| Correlation matrix | ![Corr](./img/cov.png) |
| Param-sweep: XGB depth × η | ![Heat](./img/21.png) |
| ANN learning curves | ![ANN curve](./img/31.png) |

---

## ▶️ How to Run

```bash
# clone
git clone https://github.com/your-username/dsgc-stability.git
cd dsgc-stability

# create env
conda env create -f environment.yml
conda activate dsgc

# reproduce ANN experiment
python src/train_ann.py --epochs 50 --batch 32

# evaluate all models
python src/evaluate_all.py --saved_models models/
