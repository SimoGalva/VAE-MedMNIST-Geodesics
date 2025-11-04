# VAE Latent Geometry on ChestMNIST (medMNIST Benchmark)

This project explores the latent geometry induced by a Variational Autoencoder trained on the **ChestMNIST** dataset from **medMNIST** (public benchmark – non-personal, non-identifiable data).

Motivation: after reading *“The Riemannian Geometry of Deep Generative Models” (Shao, Kumar, Fletcher, Thomas 2017)* I wanted to empirically verify whether geodesics computed on the VAE latent manifold differ meaningfully from simple straight line interpolations.

**Outcome**: in this experimental setup, geodesics and straight latent segments have extremely similar length → suggesting the induced data manifold is nearly flat locally.

> This is purely a representational / theoretical exploration of model geometry — **not clinical or diagnostic**.

---

## 📦 Dataset

- Dataset: **ChestMNIST** (CC BY 4.0)  
  https://github.com/MedMNIST/MedMNIST/

- Grayscale 1×28×28 images  
- Rounded pixel intensities to 3 decimals for computational simplicity

---

## 🔧 Installation

```bash
pip install torch torchvision medmnist numpy
# optional
pip install matplotlib tqdm
```

---

## 📁 Repository Structure

```
src/               # VAE + Geodesic modules (clean python)
notebooks/         # training + analysis experiments
results/figs/      # selected qualitative outputs included in repo
doc/               # optional PDF written report
```

---

## ▶️ Usage

### 1) Train VAE
Notebook: `notebooks/01_train_vae.ipynb`

- downloads ChestMNIST  
- trains the model  
- saves `VAE.pt`

### 2) Geodesic Experiment
Notebook: `notebooks/02_geodesic_analysis.ipynb`

- loads trained VAE  
- samples two points in data space  
- encodes → latent  
- computes straight path vs geodesic optimization  
- decodes and compares

---

## 📌 Results (qualitative)

In this setup, geodesic paths and straight latent interpolations decode to near-identical visual output — supporting the “near-flat latent manifold” behavior reported in prior literature.

---

## 📄 License

MIT — see `LICENSE`.

---

## Reference Inspiration

> Shao, Hang and Kumar, Abhishek and Fletcher, P. Thomas  
> *The Riemannian Geometry of Deep Generative Models* (arXiv:1711.08014)
