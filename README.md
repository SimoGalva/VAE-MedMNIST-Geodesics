VAE Latent Geometry on ChestMNIST (medMNIST Benchmark)

This project explores the latent geometry induced by a Variational Autoencoder trained on the ChestMNIST dataset from medMNIST (public benchmark – non-personal, non-identifiable data).

The aim of this experiment was research curiosity driven: after reading “The Riemannian Geometry of Deep Generative Models” (Shao, Kumar, Fletcher, Thomas 2017) I wanted to empirically test if geodesics computed on the latent manifold of a trained VAE differ meaningfully from simple straight line interpolations.

The result: in this small experimental setup, geodesics and straight lines in latent space have almost identical length → suggesting the induced data manifold is almost flat in this configuration.

This project is purely representational / mathematical exploration of generative model geometry — not clinical or diagnostic.


VAE-MedMNIST-Geodesics/
  LICENSE
  README.md
  .gitignore
  data/            ← empty placeholder or download instructions
  notebooks/
     01_train_vae.ipynb
     02_geodesic_analysis.ipynb
  src/
     vae.py
     geodesic.py
     utils.py
  results/
     figs/          ← PNGs of sample images, interpolation curves, geodesic comparison
  doc/
     report.pdf     ← optional compiled version of your Italian report
