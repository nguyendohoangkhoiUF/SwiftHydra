# Introduction 
Despite a plethora of anomaly detection models developed over the years, their
ability to generalize to unseen anomalies remains an issue, particularly in critical
systems. This paper aims to address this challenge by introducing Swift Hydra, a
new framework for training an anomaly detection method based on generative AI
and reinforcement learning (RL). Through featuring an RL policy that operates on
the latent variables of a generative model, the framework synthesizes novel and
diverse anomaly samples that are capable of bypassing a detection model. These
generated synthetic samples are, in turn, used to augment the detection model,
further improving its ability to handle challenging anomalies. Swift Hydra also
incorporates Mamba models structured as a Mixture of Experts (MoE) to enable
scalable adaptation of the number of Mamba experts based on data complexity,
effectively capturing diverse feature distributions without increasing the model’s
inference time. Empirical evaluations on ADBench benchmark demonstrate that
Swift Hydra outperforms other state-of-the-art anomaly detection models while
maintaining a relatively short inference time. From these results, our research
highlights a new and auspicious paradigm of integrating RL and generative AI for
advancing anomaly detection.

# Instructions
- model.py contains the architecture of the model with the Mamba MoE and C-VAE.
- ADBenchDatsets contain a collection of datasets used to benchmark the model.
- In order to run a specific dataset, the path to it needs to be added into the dataloader for pretrain.py and then run that file to pretrain the Beta C-VAE and the detector.
- Then specify the path to the dataset in SwiftHydra.py and run it to see the accuracy based on the AUC-ROC metric for anomaly detection.
