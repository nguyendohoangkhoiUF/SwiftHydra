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
- ADBenchDatsets contain a collection of datasets used to benchmark the model (https://github.com/Minqi824/ADBench).
- In order to run a specific dataset, the path to it needs to be added into the dataloader for pretrain.py and then run that file to pretrain the Beta C-VAE and the detector.
- Then specify the path to the dataset in SwiftHydra.py and run it to see the accuracy based on the AUC-ROC metric for anomaly detection.

# Preliminary Imaging Results
Mapping generated synthetic anomalous points over the test/train sets gives us an idea that the generated points are close enough to those that can be possibly tested on the test set. For example referencing this plot of synthetic/real points for the yeast.npz dataset from ADBench:
![image](https://github.com/user-attachments/assets/6c1cf77d-7a7d-4b27-8ff5-6939979fccde)
This shows that the generated datapoints using the C-VAE are diverse enough to provide high accuracy and plot an intrinsic decision boundary.

Corresponding AUC-ROC metric results:
![image](https://github.com/user-attachments/assets/7906615c-5e33-4006-883d-41fd48922914)


# Comparison with other anomaly detection methods
ON DIFFUSION MODELING FOR ANOMALY DETECTION (https://openreview.net/pdf?id=lR3rk7ysXz) lists several anomaly detection models tested with the AUC-ROC metric. Swift Hydra manages to consistently outperform them. For eg. for the yeast dataset, the state of the art model can get an accuracy of 50.4 (CBLOF) compared to 74.26 (Swift Hydra)

