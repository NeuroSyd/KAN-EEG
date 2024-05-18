# KAN-EEG

The KAN-EEG leverages a novel AI paradigm based on the Kolmogorov-Arnold Network [1], demonstrating superior flexibility, robustness, and generalization across various out-of-sample seizure detection datasets. This study pioneers the replacement of traditional MLP in seizure detection systems.

References:

[1] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., ... & Tegmark, M. (2024). Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756.

To train the model, run TUH.py

For inferences, run EPILEPSIAE.py or RPA.py. 

Both can be modified depending on the dataset you want to use, as well number of channels, or if the signal is in time-domain or time-frequency domain. We evaluated the EEG signal as time-frequency domain through Short-Time Fourier Transform (STFT).
