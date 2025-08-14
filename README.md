## Project Overview

This project is part of the [XJTLU SURF](https://www.xjtlu.edu.cn/en/study/surf) program, led by **Professor Filbert Juwono**, focusing on developing a neural network model for cuffless blood pressure estimation.  

We adopted the [S-MAMBA](https://github.com/wzhwzhwzh0921/S-D-Mamba/tree/main?tab=readme-ov-file) architecture for its outstanding performance in time-series modeling. Using photoplethysmography (PPG) signal segments as input, the model predicts systolic blood pressure (SBP) and diastolic blood pressure (DBP).  

This work builds upon the [S-D-Mamba codebase](https://github.com/wzhwzhwzh0921/S-D-Mamba), with significant modifications tailored to blood pressure prediction. We gratefully acknowledge [wzhwzhwzh0921](https://github.com/wzhwzhwzh0921) for making their work publicly available.  

The dataset is derived from the [MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/1.0/), using raw PPG and arterial blood pressure (ABP) waveforms for training.  
