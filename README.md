## Project Overview

This project is part of the [XJTLU SURF](https://www.xjtlu.edu.cn/en/study/surf) program, led by **Professor Filbert Juwono**, focusing on developing a neural network model for cuffless blood pressure estimation.  

We adopted the [S-MAMBA](https://github.com/wzhwzhwzh0921/S-D-Mamba/tree/main?tab=readme-ov-file) architecture for its outstanding performance in time-series modeling. Using photoplethysmography (PPG) signal segments as input, the model predicts systolic blood pressure (SBP) and diastolic blood pressure (DBP).  

This work builds upon the [S-D-Mamba codebase](https://github.com/wzhwzhwzh0921/S-D-Mamba), with significant modifications tailored to blood pressure prediction. We gratefully acknowledge [wzhwzhwzh0921](https://github.com/wzhwzhwzh0921) for making their work publicly available.  

The dataset is derived from the [MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/1.0/), using raw PPG and arterial blood pressure (ABP) waveforms for training.  

## How to Start
This project requires Linux system to run, therefore, in our case, we applied WSL subsystem. 
For the further detailed steps, please read the file [SETUP](https://github.com/Liberty666-666/CufflessBPNet-Development-of-Neural-Network-Model-for-Cuffless-Blood-Pressure-Estimation/blob/main/SETUP.docx), which contains 11 sections summarizing every crucial step. Section 1 and 2 focus on configuring the Linux subsystem, Section 3 and 4 focus on setting up the virtual environment, Section 5 focuses on requirement installations, Section 6 presents instructions for operating on VS code,  Section 7 was for validating the data code in the begining phase, which can be neglected and no longer has use for the project, Section 8 to 10 present the commands for running the project, Section 11 focuses on uploading the project to GitHub repository, which does not matter running the project.

## Theoretical Supports and Demonstrations 
Please read the file, [A Brief Demostration of BP Estimation Using MAMBA](https://github.com/Liberty666-666/CufflessBPNet-Development-of-Neural-Network-Model-for-Cuffless-Blood-Pressure-Estimation/blob/main/A%20Brief%20Demostration%20of%20BP%20Estimation%20Using%20MAMBA.docx), to have more information about the working principles of this project, as well as the testing results obtained by us. 
