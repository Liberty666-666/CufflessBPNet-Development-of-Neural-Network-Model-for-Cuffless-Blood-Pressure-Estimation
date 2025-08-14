## Project Overview
This project is part of the [XJTLU SURF](https://www.xjtlu.edu.cn/en/study/surf) program, led by **Professor Filbert Juwono**, focusing on developing a neural network model for cuffless blood pressure estimation.

We adopted the [S-MAMBA](https://github.com/wzhwzhwzh0921/S-D-Mamba/tree/main?tab=readme-ov-file) architecture for its excellent performance in time-series modeling. Using photoplethysmography (PPG) signal segments as input, the model predicts systolic blood pressure (SBP) and diastolic blood pressure (DBP).

This project builds upon the [S-D-Mamba codebase](https://github.com/wzhwzhwzh0921/S-D-Mamba), with modifications tailored for blood pressure prediction. We sincerely thank [wzhwzhwzh0921](https://github.com/wzhwzhwzh0921) for making their work publicly available.

The dataset is derived from the [MIMIC-III Waveform Database](https://physionet.org/content/mimic3wdb/1.0/), using raw PPG and arterial blood pressure (ABP) waveforms for training.

## Features
- Predicts **systolic and diastolic blood pressure** from PPG segments
- Implements **S-MAMBA** for accurate time-series analysis
- Modular code for easy adaptation to new datasets
- Ready for training and inference on Linux (WSL supported)
- Fully documented with step-by-step setup instructions

## How to Start
This project requires a **Linux environment**. We applied the **Windows Subsystem for Linux (WSL)** in our case.

For detailed setup instructions, see the [SETUP](https://github.com/Liberty666-666/CufflessBPNet-Development-of-Neural-Network-Model-for-Cuffless-Blood-Pressure-Estimation/blob/main/SETUP.docx) file, which contains 11 sections:
- **Section 1–2**: Configure the Linux subsystem
- **Section 3–4**: Set up the virtual environment
- **Section 5**: Install required dependencies
- **Section 6**: Operate in VS Code
- **Section 7**: Early-phase data validation (obsolete; can be skipped)
- **Section 8–10**: Run the project using the provided commands
- **Section 11**: Upload the project to GitHub (not required to run the project)

## Theoretical Support and Demonstrations
For details on the working principles and testing results, see [A Brief Demonstration of BP Estimation Using MAMBA](https://github.com/Liberty666-666/CufflessBPNet-Development-of-Neural-Network-Model-for-Cuffless-Blood-Pressure-Estimation/blob/main/A%20Brief%20Demostration%20of%20BP%20Estimation%20Using%20MAMBA.docx).
