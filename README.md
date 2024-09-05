# Projet ML Ops Loan Default App


## Table of Contents

-   [Technologies Used](#technologies-used)
-   [Description](#description)
-   [Objectives](#objectives)
-   [Notebooks Overview](#notebooks-overview)
-   [Installation](#installation)
-   [Usage](#usage)
-   [Project Structure](#project-structure)
-   [Collaborators](#collaborators)
-   [License](#license)

---

## Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![LightFM](https://img.shields.io/badge/lightfm-%2300422e.svg?style=for-the-badge&logo=lightfm&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

---

## Description
The main objective is to build and app that estimates the probability of default for each client based on their characteristics. Then create a CI/CD pipeline to deploy it with AWS Elastic Container Service

### Objectives
The main objective of this project is build an image and a pipeline with all the necessary tools to achieve the deployment of a predictive app. The specific steps include:

1. **Data Preprocessing**: Filtering and preparing the dataset for analysis.
2. **Model Engineering**: Test at least two classification algorithms.
3. **Model Tracking**: Tracking the metrics and artifacts of the model with MLflow.
4. **App Deployment**: Create a CI/CD pipeline.

---

## Notebooks Overview

1. **app.py**:
   - This is the predictive python app.

2. **¨Projet_ML_Ops.ipynb**:
   - This notebook contains the data preprocessing, the model training, and the creation of the experiment with MLflow.
---

## Usage 

1. **Use the App** Just run the app.py
---
## Project structure
```sh
📦 mlops-project/
├── 📁.github/
│   ├── 📁workflows/
│       ├── aws.yaml
├── 📁templates/
│   ├── index.html
├── 📄.dockerignore
├── 📄.gitignore
├── 📄Dockerfile
├── 📄README.md
├── 🐍app.py
├── 🗃️model.pkl
├── 🐍Projet_ML_Ops.ipynb
├── 📄requirements.txt
└── 🐍test.py


---


## Collaborators

This project was developed by a collaborative team. Each member played a crucial role in the research, development, and analysis:

- **Mohamed Kallel**
- **Jean Christophe Rigoni**
- **Simon Pierre Rodner**
---



## License
This project is under the **CC BY-NC 4.0 License**. For more information, refer to the license file. <br/>
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
