
# Implementing Differential Privacy in Machine Learning for Cancer Prediction
## Project Overview
This project, developed by Group 14 as part of CSI5195, focuses on implementing Differential Privacy (DP) in machine learning to enhance data security and fairness in cancer prediction. By applying DP techniques, we aim to protect sensitive patient information while maintaining the accuracy of a predictive model. The project uses a Logistic Regression model trained on a cancer dataset, with privacy-preserving methods such as Differentially Private Stochastic Gradient Descent (DP-SGD) and noise addition to sensitive features.

## Objectives
Protect patient privacy by applying Differential Privacy.
Train an accurate ML model to predict cancer diagnosis.
Compare model performance with and without DP.
Provide an interpretable prediction tool for end-users.

## Dataset
The dataset used is "The Cancer Prediction Dataset" from Kaggle, containing 1,500 records with the following features:
<li>Age: Patient's age (int)</li>
<li>Gender: Encoded (0 = Male, 1 = Female)</li>
<li>BMI: Body Mass Index (float)</li>
<li>Smoking: 0 (No), 1 (Yes)</li>
<li>GeneticRisk: Genetic predisposition (0 = Low, 1 = Medium, 2 = High)</li>
<li>PhysicalActivity: Weekly physical activity level (float)</li>
<li>AlcoholIntake: Alcohol consumption level (float)</li>
<li>CancerHistory: Family history of cancer (0 = No, 1 = Yes)</li>
<li>Diagnosis: Target variable (0 = No Cancer, 1 = Cancer)</li>


## Prerequisites
Python 3.8+
Required libraries (install via pip):

<code>pip install pandas numpy matplotlib seaborn sklearn torch opacus joblib</code>

## Installation
1. Clone the Repository:

<code>git clone 
cd CSI5195-Project-Group14</code>

2. Install Dependencies:

<code>pip install -r requirements.txt</code>

3. Download the Dataset:
<code>Download The_Cancer_data_1500_V2.csv from the Kaggle link.
Place it in the Data/ folder.</code>

## Usage
1. Exploratory Data Analysis (EDA)
Run the EDA script to analyze the dataset and generate visualizations:

<code>python eda.py</code>
Outputs:
EDA_Report.txt: Summary statistics and missing value counts.
Visualizations: data_visualization.png, eda_boxplots.png, eda_correlation_heatmap.png.

2. Baseline Logistic Regression Model
Train a standard Logistic Regression model without DP:
<code>python logistic_regression.py</code>

Outputs:
Model and scaler saved as logistic_model.pkl and scaler.pkl.
Accuracy plots and confusion matrix displayed.

3. Adding Noise (Local DP)
Apply Gaussian noise to sensitive features and save the noised dataset:
<code>python add_noise.py</code>

Outputs:
The_Cancer_data_1500_V2_Noised.csv: Dataset with noise added to BMI, PhysicalActivity, and AlcoholIntake.

4. Logistic Regression on Noised Data
Train a Logistic Regression model on the noised dataset:
<code>python logistic_regression_noised.py</code>

Outputs:
Updated logistic_model.pkl and scaler.pkl.
Performance metrics and visualizations.

5. DP-SGD Model Training
Train a Logistic Regression model with DP-SGD using PyTorch and Opacus:

<code>python dp_sgd_training.py</code>
Outputs:
dp_sgd_model.pth: Trained DP-SGD model.
scaler_SGD.pkl: Scaler for feature scaling.
Test accuracy printed.

6. Prediction Tool
Use the trained DP-SGD model to predict cancer diagnosis for new patient data:
<code>python predict_cancer.py</code>

Interactive: Enter patient details via prompts.
Outputs:
Prediction (Cancer Positive/Negative).
Explanation of top factors influencing the prediction.

                        

## Results


<li>Baseline Model: Achieves high accuracy but risks patient data exposure.</li>
<li>Noised Data Model: Slightly reduced accuracy due to noise, enhances local DP.</li>
<li>DP-SGD Model: Balances privacy (ε = 3.0) and accuracy, with clipped gradients and noise-added training. </li>

## Future Work

<li>Experiment with different ε values to optimize privacy-accuracy trade-off.</li>
<li>Implement additional DP techniques</li>
<li>Extend to other ML models </li>


## Team Members

<li>Ovais Azeem</li>
<li>Rachna Sunilkumar Deshpande</li>
<li>Sakshi Devendrabhai Patel</li>
<li>Sugan Subramanian</li>






