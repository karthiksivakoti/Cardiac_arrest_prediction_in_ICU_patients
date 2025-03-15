# 🫀 Cardiac Arrest Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=Python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![BigQuery](https://img.shields.io/badge/BigQuery-%234285F4.svg?style=for-the-badge&logo=Google-Cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)](https://matplotlib.org/)

An advanced clinical decision support system for predicting cardiac arrest risk in hospitalized patients using ensemble machine learning and deep learning techniques. Built with TensorFlow, scikit-learn, and BigQuery on MIMIC-III clinical database.

<p align="center">
  <img src="feature_importance.png" alt="Deep Learning Feature Importance" width="600"/>
  <br>
  <em>Feature Importance for Cardiac Arrest Prediction</em>
</p>

## 🌟 Features

- **Real-time risk assessment** for hospitalized patients
- **Ensemble model approach** combining Random Forest and Deep Learning
- **Personalized risk scores** on a 0-100 scale
- **Interpretable predictions** with key contributing factors
- **Clinical recommendation engine** based on patient-specific risk factors
- **Multi-level risk stratification** from Very Low to Critical
- **Interactive visualization** of model performance and patient features

## 📊 Model Performance

Our hybrid prediction system achieves excellent performance metrics:

- **AUC-ROC**: 0.877
- **Accuracy**: 81% (Random Forest), 75% (Deep Learning)
- **Precision**: 83% (weighted average)
- **Recall**: 81% (weighted average)

<p align="center">
  <img src="roc_curve.png" alt="ROC Curve" width="400"/>
  <img src="learning_curves.png" alt="Learning Curves" width="400"/>
  <br>
  <em>Model Performance: ROC Curve and Learning Curves</em>
</p>

## 🛠️ Technology Stack

<table>
<tr>
<td width="50%" valign="top">

### Machine Learning & Data
- **Core ML**:
  - TensorFlow 2.x
  - Keras
  - scikit-learn
  - NumPy/Pandas
  
- **Deep Learning**:
  - Multi-layer neural networks
  - Batch normalization
  - Dropout regularization
  - Focal loss for imbalanced data
  
- **Data Sources**:
  - MIMIC-III clinical database
  - Google BigQuery
  - Custom ETL pipelines

</td>
<td width="50%" valign="top">

### Model & Analysis
- **Models**:
  - Random Forest (baseline)
  - Deep Neural Network
  - Weighted ensemble
  
- **Feature Engineering**:
  - Clinical domain-specific features
  - Vital sign derivatives
  - Medical history encoding
  
- **Visualization**:
  - Matplotlib
  - Seaborn
  - Feature importance analysis
  - Risk stratification charts

</td>
</tr>
</table>

## 🧬 Key Components

### Data Pipeline
- Extracts clinical data from MIMIC-III database
- Identifies cardiac arrest cases using ICD-9 codes
- Joins patient demographics, vital signs, and laboratory values
- Handles missing data with domain-appropriate strategies
- Normalizes and scales features for optimal model performance

### Random Forest Model (Baseline)
- Provides excellent interpretability and baseline performance
- Identifies key predictors including arrhythmia, glucose levels, and length of stay
- Achieves good balance of precision and recall
- Used as the primary model (70% weight) in the ensemble

### Deep Learning Model
- Multi-layer neural network with regularization techniques
- Leverages non-linear patterns in the data
- Incorporates batch normalization and dropout layers
- Uses focal loss to address class imbalance
- Complementary model (30% weight) in the ensemble

### Risk Calculator
- Generates intuitive 0-100 risk scores tied to probability
- Provides granular risk levels (Very Low to Critical)
- Identifies key contributing factors for individual predictions
- Delivers tailored clinical recommendations based on specific risk factors
- Adapts to available patient data (works with partial information)

## 📈 Clinical Impact

The cardiac arrest prediction system offers several key clinical benefits:

- **Early Warning System**: Identifies at-risk patients before deterioration
- **Resource Optimization**: Directs monitoring resources to highest-risk patients
- **Personalized Care**: Tailors recommendations to specific patient risk factors
- **Decision Support**: Provides evidence-based guidance for clinical teams
- **Risk Communication**: Facilitates clear communication of patient status

## 🏥 Sample Risk Profiles

The system provides detailed risk assessments for patients across the risk spectrum:

### Low Risk Patient
- **Risk Score**: 30/100
- **Risk Level**: Moderate
- **Key Factors**: Elevated glucose levels
- **Recommendations**: Increased monitoring frequency, consider telemetry, cardiology consultation

### Moderate Risk Patient
- **Risk Score**: 52/100
- **Risk Level**: High
- **Key Factors**: History of arrhythmia, elevated glucose levels
- **Recommendations**: Continuous cardiac monitoring, urgent cardiology consultation, electrolyte evaluation

### High Risk Patient
- **Risk Score**: 56/100
- **Risk Level**: High
- **Key Factors**: Arrhythmia, elevated glucose, low blood pressure
- **Recommendations**: ICU consideration, cardiology consultation, fluid management assessment

### Critical Risk Patient
- **Risk Score**: 63/100
- **Risk Level**: Very High
- **Key Factors**: Arrhythmia, elevated glucose, low SpO2, elevated respiratory rate
- **Recommendations**: ICU admission, continuous monitoring, urgent interventions, resuscitation preparation

## 🔍 Feature Importance

The predictive models identified several key factors associated with cardiac arrest risk:

### Top Predictors (Random Forest)
1. Arrhythmia history (0.466)
2. Length of stay (0.057)
3. Glucose levels (0.054)
4. Mortality indicator (0.053)
5. Creatinine levels (0.037)

### Top Predictors (Deep Learning)
1. Arrhythmia history (0.187)
2. Mortality indicator (0.015)
3. Bicarbonate levels (0.006)
4. Length of stay (0.004)
5. Myocardial infarction history (0.003)

## 🚀 Future Directions

Ongoing development focuses on:

- Integration with electronic health record systems
- Real-time monitoring and alert systems
- Mobile application for clinical team notifications
- Expanded recommendation engine with medication-specific guidance
- Transfer learning to adapt to different clinical environments
- Incorporating time-series data for dynamic risk assessment

## 📚 References

This project utilizes data from the MIMIC-III Clinical Database:

- Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III, a freely accessible critical care database. Scientific Data, 3, 160035.

## 🙏 Acknowledgments

- MIMIC-III database and PhysioNet for providing critical care data
- TensorFlow and scikit-learn communities
- Google Cloud BigQuery for powerful data processing capabilities

---

Made with ❤️ for improving patient outcomes and advancing predictive healthcare analytics.
