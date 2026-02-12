🚀 DB15 – Sleep Disorder Detection Using Deep Learning  
Based on ANN, Random Forest & XGBoost Optimization  

👥 Team Information  

Project Lead  
Beesabattuni Sarath Chandra — 23475A0510  
🔗 LinkedIn: https://www.linkedin.com/in/your-linkedin-profile  

Contribution:

Complete end-to-end project implementation  
Dataset selection and preprocessing  
Feature engineering and encoding  
Blood pressure feature transformation  
Model building (ANN, Random Forest, XGBoost)  
Hyperparameter tuning  
5-Fold Cross Validation implementation  
Model training, validation, and evaluation  
Comparative performance analysis  
Result interpretation and documentation  
GitHub repository setup  

Team Member  
Konakanchi Sai Manikanta — 22471A05O8  
🔗 LinkedIn: https://www.linkedin.com/in/your-linkedin-profile  

Contribution:

Literature survey assistance  
Dataset validation and understanding  
Model testing support  
Performance verification  
Documentation and presentation preparation  

---

📌 Abstract  

Sleep disorders such as Insomnia and Sleep Apnea significantly affect human health, cognitive performance, and overall quality of life. Traditional diagnosis methods are often manual, time-consuming, and prone to subjective errors. Early and accurate detection is crucial for effective treatment and monitoring.

This project presents a machine learning and deep learning framework for automated sleep disorder detection using the Sleep Health and Lifestyle dataset. The system compares three models: Artificial Neural Network (ANN), Random Forest Classifier, and XGBoost Classifier.

The dataset contains 400 records with 13 lifestyle and physiological attributes including sleep duration, stress level, BMI category, blood pressure, heart rate, and daily steps. Preprocessing techniques such as label encoding, blood pressure splitting, feature standardization, and train-test splitting were applied to ensure model stability.

Among the evaluated models, the ANN achieved the highest test accuracy of 94.17% with strong cross-validation performance (92.85% ± 1.23), demonstrating its effectiveness for structured healthcare data classification.

---

✨ Improvements and Adaptations  

Optimized ANN architecture for tabular healthcare data  
Comparative evaluation with ensemble models  
Feature engineering by splitting blood pressure into systolic and diastolic  
5-Fold Cross Validation for reliable performance estimation  
Balanced evaluation using precision, recall, F1-score  
Designed for scalability in healthcare screening systems  

---

🧠 About the Project  

What the Project Does  

Automatically classifies individuals into:

None (No Sleep Disorder)  
Sleep Apnea  
Insomnia  

Why It Is Useful  

Assists healthcare professionals in early screening  
Reduces manual diagnostic workload  
Provides scalable AI-based health monitoring  
Supports integration into wearable and telemedicine systems  
Improves accuracy and consistency in diagnosis  

---

🔄 System Workflow  

Dataset Collection  
↓  
Data Preprocessing  
↓  
Label Encoding  
↓  
Blood Pressure Splitting  
↓  
Feature Standardization  
↓  
Train-Test Split (70:30)  
↓  
Model Training (ANN / Random Forest / XGBoost)  
↓  
5-Fold Cross Validation  
↓  
Model Evaluation  
↓  
Sleep Disorder Prediction  

---

📂 Dataset Used  

Sleep Health and Lifestyle Dataset (Kaggle)  
🔗 https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset  

Dataset Details  

Total Records: 400  
Total Features: 13  
Target Classes: 3  

Classes:

None  
Sleep Apnea  
Insomnia  

Key Features:

Gender  
Age  
Occupation  
Sleep Duration  
Quality of Sleep  
Physical Activity Level  
Stress Level  
BMI Category  
Blood Pressure  
Heart Rate  
Daily Steps  

---

🛠 Technologies & Dependencies  

Python 3.x  
TensorFlow / Keras  
Scikit-learn  
XGBoost  
Pandas  
NumPy  
Matplotlib  
Seaborn  
Google Colab  

---

🔎 Data Preprocessing  

Label Encoding for categorical variables  
Blood Pressure split into:
- Systolic Blood Pressure  
- Diastolic Blood Pressure  

StandardScaler normalization  
Feature scaling (Mean = 0, Std = 1)  
Train-Test Split (70% Training, 30% Testing)  
5-Fold Cross Validation  

---

🧪 Model Training  

Component	Description  
ANN	Sequential Dense + Dropout Architecture  
Random Forest	Ensemble Decision Trees  
XGBoost	Gradient Boosting Framework  
Optimizer (ANN)	Adam  
Loss Function	Categorical Cross-Entropy  
Validation	5-Fold Cross Validation  
Platform	Google Colab (GPU Support)  

---

📊 Model Evaluation  

Metrics Used  

Accuracy  
Precision  
Recall  
F1-Score  
Confusion Matrix  

Performance Comparison  

Model	Test Accuracy	Cross-Val Accuracy  
ANN	94.17%	92.85% ± 1.23  
Random Forest	91.67%	90.42% ± 1.56  
XGBoost	92.50%	91.18% ± 1.34  

---

🏆 Results  

ANN achieved highest accuracy of 94.17%  
Strong cross-validation stability  
Balanced precision and recall across all classes  
Random Forest and XGBoost showed competitive performance  
Minor confusion observed between Sleep Apnea and Insomnia due to overlapping symptoms  

---

⚠ Limitations & Future Work  

Limitations  

Limited dataset size (400 samples)  
Structured data only (no EEG/PSG signals)  
Single dataset evaluation  

Future Enhancements  

Incorporate larger and diverse datasets  
Integrate EEG and PSG physiological signals  
Apply Explainable AI techniques (SHAP, LIME)  
Deploy in real-time healthcare systems  
Integration with wearable health monitoring devices  

---

🚀 Deployment  

Suitable for:

Healthcare screening systems  
Clinical decision support tools  
Telemedicine platforms  
Remote patient monitoring applications  

---

👨‍💻 Developed By  

Beesabattuni Sarath Chandra  
Project Lead & Developer  

Konakanchi Sai Manikanta  
Team Member  

---

🙏 Acknowledgments  

Kaggle for providing the Sleep Health and Lifestyle dataset  
TensorFlow/Keras for deep learning framework  
Scikit-learn & XGBoost libraries  
Google Colab for computational resources  
Research community for inspiration and guidance  

---

📧 Contact  

For collaborations or queries:

Beesabattuni Sarath Chandra  
LinkedIn: https://www.linkedin.com/in/your-linkedin-profile  

⭐ If you find this project useful, please consider giving it a star on GitHub!
