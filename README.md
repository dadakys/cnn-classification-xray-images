#  Chest X-Ray Classification Using CNNs and Transfer Learning

This project implements deep learning methods to classify chest X-ray (CXR) images into four categories: **Normal**, **COVID-19**, **Lung Opacity**, and **Viral Pneumonia**. It compares a CNN model trained from scratch with various transfer learning techniques, including MobileNetV2, VGG19, EfficientNetB0, and DenseNet121.

##  Dataset
- **COVID-19 Radiography Database**  
  Source: [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)  
  Over 21,000 labeled chest X-ray images, categorized into 4 classes.

##  Models Used
- CNN from scratch (baseline)
- MobileNetV2
- VGG19
- EfficientNetB0
- DenseNet121

All pretrained models were fine-tuned using 4-fold stratified cross-validation.

##  Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  
- Confusion Matrix  
- Training/Evaluation Time

##  Results Summary
- **VGG19** achieved the best performance across most metrics and was the fastest to converge.
- All transfer learning models outperformed the custom CNN baseline in accuracy and efficiency.
- Detailed per-class performance and comparative analysis are included in the report.

##  Contents
- `models/`: Saved `.h5` and `.pkl` model files  
- `notebooks/`: Training and evaluation notebooks  
- `plots/`: ROC curves, confusion matrices, accuracy/loss curves  
- `report/`: Final research document in DOCX and PDF  
- `results.csv`: Metric scores for all models

