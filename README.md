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
- VGG19 achieved the best overall performance with the highest accuracy, precision, recall, and F1-score. It also trained the fastest due to early stopping (epoch 18).
- MobileNetV2 and DenseNet121 showed similar strong performance, with MobileNetV2 offering better efficiency and DenseNet121 requiring more time due to its depth.
- EfficientNetB0 had slightly lower metrics, close to the CNN trained from scratch.
- The CNN from scratch performed reasonably well but was outperformed by all transfer learning models.
- Overall, transfer learning outperformed training from scratch, confirming its effectiveness for multi-class chest X-ray classification.

## Project Structure

├── cnn_classification.ipynb     # Jupyter Notebook with full code and analysis  
├── xray_classification_nn.pdf               # Complete written thesis  
└── README.md                      # Project description and setup  

## Contact
For questions or collaboration:
- Email: dadakidisgiorgos@gmail.com
- LinkedIn: [https://www.linkedin.com/in/dadakys](https://www.linkedin.com/in/giorgos-dadakidis/)
