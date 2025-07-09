============================================================
PTBXL 1D CNN Model Training v13 with expanded evaluation
============================================================
Using device: cuda
Enabling CUDA error debugging...
Loading data from: ptbxl_cnn_ready_v13.csv
Dataset shape: (142962, 187)
Features shape: (142962, 186)
Labels shape: (142962,)

Raw class distribution:
  Class 0.0: 99322 samples (69.5%)
  Class 2.0: 15774 samples (11.0%)
  Class 3.0: 6402 samples (4.5%)
  Class 4.0: 3763 samples (2.6%)
  Class 5.0: 9714 samples (6.8%)
  Class 6.0: 7987 samples (5.6%)

Total training time: 0:06:59.088439
Model saved to: ptbxl_ecg_model_v13.pth

Evaluating model on test set...
Overall Test Accuracy: 89.07%

Classification Report:
                                              precision    recall  f1-score   support

         Sinus Rhythm (Normal) (was class 0)       0.93      0.94      0.94     19865
               T-wave abnormal (was class 2)       0.79      0.74      0.77      3155
           Atrial Fibrillation (was class 3)       0.78      0.74      0.76      1280
           1st degree AV block (was class 4)       0.86      0.85      0.86       753
Left anterior fascicular block (was class 5)       0.80      0.76      0.78      1943
             Other arrhythmias (was class 6)       0.83      0.81      0.82      1597

                                    accuracy                           0.89     28593
                                   macro avg       0.83      0.81      0.82     28593
                                weighted avg       0.89      0.89      0.89     28593


Per-Class Metrics:
------------------------------
Class                | Accuracy (%) | Sensitivity (%) | Specificity (%)
----------------------------------------------------------------------
Sinus Rhythm (Normal) (was class 0) | 90.93        | 94.40           | 83.03          
T-wave abnormal (was class 2) | 95.01        | 74.42           | 97.57          
Atrial Fibrillation (was class 3) | 97.88        | 73.67           | 99.02          
1st degree AV block (was class 4) | 99.25        | 85.13           | 99.63          
Left anterior fascicular block (was class 5) | 97.06        | 76.38           | 98.57          
Other arrhythmias (was class 6) | 98.01        | 81.34           | 98.99          
----------------------------------------------------------------------

Training history plot saved to: ptbxl_training_history_v13.png

Confusion matrix saved to: ptbxl_confusion_matrix_v13.png

Training completed successfully!
Final test accuracy: 89.07%

Summary of Per-Class Metrics:
------------------------------
Class                | Accuracy (%) | Sensitivity (%) | Specificity (%)
----------------------------------------------------------------------
Sinus Rhythm (Normal) (was class 0) | 90.93        | 94.40           | 83.03          
T-wave abnormal (was class 2) | 95.01        | 74.42           | 97.57          
Atrial Fibrillation (was class 3) | 97.88        | 73.67           | 99.02          
1st degree AV block (was class 4) | 99.25        | 85.13           | 99.63          
Left anterior fascicular block (was class 5) | 97.06        | 76.38           | 98.57          
Other arrhythmias (was class 6) | 98.01        | 81.34           | 98.99          
----------------------------------------------------------------------
Model saved as: ptbxl_ecg_model_v13.pth
============================================================