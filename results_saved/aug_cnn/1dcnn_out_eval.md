(ecg) jyoon@ROG-G17:~/repos1/ecg-augdata/cnn/cnn_aug_g1$ python 1dcnn_aug_g1eval.py
============================================================
PTBXL 1D CNN Model Training v13 Augmented Data
============================================================
Using device: cuda
Enabling CUDA error debugging...
Loading data from: cnn_ready_aug_g1.csv
Dataset shape: (143683, 187)
Features shape: (143683, 186)
Labels shape: (143683,)

Raw class distribution:
  Class 0.0: 99814 samples (69.5%)
  Class 2.0: 16214 samples (11.3%)
  Class 3.0: 6314 samples (4.4%)
  Class 4.0: 3662 samples (2.5%)
  Class 5.0: 9674 samples (6.7%)
  Class 6.0: 8005 samples (5.6%)

Label range: 0 to 6
WARNING: Labels are not continuous sequence. Remapping...
Found labels: [np.int64(0), np.int64(2), np.int64(3), np.int64(4), np.int64(5), np.int64(6)]
Expected: [0, 1, 2, 3, 4, 5]
Label mapping: {np.int64(0): 0, np.int64(2): 1, np.int64(3): 2, np.int64(4): 3, np.int64(5): 4, np.int64(6): 5}
After remapping - Label range: 0 to 5
Remapped class names: ['Sinus Rhythm (Normal) (was class 0)', 'T-wave abnormal (was class 2)', 'Atrial Fibrillation (was class 3)', '1st degree AV block (was class 4)', 'Left anterior fascicular block (was class 5)', 'Other arrhythmias (was class 6)']

Final class distribution:
  Class 0: 99814 samples (69.5%)
  Class 1: 16214 samples (11.3%)
  Class 2: 6314 samples (4.4%)
  Class 3: 3662 samples (2.5%)
  Class 4: 9674 samples (6.7%)
  Class 5: 8005 samples (5.6%)
✓ Label validation passed. Expected classes: 0-5
Using remapped class names: ['Sinus Rhythm (Normal) (was class 0)', 'T-wave abnormal (was class 2)', 'Atrial Fibrillation (was class 3)', '1st degree AV block (was class 4)', 'Left anterior fascicular block (was class 5)', 'Other arrhythmias (was class 6)']

Splitting data: 80% train, 20% test
Training set: 114946 samples
Test set: 28737 samples

Training set class distribution:
  Class 0: 79851 samples (69.5%)
  Class 1: 12971 samples (11.3%)
  Class 2: 5051 samples (4.4%)
  Class 3: 2930 samples (2.5%)
  Class 4: 7739 samples (6.7%)
  Class 5: 6404 samples (5.6%)

Preparing PyTorch data with batch size: 64
Training tensor shape: torch.Size([114946, 1, 186])
Test tensor shape: torch.Size([28737, 1, 186])
Created DataLoaders: 1797 training batches, 450 test batches

Initializing model with 6 classes

Model architecture:
PTBXL_ECGNet(
  (conv1): Conv1d(1, 128, kernel_size=(80,), stride=(4,), padding=(38,))
  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool1): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (conv2): Conv1d(128, 128, kernel_size=(3,), stride=(1,), padding=(1,))
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool2): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (conv3): Conv1d(128, 256, kernel_size=(3,), stride=(1,), padding=(1,))
  (bn3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool3): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (conv4): Conv1d(256, 512, kernel_size=(3,), stride=(1,), padding=(1,))
  (bn4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (maxpool4): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (avgpool): AdaptiveAvgPool1d(output_size=1)
  (fc): Linear(in_features=512, out_features=6, bias=True)
  (log_softmax): LogSoftmax(dim=1)
)

Total parameters: 557,062
Trainable parameters: 557,062

Training model for 20 epochs on cuda
Learning rate: 0.001
Epoch [1/20], Train Loss: 0.8707, Train Acc: 71.32%, Test Loss: 0.8156, Test Acc: 72.21%
Epoch [2/20], Train Loss: 0.7484, Train Acc: 74.19%, Test Loss: 0.7226, Test Acc: 74.87%
Epoch [3/20], Train Loss: 0.6785, Train Acc: 76.15%, Test Loss: 0.6864, Test Acc: 76.50%
Epoch [4/20], Train Loss: 0.6115, Train Acc: 78.23%, Test Loss: 0.6170, Test Acc: 77.98%
Epoch [5/20], Train Loss: 0.5406, Train Acc: 80.59%, Test Loss: 0.5862, Test Acc: 79.93%
Epoch [6/20], Train Loss: 0.4862, Train Acc: 82.49%, Test Loss: 0.5445, Test Acc: 80.72%
Epoch [7/20], Train Loss: 0.4301, Train Acc: 84.33%, Test Loss: 0.5369, Test Acc: 81.63%
Epoch [8/20], Train Loss: 0.3862, Train Acc: 85.92%, Test Loss: 0.5244, Test Acc: 82.13%
Epoch [9/20], Train Loss: 0.3460, Train Acc: 87.49%, Test Loss: 0.5318, Test Acc: 81.90%
Epoch [10/20], Train Loss: 0.3073, Train Acc: 88.71%, Test Loss: 0.5265, Test Acc: 82.77%
Epoch [11/20], Train Loss: 0.2822, Train Acc: 89.64%, Test Loss: 0.5259, Test Acc: 83.28%
Epoch [12/20], Train Loss: 0.2454, Train Acc: 90.97%, Test Loss: 0.5140, Test Acc: 84.03%
Epoch [13/20], Train Loss: 0.2258, Train Acc: 91.70%, Test Loss: 0.5353, Test Acc: 83.89%
Epoch [14/20], Train Loss: 0.2065, Train Acc: 92.29%, Test Loss: 0.5487, Test Acc: 84.18%
Epoch [15/20], Train Loss: 0.1886, Train Acc: 93.09%, Test Loss: 0.5715, Test Acc: 83.51%
Epoch [16/20], Train Loss: 0.2024, Train Acc: 92.54%, Test Loss: 0.5448, Test Acc: 84.66%
Epoch [17/20], Train Loss: 0.1540, Train Acc: 94.30%, Test Loss: 0.5627, Test Acc: 84.86%
Epoch [18/20], Train Loss: 0.1525, Train Acc: 94.39%, Test Loss: 0.5776, Test Acc: 84.18%
Epoch [19/20], Train Loss: 0.1458, Train Acc: 94.68%, Test Loss: 0.6128, Test Acc: 84.11%
Epoch [20/20], Train Loss: 0.1338, Train Acc: 95.08%, Test Loss: 0.6032, Test Acc: 84.91%

Total training time: 0:06:53.215560
Model saved to: ptbxl_ecg_model_v13.pth

Evaluating model on test set...
Final Test Accuracy: 84.91%

==================================================
Per-Class Sensitivity and Specificity
==================================================
Class                               Sensitivity (Recall)      Specificity
--------------------------------------------------------------------------------
(0) Sinus Rhythm (Normal) (was class 0) 0.9316                    0.7471
(1) T-wave abnormal (was class 2)   0.5902                    0.9723
(2) Atrial Fibrillation (was class 3) 0.6469                    0.9855
(3) 1st degree AV block (was class 4) 0.7309                    0.9961
(4) Left anterior fascicular block (was class 5) 0.6889                    0.9817
(5) Other arrhythmias (was class 6) 0.7527                    0.9849
--------------------------------------------------------------------------------
OVERALL (Macro Avg)                 0.7235                    0.9446
==================================================


Classification Report (Sensitivity is 'recall'):
                                              precision    recall  f1-score   support

         Sinus Rhythm (Normal) (was class 0)       0.89      0.93      0.91     19963
               T-wave abnormal (was class 2)       0.73      0.59      0.65      3243
           Atrial Fibrillation (was class 3)       0.67      0.65      0.66      1263
           1st degree AV block (was class 4)       0.83      0.73      0.78       732
Left anterior fascicular block (was class 5)       0.73      0.69      0.71      1935
             Other arrhythmias (was class 6)       0.75      0.75      0.75      1601

                                    accuracy                           0.85     28737
                                   macro avg       0.77      0.72      0.74     28737
                                weighted avg       0.84      0.85      0.85     28737

/home/jyoon/repos1/ecg-augdata/cnn/cnn_aug_g1/1dcnn_aug_g1eval.py:501: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
Training history plot saved to: ptbxl_training_history_v13.png
/home/jyoon/repos1/ecg-augdata/cnn/cnn_aug_g1/1dcnn_aug_g1eval.py:520: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
Confusion matrix saved to: ptbxl_confusion_matrix_v13.png

============================================================
Training completed successfully!
Final test accuracy: 84.91%
Model saved as: ptbxl_ecg_model_v13.pth
============================================================