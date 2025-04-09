# Brain Tumar Classifier

Brain tumors are among the most aggressive diseases affecting both children and adults, accounting for 85–90% of all primary CNS tumors. Annually, around 11,700 people are diagnosed, with a 5-year survival rate of only ~34% for men and 36% for women. They are classified into types such as benign, malignant, and pituitary tumors. Accurate diagnosis and early detection are crucial, typically done via MRI scans, which generate large volumes of complex image data that are manually examined by radiologists.

Automated detection using Machine Learning (ML) and Deep Learning (DL) techniques has shown higher accuracy than manual methods. Using models like Convolutional Neural Networks (CNN), Artificial Neural Networks (ANN), and Transfer Learning (TL) can significantly assist in accurate brain tumor classification and support medical professionals worldwide.


## Objective
The aim of this project is to present a new CNN architecture for brain tumor classification of three primary brain tumors gliomas, meningiomas, and pituitary and also differentiate no-tumor.
## Methodology

### Dataset

Picked the Dataset from Kaggle :
[Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

This dataset contains 7023 images of human brain MRI scans including Training and Testing data which are classified into 4 classes : glioma , meningioma , pituitary , no-tumor.


### Data Preprocessing

- Images were resized to 224 (best for VGG16) and converted into NumPy arrays for model input.
- Data was shuffled to ensure equal distribution across classes.
- The dataset was split into training and testing sets using train_test_split.
- Labels were encoded using LabelEncoder and converted to categorical form.

### Data Augmentation (why not performed)
Attempted to use ImageDataGenerator to artificially increase dataset size but
Augmentation was limited due to the sensitive nature of brain scan images.

## Transfer Learning Setup

Transfer learning is implemented by using the convolutional base of a pre-trained VGG16 model to extract features from images, while replacing its original classification layers with new ones tailored to your specific task. Only the new layers are trained, allowing the model to adapt quickly using the already learned features.

In this model i have used VGG16 and the include_top parameter is set to False so that the network doesn't include the top layer/ output layer from the pre-built model which allows us to add our own output layer depending upon our use case!

- Loaded VGG16 (pre-trained on ImageNet) without its top layers.

- Used it as a feature extractor (convolutional base) with frozen weights.

- Added custom dense layers on top of the VGG16 base:

- GlobalAveragePooling2D, Dense, Dropout, and final softmax output.

- Used Sequential model architecture for stacking layers.

## Evaluation
Evaluated the model's performance using:

- Accuracy

- Classification Report (precision, recall, F1-score)

- Plotted a confusion matrix heatmap for visual understanding.


## Conclusion
At the end Validation Accuracy comes to be 94.56 % (best) with training accuracy of 97.10 %

From the classification_report and confusion matrix we can find that F1 score of Glioma tumor and Meningioma tumar is low from rest that is because the tumor images of both of these looks similar and thus model is keeping hard to classify between these two.

Good part of this model is that it classifies between tumar and non tumar with a high accuracy and F1 score!


