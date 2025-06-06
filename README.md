# AI/ML Internship - Task 7: Support Vector Machines (SVM) - Breast Cancer Classification

## Objective
The objective of this task was to understand and implement Support Vector Machines (SVMs) for binary classification, focusing on linear and non-linear kernels, hyperparameter tuning, and robust performance evaluation using cross-validation.

## Dataset
The dataset used for this task is the [breast-cancer.csv](breast-cancer.csv) dataset, which contains features computed from digitized images of fine needle aspirate (FNA) of a breast mass, used to predict whether the mass is benign (B) or malignant (M).

## Tools and Libraries Used
* **Python**
* **Pandas:** For data loading and manipulation.
* **Scikit-learn:** For machine learning model implementation (SVC, train-test split, StandardScaler, LabelEncoder, GridSearchCV, cross_val_score) and evaluation metrics (accuracy_score, confusion_matrix, classification_report).
* **Matplotlib:** For creating static visualizations (decision boundary plots).
* **NumPy:** For numerical operations, especially for creating meshgrids for decision boundary visualization.
* **Seaborn:** For enhanced statistical graphics.

## SVM Classification Steps Performed:

### 1. Load and prepare a dataset for binary classification
* Loaded the `breast-cancer.csv` dataset.
* Identified features (all columns except 'id' and 'diagnosis') and the binary target ('diagnosis'). The 'id' column and any empty columns were dropped.
* Encoded the 'diagnosis' target variable ('B' for Benign, 'M' for Malignant) into numerical labels (0 and 1).
* **Scaled** all numerical features using `StandardScaler` to ensure optimal performance of the SVM algorithm, which is sensitive to feature magnitudes.
* Split the scaled data into training (70%) and testing (30%) sets, ensuring stratification to maintain the original class proportions.
* **Outcome:** The dataset was successfully prepared, scaled, and split, making it ready for SVM model training.

### 2. Train an SVM with linear and RBF kernel
* Initialized and trained two Support Vector Classifier (SVC) models:
    * One with a `linear` kernel.
    * One with a `Radial Basis Function (RBF)` kernel (using default C and gamma).
* Made predictions on the test set for both models.
* Evaluated their initial performance using accuracy scores and detailed classification reports.
* **Outcome:** Both linear and RBF kernel SVMs demonstrated high initial accuracy (around 96.49%), indicating the dataset's high separability and the SVMs' effectiveness.

### 3. Visualize decision boundary using 2D data
* To visualize the complex decision boundaries, a new subset of the data was created using two key features (`radius_mean` and `texture_mean`), which were also scaled and split.
* Separate SVM models (linear and RBF) were trained on this 2D data.
* A meshgrid was created, and predictions were made across this grid to plot the decision regions.
* Plots were generated showing the decision boundaries for both the linear and RBF kernels, with training and test data points overlaid.
* **Outcome:** The visualizations clearly demonstrated how the linear kernel creates a straight separation, while the RBF kernel creates a curved, non-linear boundary, illustrating the "kernel trick" for non-linear classification.

### 4. Tune hyperparameters like C and gamma
* Utilized `GridSearchCV` to systematically search for the optimal `C` and `gamma` hyperparameters for the RBF kernel SVM. The search explored a range of values for `C` (0.1, 1, 10, 100) and `gamma` (0.001, 0.01, 0.1, 1).
* `GridSearchCV` performed 5-fold cross-validation on the training data for each parameter combination.
* The best performing parameters were identified as `C=10` and `gamma=0.01`.
* The model with these optimal parameters was evaluated on the unseen test set.
* **Outcome:** Hyperparameter tuning led to a slightly improved test accuracy of `0.9766` and enhanced recall for the malignant class (`0.94`), demonstrating the value of optimizing model parameters for better performance.

### 5. Use cross-validation to evaluate performance
* The final, tuned RBF SVM model (`C=10`, `gamma=0.01`) was subjected to 10-fold cross-validation on the entire scaled dataset.
* This provided a more robust estimate of the model's generalization capabilities.
* The mean cross-validation accuracy and its standard deviation were calculated.
* **Outcome:** The mean cross-validation accuracy of `0.9789` with a low standard deviation (`0.0205`) confirmed the model's high and consistent performance, indicating excellent generalization ability for classifying breast cancer.

## Visualizations
The repository includes the following generated plots:
* `svm_linear_decision_boundary.png`: Decision boundary for the Linear SVM on `radius_mean` vs `texture_mean`.
* `svm_rbf_decision_boundary.png`: Decision boundary for the RBF SVM on `radius_mean` vs `texture_mean`.

## Conclusion
This task successfully demonstrated the implementation and evaluation of Support Vector Machines for binary classification. It covered essential aspects such as data preprocessing, feature scaling, training with different kernels, visualizing decision boundaries, hyperparameter tuning for optimization, and robust performance assessment using cross-validation. The SVM model proved to be highly accurate and reliable for the breast cancer dataset.
