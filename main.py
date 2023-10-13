# Import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Models from Scikit-learn
from sklearn.linear_model import LogisticRegression

# Model evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay, roc_curve, auc


class TelcoCustomerChurnLR:
    """
    LogisticRegression as a binary classification model to predict whether a customer will churn or not based on various
    features in the dateset. This program uses WA_Fn-UseC_-Telco-Customer-Churn.csv to train this LogisticRegression
    model and evaluates its performance on the dataset.
    """

    def __init__(self):
        # Read data from a csv file
        self.data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    def analysis(self):
        matplotlib.use('TkAgg')

        # Preprocessing of data

        # Check missing data
        # self.data.isna().sum()

        # This will turn all of the string value into integer values
        for label, content in self.data.items():
            if pd.api.types.is_string_dtype(content):
                self.data[label] = content.astype("category").cat.codes

        # Split data into X and y
        x = self.data.drop(["Churn", "customerID"], axis=1)
        y = self.data["Churn"]

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        # True LogisticRegression
        np.random.seed(42)

        # Different hyper parameter for our LogisticRegression model
        log_reg_grid = {"C": np.logspace(-4, 4, 30), "solver": ["liblinear"]}

        # Setup grid hyper parameter search for LogisticRegression
        gs_log_reg = GridSearchCV(LogisticRegression(),
                                  param_grid=log_reg_grid,
                                  cv=5,
                                  verbose=True)

        # Fit grid hyper parameter search model
        gs_log_reg.fit(x_train, y_train)

        # Evaluating our tuned machine learning classifier, beyond accuracy
        # ROC curve and AUC score, confusion matrix, classification report, precision, recall, f1-score

        # Make prediction with tuned model
        y_preds = gs_log_reg.predict(x_test)

        # Plot ROC curve and calculation and calculate AUC metric
        fpr, tpr, thresholds = roc_curve(y_test, y_preds)
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="estimator")
        display.plot()
        plt.show()

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_preds)
        print("confusion_matrix:", conf_matrix)

        sns.set(font_scale=1)
        self.plot_conf_mat(conf_matrix)
        print(classification_report(y_test, y_preds))

        # Check best hyper parameters
        best_params_dict = gs_log_reg.best_params_
        print("best parameters:", best_params_dict)

        # Create a new classification with best parameters
        clf = LogisticRegression(C=best_params_dict["C"], solver=best_params_dict["solver"])

        # Cross_validated accuracy
        cv_acc = cross_val_score(clf, x, y, cv=5, scoring="accuracy")
        cv_acc = np.mean(cv_acc)

        # Cross_validated precision
        cv_precision = cross_val_score(clf, x, y, cv=5, scoring="precision")
        cv_precision = np.mean(cv_precision)

        # Cross_validation recall
        cv_recall = cross_val_score(clf, x, y, cv=5, scoring="recall")
        cv_recall = np.mean(cv_recall)

        # Cross-validated f1-score
        cv_f1 = cross_val_score(clf, x, y, cv=5, scoring="f1")
        cv_f1 = np.mean(cv_f1)

        # Validation cross-validation metrics
        cv_metrics = pd.DataFrame({"Accuracy": [cv_acc],
                                   "Precision": [cv_precision],
                                   "Recall": [cv_recall],
                                   "F1": [cv_f1]})

        print("cross-validation metrics:", cv_metrics)

        # Feature importance
        # Fit an instance of LogisticRegression
        clf = LogisticRegression(C=best_params_dict["C"], solver=best_params_dict["solver"])
        clf.fit(x_train, y_train)

        # Check coef_
        print(clf.coef_)

        # Match coef's of features to columns
        feature_dict = dict(zip(self.data.columns, list(clf.coef_[0])))

        # Visualize feature importance
        feature_df = pd.DataFrame(feature_dict, index=[0])
        feature_df.T.plot.bar(title="Feature Importance")
        plt.show()

    @staticmethod
    def plot_conf_mat(data):
        """
        Plots and nice looking confusion matrix using Seaborn's heatmap()
        """
        # fig, ax = plt.subplots(figsize=(3, 3))
        ax = sns.heatmap(data, annot=True, cbar=False)

        plt.xlabel("True label")
        plt.ylabel("Predicted label")

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()


telco_customer_churn = TelcoCustomerChurnLR()
telco_customer_churn.analysis()
