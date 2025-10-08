# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
import time

# %% [markdown]
# Data Exploration

# %%
# load data

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                'marital-status', 'occupation', 'relationship', 'race', 
                'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                'native-country', 'income']
df = pd.read_csv('Census Income Data Set/adult.data',names=column_names,na_values=' ?',skipinitialspace=True)

df.info()

# %%
df.describe()

# %%
df.head()

# %%
df_missing = (df=='?').sum()
df_missing

# %%
# Numeric attributes
df.hist(figsize=(10, 8))
plt.show()

# Classification attributes(some)
for col in df.select_dtypes(include='object').columns[:4]:
    df[col].value_counts().plot(kind='bar', title=col)
    plt.xticks(rotation=45)
    plt.show()

# %% [markdown]
# load data

# %%
def load_census_data():

    try:
        # load data Feature's name come from adult.name 
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
            'marital-status', 'occupation', 'relationship', 'race', 'sex', 
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        # read training data 
        train_data = pd.read_csv(
            'Census Income Data Set/adult.data', 
            names=column_names, 
            na_values=' ?', 
            skipinitialspace=True
        )
        
        # read testing data 
        test_data = pd.read_csv(
            'Census Income Data Set/adult.test', 
            names=column_names, 
            na_values=' ?', 
            skipinitialspace=True,
            skiprows=1  # skip the first line(title)
        )
        return train_data, test_data, column_names
    
    except FileNotFoundError as e:
        print(f"can't find the file: {e}")
        return None, None, None


# %% [markdown]
# data preprocess

# %%
def preprocess_data(train_data, test_data):
    
    # Create a copy of your data
    train_clean = train_data.copy()
    test_clean = test_data.copy()
    
    
    # Process the target variable (remove periods, unify format)
    train_clean['income'] = train_clean['income'].str.strip()
    test_clean['income'] = test_clean['income'].str.replace('.', '').str.strip()
    
    #Handling missing values ​​for categorical features (filling with mode)
    categorical_cols = ['workclass', 'occupation', 'native-country']
    
    for col in categorical_cols:
        if col in train_clean.columns:
            train_mode = train_clean[col].mode()[0]
            train_clean[col].fillna(train_mode, inplace=True)
            test_clean[col].fillna(train_mode, inplace=True)  # Use the majority
    
    
    #Feature Engineering
    # for age data,creat a group to devide
    def age_group(age):
        if age < 25:
            return 'Young'
        elif age < 45:
            return 'Adult'
        elif age < 65:
            return 'Senior'
        else:
            return 'Elderly'
    
    train_clean['age_group'] = train_clean['age'].apply(age_group)
    test_clean['age_group'] = test_clean['age'].apply(age_group)
    
    # for working time,too
    def hours_group(hours):
        if hours < 30:
            return 'Part-time'
        elif hours < 40:
            return 'Full-time'
        elif hours == 40:
            return 'Standard'
        else:
            return 'Overtime'
    
    train_clean['hours_group'] = train_clean['hours-per-week'].apply(hours_group)
    test_clean['hours_group'] = test_clean['hours-per-week'].apply(hours_group)
    
    # capital loss and gain
    train_clean['has_capital_gain'] = (train_clean['capital-gain'] > 0).astype(int)
    test_clean['has_capital_gain'] = (test_clean['capital-gain'] > 0).astype(int)
    
    train_clean['has_capital_loss'] = (train_clean['capital-loss'] > 0).astype(int)
    test_clean['has_capital_loss'] = (test_clean['capital-loss'] > 0).astype(int)
    
    
    return train_clean, test_clean

# encode and scale
def encode_and_scale(train_data, test_data):
    
    # final features
    categorical_features = [
        'workclass', 'education', 'marital-status', 'occupation', 
        'relationship', 'race', 'sex', 'native-country',
        'age_group', 'hours_group'
    ]
    
    numerical_features = [
        'age', 'fnlwgt', 'education-num', 'capital-gain', 
        'capital-loss', 'hours-per-week', 'has_capital_gain', 'has_capital_loss'
    ]
    
    # One-Hot encoding of categorical features
    train_encoded = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)
    test_encoded = pd.get_dummies(test_data, columns=categorical_features, drop_first=True)
    
    #make sure train data and test data have the same cols
    train_cols = set(train_encoded.columns)
    test_cols = set(test_encoded.columns)
    
    for col in train_cols - test_cols:
        if col != 'income':
            test_encoded[col] = 0
    
    for col in test_cols - train_cols:
        if col != 'income':
            train_encoded[col] = 0
    
    test_encoded = test_encoded[train_encoded.columns]
    
    
    #Separating features and target variables
    X_train = train_encoded.drop('income', axis=1)
    y_train = train_encoded['income']
    X_test = test_encoded.drop('income', axis=1)
    y_test = test_encoded['income']
    
    #Normalizing Numerical Features
    scaler = StandardScaler()
    
    # standardization
    numerical_cols_in_encoded = [col for col in numerical_features if col in X_train.columns]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols_in_encoded] = scaler.fit_transform(X_train[numerical_cols_in_encoded])
    X_test_scaled[numerical_cols_in_encoded] = scaler.transform(X_test[numerical_cols_in_encoded])
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def main():
    
    
    # load data
    train_data, test_data, column_names = load_census_data()
    
    if train_data is None:
        return
    
    # preprocess
    train_clean, test_clean = preprocess_data(train_data, test_data)
    

    
    # encoding and scale
    X_train, X_test, y_train, y_test, scaler = encode_and_scale(train_clean, test_clean)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    return X_train, X_test, y_train_encoded, y_test_encoded, scaler

X_train, X_test, y_train, y_test, scaler = main()

# %%
def basic_decision_tree(X_train, X_test, y_train, y_test):
    
    # 1. data load

    print("prepare data")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 2. model build
    print("\n-----training-----")
    start_time = time.time()
    
    # Fitting the decision tree with default hyperparameters, apart from max_depth which is 5 so that can plot and read the tree.
    dt_default = DecisionTreeClassifier(max_depth=5)
    dt_default.fit(X_train,y_train_encoded)

    training_time = time.time() - start_time
    
    print(f"Model training is completed, time-consuming: {training_time:.2f}s")
    
    # 3. model predict
    print("\n-----predicting-----")
    start_time = time.time()
    y_pred = dt_default.predict(X_test)
    y_pred_proba = dt_default.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    
    print(f"Prediction completed, time consuming: {prediction_time:.2f}s")
    
    return dt_default, le, y_pred, y_pred_proba, training_time, prediction_time

def evaluate_model(model, le, y_test, y_pred, y_pred_proba):
    
    print("\n-----evaluation-----")
    
    y_test_encoded = le.transform(y_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred)
    recall = recall_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)
    roc_auc = roc_auc_score(y_test_encoded, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    

    print(classification_report(y_test_encoded, y_pred, 
                              target_names=le.classes_))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

def visualize_results(model, le, X_test, y_test, y_pred, y_pred_proba, feature_names):
    
    y_test_encoded = le.transform(y_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. confusion_matrix
    cm = confusion_matrix(y_test_encoded, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0, 0])
    axes[0, 0].set_title('confusion matrix')
    axes[0, 0].set_xlabel('predict label')
    axes[0, 0].set_ylabel('true label')
    
    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC Curve (AUC = {roc_auc_score(y_test_encoded, y_pred_proba):.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(True)
    
    # 3. feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[1, 0].set_yticks(range(len(feature_importance)))
    axes[1, 0].set_yticklabels(feature_importance['feature'])
    axes[1, 0].set_xlabel('Importance')
    axes[1, 0].set_title('Top 10 Feature Importances')
    axes[1, 0].invert_yaxis()
    
    # 4. Performance Metrics Comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    values = [
        accuracy_score(y_test_encoded, y_pred),
        precision_score(y_test_encoded, y_pred),
        recall_score(y_test_encoded, y_pred),
        f1_score(y_test_encoded, y_pred)
    ]
    
    bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Model Performance Metrics')
    axes[1, 1].set_ylim(0, 1)
    
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def show_tree_structure(model, feature_names, class_names):
   
    print("\n-----decision tree-----")
    
    tree_depth = model.get_depth()
    tree_leaves = model.get_n_leaves()
    n_features = model.n_features_in_
    
    print(f"Tree depth: {tree_depth}")
    print(f"Number of leaf nodes: {tree_leaves}")
    print(f"Number of features used: {n_features}")
    
    # Plot decision tree
    plt.figure(figsize=(15, 8))
    plot_tree(model, 
              feature_names=feature_names,
              class_names=class_names,
              filled=True, 
              rounded=True,
              max_depth=5,  
              fontsize=10)
    plt.title('Decision Tree Structure', fontsize=14)
    plt.show()

def run_basic_decision_tree(X_train, X_test, y_train, y_test):
    
    feature_names = X_train.columns.tolist()
    
    dt_model, le, y_pred, y_pred_proba, train_time, pred_time = basic_decision_tree(
        X_train, X_test, y_train, y_test
    )
    
    eval_results = evaluate_model(dt_model, le, y_test, y_pred, y_pred_proba)
    
    feature_importance = visualize_results(
        dt_model, le, X_test, y_test, y_pred, y_pred_proba, feature_names
    )
    
    show_tree_structure(dt_model, feature_names, le.classes_)
    
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Prediction time: {pred_time:.2f} seconds")
    print(f"Test set accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test set F1-score: {eval_results['f1']:.4f}")
    print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
    print(f"\nTop 5 most important features:")
    print(feature_importance.head(5))
    return {
        'model': dt_model,
        'label_encoder': le,
        'predictions': y_pred,
        'prediction_proba': y_pred_proba,
        'evaluation': eval_results,
        'feature_importance': feature_importance
    }

basic_dt_results = run_basic_decision_tree(X_train, X_test, y_train, y_test)


# %%
def optimize_hyperparameters(X_train, y_train):
    param_grid = {
        'max_depth': range(1, 30),
        'min_samples_leaf': [1, 5, 10, 20, 50],
        'min_samples_split': [2, 5, 10, 20],
        'criterion': ['gini', 'entropy']
    }
    
    dtree = DecisionTreeClassifier(random_state=100)
    grid_search = GridSearchCV(
        dtree, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

# %%
def final_decision_tree(X_train, X_test, y_train, y_test,best_params):
    
    
    # 1. data load

    print("prepare data")
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # 2. model build
    print("\n-----training-----")
    start_time = time.time()
    
    final_model = DecisionTreeClassifier(
        criterion=best_params.get('criterion', 'gini'),
        max_depth=best_params['max_depth'],
        min_samples_leaf=best_params['min_samples_leaf'],
        min_samples_split=best_params['min_samples_split'],
        random_state=100
    )
    final_model.fit(X_train, y_train_encoded)
    
    training_time = time.time() - start_time
    
    print(f"Model training is completed, time-consuming: {training_time:.2f}s")
    
    # 3. model predict
    print("\n-----predicting-----")
    start_time = time.time()
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    prediction_time = time.time() - start_time
    
    print(f"Prediction completed, time consuming: {prediction_time:.2f}s")
    
    return final_model, le, y_pred, y_pred_proba, training_time, prediction_time

# %%
def run_final_decision_tree(X_train, X_test, y_train, y_test,best_params=None):
    
    feature_names = X_train.columns.tolist()

    if best_params is None:
        _, best_params = optimize_hyperparameters(X_train, y_train)
    
    dt_model, le, y_pred, y_pred_proba, train_time, pred_time = final_decision_tree(
        X_train, X_test, y_train, y_test,best_params
    )
    
    eval_results = evaluate_model(dt_model, le, y_test, y_pred, y_pred_proba)
    
    feature_importance = visualize_results(
        dt_model, le, X_test, y_test, y_pred, y_pred_proba, feature_names
    )
    
    #show_tree_structure(dt_model, feature_names, le.classes_)
    
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Prediction time: {pred_time:.2f} seconds")
    print(f"Test set accuracy: {eval_results['accuracy']:.4f}")
    print(f"Test set F1-score: {eval_results['f1']:.4f}")
    print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
    print(f"\nTop 5 most important features:")
    print(feature_importance.head(5))
    return {
        'model': dt_model,
        'label_encoder': le,
        'predictions': y_pred,
        'prediction_proba': y_pred_proba,
        'evaluation': eval_results,
        'feature_importance': feature_importance,
        'best_params': best_params
    }

final_dt_results = run_final_decision_tree(X_train, X_test, y_train, y_test)


