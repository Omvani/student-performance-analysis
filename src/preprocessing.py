import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_data(path):
    return pd.read_csv(path)
def add_column(data):
    data['effort_score']=(data['study_hours']*0.4+data['assignment']*0.3+data['internal_marks']*0.3)
def split_target_features(data):
    X=data.drop("final_score",axis=1)
    y=data["final_score"]
    return X,y
def split_test_train(X,y,test_size=0.2,random_state=42):
    return train_test_split(X,y,test_size=test_size,random_state=random_state)
def scale_features(X_train,X_test):
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    return X_train_scaled,X_test_scaled
def preprocess_data(path):
    data=load_data(path)
    add_column(data)
    X,y=split_target_features(data)
    X_train,X_test,y_train,y_test=split_test_train(X,y)
    X_train_scaled,X_test_scaled=scale_features(X_train,X_test)
    return X_train_scaled,X_test_scaled,y_train,y_test