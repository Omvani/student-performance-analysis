from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from preprocessing import preprocess_data
def train_test_evaluate():
    X_train,X_test,y_train,y_test=preprocess_data("D:/project-1/student-performance-analysis/DATA/STUDENT_DATASET.csv")
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print("Mean Squared Error=",mse)
    print("R2 Score=",r2)
train_test_evaluate()