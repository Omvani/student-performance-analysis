from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from preprocessing import preprocess_data
import matplotlib.pyplot as plt
def train_test_evaluate():
    X_train,X_test,y_train,y_test=preprocess_data("D:/project-1/student-performance-analysis/DATA/STUDENT_DATASET.csv")
    model=Pipeline([
        ('poly',PolynomialFeatures(degree=2)),
        ('lr',LinearRegression())
    ])
    # model=RandomForestRegressor(
    #     n_estimators=200,
    #     random_state=42
    # )
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print("Mean Squared Error=",mse)
    print("R2 Score=",r2)
    plt.figure(figsize=(7,5))
    plt.scatter(y_test,y_pred)
    plt.xlabel("Actual score")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.plot([y_test.min(),y_test.max()],
             [y_test.min(),y_test.max()])
    plt.show()
train_test_evaluate()