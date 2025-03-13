import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
import joblib

# 读取数据
file_path = r"C:\Users\WangZheng\Modeling\2.xlsx"
sheets = ["A"]

# 数据预处理
def load_and_preprocess_data(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    X = df.iloc[:, :9].to_numpy()  # 前X列作为输入特征
    y = df.iloc[:, 9].to_numpy()   # 第X列作为输出（灌溉量）
    return X, y

# 创建模型
def create_ridge_model():
    return Ridge(alpha=1.0)

def create_linear_model():
    return LinearRegression()

def create_svm_model():
    return SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)

# 训练与评估
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, scaler_y, sheet_name):
    # 记录训练时间
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # 保存模型到文件
    model_file = f"{sheet_name}_{model_name}.pkl"
    joblib.dump(model, model_file)
    model_size = os.path.getsize(model_file) / (1024 * 1024)  # 转为MB

    # 记录预测时间
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # 模型评估
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rmse, mse, r2, training_time, prediction_time, model_size, y_test, y_pred

# 绘制结果
def plot_results_and_save(y_train_actual, y_train_pred, y_test_actual, y_pred, sheet_name, model_name):
    plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Times New Roman'

    # 计算绘图的最大值（确保是标量）
    max_val = max(y_train_actual.max(), y_test_actual.max(), y_train_pred.max(), y_pred.max())
    max_val = float(max_val)  # 确保 max_val 是一个标量

    # 绘制训练点
    plt.scatter(y_train_actual, y_train_pred, color='gray', s=100, edgecolor='black', alpha=0.5, label='Training Data')

    # 绘制预测点
    plt.scatter(y_test_actual, y_pred,  color={'Linear Regression': 'orange', 'Ridge': 'yellow', 'SVM': 'green'}[model_name],  s=100, edgecolor='black', alpha=0.7, label=f'{model_name} Predicted')

    # 设置坐标范围和刻度
    plt.xlim(0, 45)
    plt.ylim(0, 45)
    plt.xticks(np.arange(0, 46, 10))
    plt.yticks(np.arange(0, 46, 10))

    # 理想拟合线
    plt.plot([0, max_val], [0, max_val], color='gray', linestyle='--', label='Perfect Fit')


    plt.title(f'{model_name} Prediction vs True Values', fontsize=14)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(False)

    plt.savefig(f"{sheet_name}_{model_name}_results.png")
    plt.close()


# 主流程
def main():
    results = []
    for sheet_name in sheets:
        X, y = load_and_preprocess_data(sheet_name)

        # 归一化
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 划分数据
        train_size = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

        # 模型
        models = {'Linear Regression': create_linear_model(),
                  'Ridge': create_ridge_model(),
                  'SVM': create_svm_model()}

        for model_name, model in models.items():
            rmse, mse, r2, training_time, prediction_time, model_size, y_test_actual, y_pred = train_and_evaluate_model(
                model, model_name, X_train, y_train, X_test, y_test, scaler_y, sheet_name)

            # 反归一化训练集结果
            y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
            y_train_pred = scaler_y.inverse_transform(model.predict(X_train).reshape(-1, 1))

            # 保存结果
            results.append({
                'Sheet': sheet_name,
                'Model': model_name,
                'Features': 'Temperature, Humidity, Light, DewPoint, Moisture, Irrigation',
                'RMSE': rmse,
                'MSE': mse,
                'R²': r2,
                'Training Time (s)': training_time,
                'Prediction Time (s)': prediction_time,
                'Model Size (MB)': model_size
            })

            # 绘制结果
            plot_results_and_save(y_train_actual, y_train_pred, y_test_actual, y_pred, sheet_name, model_name)

    # 保存为Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel("model_results.xlsx", index=False)

if __name__ == "__main__":
    main()
