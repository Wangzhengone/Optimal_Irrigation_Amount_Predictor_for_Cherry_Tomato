import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import time
import os
import joblib

# 文件路径和Sheet名称
file_path = r"C:\Users\WangZheng\Modeling\2.xlsx"
sheets = ["A"]

# 数据预处理
def load_and_preprocess_data(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    # 选择前X列作为输入特征，第X列作为输出
    X = df.iloc[:, :9].to_numpy()
    y = df.iloc[:, 9].to_numpy()
    return X, y

# 创建XGBoost模型
def create_xgboost_model():
    return XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=6)

# 创建随机森林模型
def create_rf_model():
    return RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# 模型训练、预测和评估
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler_y, model_name, sheet_name):
    start_time = time.time()
    model.fit(X_train, y_train)  # 模型训练
    training_time = time.time() - start_time

    # 保存模型并获取大小
    model_filename = f"{sheet_name}_{model_name}.joblib"
    joblib.dump(model, model_filename)
    model_size = os.path.getsize(model_filename) / (1024 * 1024)  # 转换为MB

    # 测量预测时间
    start_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_time

    # 反归一化
    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))

    # 模型评估
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model_name": model_name,
        "rmse": rmse,
        "mse": mse,
        "r2": r2,
        "training_time": training_time,
        "prediction_time": prediction_time,
        "model_size": model_size,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_train": y_train
    }

# 绘制结果图
def plot_results(y_train, y_test, y_pred, model_name, color):
    plt.figure(figsize=(6, 6))

    # 设置字体为 Times New Roman
    plt.rcParams["font.family"] = "Times New Roman"

    # 绘制训练点
    plt.scatter(y_train, y_train, s=100, color='gray', alpha=0.4, label='Training Data')

    # 绘制预测点
    plt.scatter(y_test, y_pred, s=100, color=color, edgecolor='black',  alpha=0.4, label='Predicted Data')

    # 绘制完美拟合参考线
    plt.plot([0, 45], [0, 45], color='darkgray', linestyle='--', label='Perfect Fit')

    # 设置坐标范围和刻度
    plt.xlim(0, 45)
    plt.ylim(0, 45)
    plt.xticks(np.arange(0, 46, 10))
    plt.yticks(np.arange(0, 46, 10))

    # 添加标签和标题
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{model_name} Prediction vs True Values', fontsize=14)

    # 添加图例
    plt.legend(fontsize=10)

    # 保存图像
    plt.grid(False)
    plt.savefig(f"{model_name}_prediction_vs_true.png", bbox_inches='tight')
    plt.close()

# 主流程
def main():
    results = []
    for sheet_name in sheets:
        X, y = load_and_preprocess_data(sheet_name)

        # 数据归一化
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # 数据划分
        train_size = int(len(X_scaled) * 0.7)
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

        # 训练XGBoost模型
        xgb_model = create_xgboost_model()
        xgb_result = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, scaler_y, "XGBoost", sheet_name)
        plot_results(xgb_result["y_train"], xgb_result["y_test"], xgb_result["y_pred"], "XGBoost", "pink")
        results.append(xgb_result)

        # 训练随机森林模型
        rf_model = create_rf_model()
        rf_result = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, scaler_y, "Random Forest", sheet_name)
        plot_results(rf_result["y_train"], rf_result["y_test"], rf_result["y_pred"], "Random Forest", "red")
        results.append(rf_result)

    # 保存结果到Excel
    result_df = pd.DataFrame(results)
    result_df.to_excel("model_evaluation_results.xlsx", index=False)

    print("Results saved to model_evaluation_results.xlsx")

if __name__ == "__main__":
    main()
