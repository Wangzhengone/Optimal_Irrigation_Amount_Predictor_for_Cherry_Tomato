import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
import tempfile

# 文件路径
file_path = r"C:\Users\WangZheng\Modeling\2.xlsx"
result_folder = r"C:\Users\WangZheng\Modeling\Results"

# 确保结果文件夹存在
os.makedirs(result_folder, exist_ok=True)

# 加载数据，第一行为列名
def load_data(sheet_name="A"):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)  # 第一行作为列名
    return df

# 创建LSTM模型
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 创建GRU模型
def create_gru_model(input_shape):
    model = Sequential([
        GRU(50, activation='relu', input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 创建Transformer模型
def create_transformer_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=2)(inputs, inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(50, activation='relu')(x)
    output = Dense(1)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(), loss='mse')
    return model

# 模型训练、评估并记录时间和模型大小
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, scaler_y, model_name, sheet_name):
    start_time = time.time()
    model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0)
    training_time = time.time() - start_time  # 训练时间

    # 测试集预测时间
    start_predict_time = time.time()
    y_pred = model.predict(X_test)
    prediction_time = time.time() - start_predict_time

    # 保存模型到临时文件夹并计算大小
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file_path = os.path.join(tmpdir, f"{sheet_name}_{model_name}.h5")
        model.save(model_file_path)
        model_size = os.path.getsize(model_file_path) / (1024 * 1024)  # 转换为MB

    # 反标准化预测值和实际值
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_test, y_pred, rmse, mse, r2, training_time, prediction_time, model_size

# 绘制结果图并保存
def plot_results_and_save(y_test, y_pred, model_name, result_folder, sheet_name, X_train, y_train, scaler_y):
    plt.figure(figsize=(6, 6))
    plt.rcParams['font.family'] = 'Times New Roman'

    if model_name == "GRU":
        color = 'purple'
    elif model_name == "LSTM":
        color = 'blue'
    else:
        color = 'black'

    # 反标准化训练数据
    y_train = scaler_y.inverse_transform(y_train.reshape(-1, 1))

    # 绘制训练点
    plt.scatter(y_train, y_train, label='Training Data', s=100, edgecolor='gray', facecolor='gray', alpha=0.4)

    # 绘制预测点
    plt.scatter(y_test, y_pred, label=f'{model_name} Prediction', 
                s=100, edgecolor='black', facecolor=color, alpha=0.4)

    # 完美拟合线
    plt.plot([0, 45], [0, 45], color='grey', linestyle='--', label='Perfect fit')
    plt.title(f'{model_name} Model Prediction', fontsize=14)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.grid(False)
    plt.xlim(0, 45)
    plt.ylim(0, 45)
    plt.xticks(np.arange(0, 46, 10), fontsize=10)
    plt.yticks(np.arange(0, 46, 10), fontsize=10)
    plt.legend(fontsize=10)
    plt.savefig(os.path.join(result_folder, f"{sheet_name}_{model_name}_prediction_vs_true.png"))
    plt.close()

# 保存结果到Excel
def save_results_to_excel(results):
    result_file = os.path.join(result_folder, "Results_GRU_LSTM_Transformer.xlsx")
    df = pd.DataFrame(results)
    df.to_excel(result_file, index=False)
    print(f"Results saved to {result_file}")

# 主流程
def main():
    sheet_name = "A"
    df = load_data(sheet_name)
    results = []

    input_data = df.iloc[:, :9].to_numpy()
    output_data = df.iloc[:, 9].to_numpy()
    feature_names = df.columns[:9].tolist()

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(input_data)
    y_scaled = scaler_y.fit_transform(output_data.reshape(-1, 1)).flatten()

    train_size = int(len(X_scaled) * 0.7)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]

    input_shape = (X_train.shape[1], 1)
    X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    for model_name, create_model in [("GRU", create_gru_model), 
                                     ("LSTM", create_lstm_model), 
                                     ("Transformer", create_transformer_model)]:
        model = create_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
        y_test_pred, y_pred, rmse, mse, r2, training_time, prediction_time, model_size = train_and_evaluate_model(
            model, X_train_reshaped, y_train, X_test_reshaped, y_test, scaler_y, model_name, sheet_name)
        results.append({
            "Model": model_name,
            "Features": ', '.join(feature_names),
            "RMSE": rmse,
            "MSE": mse,
            "R²": r2,
            "Training Time (s)": training_time,
            "Prediction Time (s)": prediction_time,
            "Model Size (MB)": model_size
        })
        plot_results_and_save(y_test_pred, y_pred, model_name, result_folder, sheet_name, X_train, y_train, scaler_y)

    save_results_to_excel(results)

if __name__ == "__main__":
    main()
