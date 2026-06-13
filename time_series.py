import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# 1. データ準備 (Data Preparation)
# 日付列をdatetime型に変換し、それをインデックス(行ラベル)として設定します
# ---------------------------------------------------------
df['Month'] = pd.to_datetime(df['Month'])
time_index = df.set_index('Month').index

# ---------------------------------------------------------
# 2. 特徴量（タイムダミー）の作成 (Feature Engineering)
# DeterministicProcessを使って、時間トレンドを表す特徴量を作成します
# order=1 は1次（直線的な）トレンドを意味します
# ---------------------------------------------------------
trend_generator = DeterministicProcess(index=time_index, order=1)

# 学習用の特徴量（過去の期間）を作成
X_train = trend_generator.in_sample()

# 予測用の特徴量（未来の5ステップ分）を作成
X_future = trend_generator.out_of_sample(steps=5)

# 目的変数（予測したい値: 乗客数）
y_train = df['#Passengers']

# ---------------------------------------------------------
# 3. モデルの学習 (Model Training)
# 線形回帰モデルを使って、時間の経過（X_train）と乗客数（y_train）の関係を学習します
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. 予測 (Prediction)
# ---------------------------------------------------------
# 学習データ期間に対するモデルの予測値（フィッティングの確認用）
y_pred_train = model.predict(X_train)
y_pred_train_series = pd.Series(y_pred_train, index=time_index)

# 未来の期間に対する予測値
y_pred_future = model.predict(X_future)
y_pred_future_series = pd.Series(y_pred_future, index=X_future.index)

# ---------------------------------------------------------
# 5. 可視化 (Visualization)
# 実際のデータと、モデルによる予測結果をグラフに描画します
# ---------------------------------------------------------
fig, ax = plt.subplots()

# 実際のデータをプロット
ax.plot(time_index, y_train, label='Actual Passengers')

# 学習データ期間の予測（トレンドライン）をプロット
ax.plot(time_index, y_pred_train_series, label='Predicted (Train)')

# 未来の予測をプロット
ax.plot(y_pred_future_series.index, y_pred_future_series, label='Predicted (Future)')

ax.set_xlabel('Year')
ax.set_ylabel('Passengers')
ax.legend()
plt.show()
