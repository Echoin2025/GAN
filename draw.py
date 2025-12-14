import pandas as pd
import matplotlib.pyplot as plt

# 读取日志
df = pd.read_csv('logs/training_log.txt')

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(df['D_Loss'], label='D Loss', alpha=0.7)
plt.plot(df['G_Loss'], label='G Loss', alpha=0.7)
plt.legend()
plt.show()