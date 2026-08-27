import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\train.csv",
    encoding="utf-8",
    parse_dates=["datetime"],
)

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["minute"] = df["datetime"].dt.minute
df["second"] = df["datetime"].dt.second
df["dayofweek"] = df["datetime"].dt.dayofweek

sns.barplot(data=df, x="month", y="count")
plt.show()
