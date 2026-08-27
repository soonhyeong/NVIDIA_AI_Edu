import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\penguins.csv", encoding="utf-8")
df.columns = [
    "번호",
    "종",
    "섬",
    "부리길이",
    "부리너비",
    "날개길이",
    "체중",
    "성별",
    "연도",
]

sns.scatterplot(
    data=df, x="부리길이", y="날개길이", hue="종", style="성별", s=50, alpha=0.7
)

plt.show()
