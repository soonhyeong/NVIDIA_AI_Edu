import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

netflix_df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\netflix_titles.csv",
    encoding="utf-8",
)

netflix_df.columns = [
    "show_id",
    "분류",
    "제목",
    "감독",
    "출연진",
    "국가",
    "등록일",
    "출시연도",
    "시청등급",
    "상영시간",
    "장르",
    "설명",
]

netflix_df = netflix_df.drop(columns=["감독", "출연진", "설명"])

# 넷플릭스 콘텐츠 분석

print("넷플릭스가 신작을 가장 많이 출시하는 달은?")
month = {
    "January": 0,
    "February": 0,
    "March": 0,
    "April": 0,
    "May": 0,
    "June": 0,
    "July": 0,
    "August": 0,
    "September": 0,
    "October": 0,
    "November": 0,
    "December": 0,
}


for date in netflix_df["등록일"].dropna():
    date = str(date).strip()

    month_name = date.split()[0]

    if month_name in month:
        month[month_name] += 1
max_month = max(
    month,
    key=month.get,
)

print("가장 많이 출시한 달:", max_month, ",", month[max_month], "개")

print("넷플릭스 관람 등급 중 가장 높은 Top 5")
rating = pd.DataFrame(netflix_df["시청등급"].value_counts())
top_5 = rating.head(5)
top_5.index.name = None
print(top_5.to_string(header=False))

print("넷플릭스 콘텐츠 연도별 증가량")

# 출시연도별 콘텐츠 수를 계산하고 연도순으로 정렬
year = pd.DataFrame(netflix_df["출시연도"].value_counts()).sort_index()

year.columns = ["콘텐츠수"]

# 이전 연도와의 콘텐츠 수 차이
year["증가량"] = year["콘텐츠수"].diff().fillna(0).astype(int)

# 연도별 콘텐츠 수와 증가량 출력
print(year)

# 그래프에는 콘텐츠 수만 표시
plt.figure(figsize=(12, 6))

plt.plot(
    year.index,
    year["콘텐츠수"],
    marker="o",
)

plt.title("출시연도별 넷플릭스 콘텐츠 수")
plt.xlabel("출시연도")
plt.ylabel("콘텐츠 수")
plt.grid(True)
plt.tight_layout()
plt.show()

print("넷플릭스 시청률 영화 vs 드라마")
type = pd.DataFrame(netflix_df["분류"].value_counts())
print(type)

# 국가별 러닝타임
# K-콘텐츠 중 글로벌 Top10
# 초반 흥행과 장기 흥행의 상관관계
# 콘텐츠 공개 요일과 흥행의 상관관계
# 1위를 오랫동안 지속한 콘텐츠 Top 5
# print(new_month.head())
