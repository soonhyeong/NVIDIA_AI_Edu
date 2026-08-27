import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib

call_df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\Calldata_2008.csv",
    encoding="utf-8",
    parse_dates=["일자(YYYYMMDD)"],
)
call_df.columns = [
    "일자",
    "연령",
    "성별",
    "발신지1",
    "발신지2",
    "대분류",
    "중분류",
    "통화비율",
]

rain_df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\Raindata_2008.csv",
    encoding="cp949",
    parse_dates=["일시"],
)

rain_df.columns = [
    "지점번호",
    "지점명",
    "일자",
    "강수량",
    "1시간최대강수량",
    "1시간최대강수량시각",
]


is_true = (call_df["발신지1"] == "서울") & (call_df["대분류"] == "음식점")
call_df = call_df[is_true]
call_df = call_df.drop(
    columns=["연령", "성별", "발신지1", "발신지2", "대분류", "통화비율"]
)
# print(call_df)

rain_df["강수량"] = rain_df["강수량"].fillna(0)
rain_df = rain_df.drop(
    columns=["지점번호", "지점명", "1시간최대강수량", "1시간최대강수량시각"]
)
# print(rain_df)

tot_df = pd.merge(call_df, rain_df, on="일자")
# print(tot_df)

no_rain = tot_df[tot_df["강수량"] == 0]
no_rain = pd.DataFrame(no_rain["중분류"].value_counts())
# print(no_rain)

yes_rain = tot_df[tot_df["강수량"] >= 50]
yes_rain = pd.DataFrame(yes_rain["중분류"].value_counts())
# print(yes_rin)

plt.title("강수량에 따른 음식 종류별 주문 건수")
sns.countplot(data=tot_df, x="강수량", hue="중분류")
plt.legend

figure, (ax1, ax2) = plt.subplots(ncols=2)
# figure.set_size_inches(40, 15)
no_rain.plot.pie(y="count", autopct="%.2f%%", ax=ax1)
yes_rain.plot.pie(y="count", autopct="%.2f%%", ax=ax2)
ax1.set_title("비가 오지 않은 날")
ax2.set_title("강수량이 50mm 이상인 날")
plt.show()
