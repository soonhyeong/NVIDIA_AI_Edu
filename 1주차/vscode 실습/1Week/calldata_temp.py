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

temp_df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\Tempdata_2008.csv",
    encoding="cp949",
    parse_dates=["일시"],
)

temp_df.columns = [
    "지점번호",
    "지점명",
    "일자",
    "평균기온",
    "최고기온",
    "최고기온시각",
    "최저기온",
    "최저기온시각",
    "일교차",
]

is_true = (call_df["발신지1"] == "서울") & (call_df["대분류"] == "음식점")
call_df = call_df[is_true]
call_df = call_df.drop(
    columns=["연령", "성별", "발신지1", "발신지2", "대분류", "통화비율"]
)

temp_df = temp_df.drop(
    columns=[
        "지점번호",
        "지점명",
        "최고기온",
        "최저기온",
        "일교차",
        "최고기온시각",
        "최저기온시각",
    ]
)
tot_df = pd.merge(call_df, temp_df, on="일자")

low_temp = tot_df[tot_df["평균기온"] < 28]
low_temp = pd.DataFrame(low_temp["중분류"].value_counts())

high_temp = tot_df[tot_df["평균기온"] >= 30]
high_temp = pd.DataFrame(high_temp["중분류"].value_counts())

plt.title("평균기온에 따른 주문 건수")
sns.countplot(data=tot_df, x="평균기온", hue="중분류")
plt.xticks(rotation=45)
plt.legend()

figure, (ax1, ax2) = plt.subplots(ncols=2)
figure.set_size_inches(40, 15)
low_temp.plot.pie(y="count", autopct="%.2f%%", ax=ax1)
high_temp.plot.pie(y="count", autopct="%.2f%%", ax=ax2)
ax1.set_title("평균기온이 28도 미만인 날")
ax2.set_title("평균기온이 30도 이상인 날")

plt.show()
