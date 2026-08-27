# temperature_app.py
# project4 "서울 기온" 분석을 Streamlit 대화형 앱으로
# 실행: streamlit run temperature_app.py
# 같은 폴더에 seoul.csv 가 있어야 합니다.

import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="서울 기온 분석", layout="wide")

# 예비 데이터: seoul.csv 가 없을 때를 대비해 앞부분 일부를 코드에 담아 둠
FALLBACK = """년월,평균기온(℃),최저기온(℃),최고기온(℃)
2011년 1월,-7.2,-17.8,0.3
2011년 2월,1.2,-9.2,13.2
2011년 3월,3.6,-5.5,16.8
"""


@st.cache_data
def load_data():
    try:
        df = pd.read_csv("C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\seoul.csv")
    except FileNotFoundError:
        st.warning("seoul.csv 를 찾지 못해 예비 데이터로 표시합니다.")
        df = pd.read_csv(io.StringIO(FALLBACK))
    # 년월에서 연도와 월 뽑기
    df["연도"] = df["년월"].str.split("년").str[0].astype(int)
    df["월"] = df["년월"].str.extract(r"(\d+)월").astype(int)
    return df


df = load_data()

st.title("서울 월별 기온 분석 (2011-2020)")
st.write(
    "project4의 기온 분석을 대화형 앱으로 만들었습니다. 왼쪽에서 연도를 골라 보세요."
)

# ----- 사이드바: 보기 방식 선택 -----
mode = st.sidebar.radio("보기 방식", ["한 해 자세히 보기", "여러 해 비교하기"])
years = sorted(df["연도"].unique())

if mode == "한 해 자세히 보기":
    year = st.sidebar.selectbox("연도 선택", years)
    sub = df[df["연도"] == year].sort_values("월")

    st.subheader(f"{year}년 기온 요약")
    c1, c2, c3 = st.columns(3)
    c1.metric("최고기온", f"{sub['최고기온(℃)'].max()} ℃")
    c2.metric("최저기온", f"{sub['최저기온(℃)'].min()} ℃")
    c3.metric("평균기온", f"{sub['평균기온(℃)'].mean():.1f} ℃")

    st.subheader("월별 기온 변화")
    chart_df = sub.set_index("월")[["최저기온(℃)", "평균기온(℃)", "최고기온(℃)"]]
    st.line_chart(chart_df)

    # 가장 덥고 추웠던 달
    hot = sub.loc[sub["최고기온(℃)"].idxmax()]
    cold = sub.loc[sub["최저기온(℃)"].idxmin()]
    st.info(
        f"가장 더웠던 달: {hot['월']}월 ({hot['최고기온(℃)']} ℃) / "
        f"가장 추웠던 달: {cold['월']}월 ({cold['최저기온(℃)']} ℃)"
    )

    with st.expander("원본 데이터 보기"):
        st.dataframe(sub[["년월", "평균기온(℃)", "최저기온(℃)", "최고기온(℃)"]])

else:  # 여러 해 비교하기
    picked = st.sidebar.multiselect("연도 선택 (여러 개)", years, default=years[:3])
    if not picked:
        st.warning("연도를 하나 이상 선택하세요.")
        st.stop()

    sub = df[df["연도"].isin(picked)]

    st.subheader("연도별 월평균 기온 비교")
    # 월을 인덱스로, 연도를 열로 하는 표로 변환
    pivot = sub.pivot_table(index="월", columns="연도", values="평균기온(℃)")
    st.line_chart(pivot)

    st.subheader("연도별 연평균 기온")
    yearly = sub.groupby("연도")["평균기온(℃)"].mean().round(2)
    st.bar_chart(yearly)

    with st.expander("연도별 평균 표 보기"):
        st.dataframe(
            yearly.reset_index().rename(columns={"평균기온(℃)": "연평균기온(℃)"})
        )

st.caption("데이터 출처: 기상자료개방포털 (서울 월별 기온)")
