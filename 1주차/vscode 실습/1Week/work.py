import ast
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="LoL 챔피언 데이터 분석", page_icon="🎮", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "csv"
INFO_FILE = DATA_DIR / "Champions_Info.csv"
RATE_FILE = DATA_DIR / "Champions_rate.csv"
DDRAGON_BASE_URL = "https://ddragon.leagueoflegends.com"
PERCENT_COLUMNS = ["승률", "포지션_사용률", "픽률", "밴률"]
POSITION_ORDER = ["탑", "정글", "미드", "원거리 딜러", "서포터"]
PERCENT_CONFIG = {
    column: st.column_config.NumberColumn(column, format="%.2f%%")
    for column in PERCENT_COLUMNS
}


def configure_matplotlib() -> None:
    """Windows에서 도넛 차트의 한글과 마이너스 기호가 깨지지 않게 한다."""
    try:
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def show_donut_chart(values: pd.Series, center_text: str, colors=None) -> None:
    """범주별 구성비를 도넛 차트로 표시한다."""
    clean = values.dropna()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.pie(
        clean.values,
        labels=clean.index,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        colors=colors,
        wedgeprops={"width": 0.42, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    ax.text(0, 0, center_text, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.axis("equal")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)


configure_matplotlib()


def read_csv_safely(path: Path) -> pd.DataFrame:
    """BOM 포함 UTF-8을 우선 시도하고 일반 UTF-8을 대안으로 사용한다."""
    last_error = None
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error


def preprocess_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """퍼센트 문자열과 0~1 비율을 모두 0~100 숫자로 통일한다."""
    result = df.copy()
    for column in PERCENT_COLUMNS:
        if column not in result:
            continue
        text_values = result[column].astype("string").str.strip()
        has_percent_sign = text_values.str.contains("%", regex=False, na=False)
        values = pd.to_numeric(
            text_values.str.replace("%", "", regex=False),
            errors="coerce",
        )
        # 0.99%는 0.99% 그대로 두고, % 표시가 없는 0.523만 52.3%로 바꾼다.
        ratio_mask = (
            values.notna() & ~has_percent_sign & values.between(0, 1, inclusive="both")
        )
        values.loc[ratio_mask] = values.loc[ratio_mask] * 100
        result[column] = values.where(values.between(0, 100, inclusive="both"))
    return result


def normalize_position(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    aliases = {
        "top": "탑",
        "탑": "탑",
        "jungle": "정글",
        "jg": "정글",
        "정글": "정글",
        "middle": "미드",
        "mid": "미드",
        "미드": "미드",
        "bottom": "원거리 딜러",
        "bot": "원거리 딜러",
        "adc": "원거리 딜러",
        "바텀": "원거리 딜러",
        "원거리 딜러": "원거리 딜러",
        "support": "서포터",
        "sup": "서포터",
        "서포터": "서포터",
    }
    return aliases.get(text, str(value).strip())


def parse_recommended_positions(value) -> list[str]:
    """{미드, 탑}, 리스트, 다양한 구분자 형태를 포지션 목록으로 변환한다."""
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        text = str(value).strip().strip("[]{}()")
        parts = re.split(r"[,/|;]", text)
    normalized = [normalize_position(str(part).strip(" '\"")) for part in parts]
    return list(dict.fromkeys(position for position in normalized if position))


def format_skill(value) -> str:
    """딕셔너리·리스트 문자열로 저장된 스킬을 읽기 좋은 문장으로 만든다."""
    if pd.isna(value) or str(value).strip() == "":
        return "정보 없음"
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return str(value)
    items = (
        list(parsed.values())
        if isinstance(parsed, dict)
        else list(parsed) if isinstance(parsed, (list, tuple, set)) else [parsed]
    )
    return " · ".join(dict.fromkeys(str(item) for item in items))


def clean_html_text(value: str) -> str:
    """Data Dragon 설명에 포함된 간단한 HTML 태그를 제거한다."""
    text = re.sub(r"<br\s*/?>", "\n", str(value), flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


@st.cache_data(ttl=86400, show_spinner=False)
def load_ddragon_champion(api_name: str) -> dict:
    """최신 한국어 Data Dragon 챔피언 상세 정보를 가져온다."""
    headers = {"User-Agent": "LoL-Data-Analysis-Streamlit/1.0"}
    version_response = requests.get(
        f"{DDRAGON_BASE_URL}/api/versions.json", headers=headers, timeout=8
    )
    version_response.raise_for_status()
    version = version_response.json()[0]

    detail_url = (
        f"{DDRAGON_BASE_URL}/cdn/{version}/data/ko_KR/champion/{api_name}.json"
    )
    detail_response = requests.get(detail_url, headers=headers, timeout=8)
    detail_response.raise_for_status()
    champion_data = detail_response.json()["data"]
    if api_name in champion_data:
        detail = champion_data[api_name]
    else:
        detail = next(iter(champion_data.values()))
    return {"version": version, "detail": detail}


def show_ddragon_skills(api_data: dict) -> None:
    """챔피언 아이콘과 패시브·QWER 아이콘 및 설명을 표시한다."""
    version = api_data["version"]
    detail = api_data["detail"]
    champion_icon = (
        f"{DDRAGON_BASE_URL}/cdn/{version}/img/champion/{detail['image']['full']}"
    )

    icon_column, title_column = st.columns([1, 6], vertical_alignment="center")
    icon_column.image(champion_icon, width=110)
    title_column.subheader(f"{detail['name']} · {detail['title']}")
    title_column.caption(f"Data Dragon {version} · {clean_html_text(detail.get('blurb', ''))}")

    passive = detail["passive"]
    skills = [
        {
            "key": "패시브",
            "name": passive["name"],
            "description": passive["description"],
            "icon": f"{DDRAGON_BASE_URL}/cdn/{version}/img/passive/{passive['image']['full']}",
        }
    ]
    for key, spell in zip(["Q", "W", "E", "R"], detail["spells"]):
        skills.append(
            {
                "key": key,
                "name": spell["name"],
                "description": spell["description"],
                "icon": f"{DDRAGON_BASE_URL}/cdn/{version}/img/spell/{spell['image']['full']}",
            }
        )

    st.markdown("#### 스킬 아이콘과 설명")
    for skill in skills:
        image_column, text_column = st.columns([1, 9], vertical_alignment="center")
        image_column.image(skill["icon"], width=64)
        text_column.markdown(f"**{skill['key']} · {skill['name']}**")
        text_column.write(clean_html_text(skill["description"]))


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    info_df = read_csv_safely(INFO_FILE)
    raw_rate_df = read_csv_safely(RATE_FILE)

    required_info = {
        "챔피언명",
        "별명",
        "주_직업군",
        "보조_직업군",
        "추천_포지션",
        "패시브_스킬",
        "Q_스킬",
        "W_스킬",
        "E_스킬",
        "R_스킬",
    }
    required_rate = {"챔피언명", "포지션", *PERCENT_COLUMNS}
    missing_info = required_info - set(info_df.columns)
    missing_rate = required_rate - set(raw_rate_df.columns)
    if missing_info or missing_rate:
        raise ValueError(
            f"필수 열 누락 - 정보 파일: {sorted(missing_info)}, 통계 파일: {sorted(missing_rate)}"
        )

    info_df = info_df.copy()
    info_df["보조_직업군"] = info_df["보조_직업군"].fillna("없음")
    info_df["추천_포지션_목록"] = info_df["추천_포지션"].apply(
        parse_recommended_positions
    )

    rate_df = preprocess_percentage_columns(raw_rate_df)
    rate_df["포지션"] = rate_df["포지션"].apply(normalize_position)
    rate_df = rate_df.drop(columns=["직업군"], errors="ignore")

    merged_df = pd.merge(rate_df, info_df, on="챔피언명", how="inner")
    merged_df["포지션_일치"] = merged_df.apply(
        lambda row: "일치" if row["포지션"] in row["추천_포지션_목록"] else "불일치",
        axis=1,
    )
    audit = {
        "정보_행": len(info_df),
        "정보_챔피언": info_df["챔피언명"].nunique(),
        "통계_행": len(rate_df),
        "통계_챔피언": rate_df["챔피언명"].nunique(),
        "병합_행": len(merged_df),
        "병합_챔피언": merged_df["챔피언명"].nunique(),
        "통계_결측": rate_df[PERCENT_COLUMNS].isna().sum().to_dict(),
    }
    return info_df, rate_df, merged_df, audit


def get_main_position_data(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["포지션_사용률"]).copy()
    return (
        valid.sort_values("포지션_사용률", ascending=False)
        .drop_duplicates("챔피언명")
        .reset_index(drop=True)
    )


def download_csv(df: pd.DataFrame, filename: str, key: str) -> None:
    export = df.drop(columns=["추천_포지션_목록"], errors="ignore").copy()
    st.download_button(
        "분석 결과 CSV 다운로드",
        export.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def show_filter_summary(**filters) -> None:
    values = [f"{name}: {value}" for name, value in filters.items()]
    st.caption("현재 필터 · " + " | ".join(values))


def percent_table(df: pd.DataFrame, columns: list[str] | None = None):
    target = df if columns is None else df[columns]
    return st.dataframe(
        target,
        width="stretch",
        hide_index=True,
        column_config=PERCENT_CONFIG,
    )


def show_basic_information(info_df: pd.DataFrame) -> None:
    st.header("1. 챔피언 기본 정보")
    st.write(
        "챔피언의 기본 정보와 스킬을 조회하고 직업군·추천 포지션 분포를 확인합니다."
    )

    champion = st.selectbox(
        "챔피언 검색", sorted(info_df["챔피언명"].unique()), key="basic_champion"
    )
    selected = info_df.loc[info_df["챔피언명"] == champion].iloc[0]
    api_data = None
    api_name = selected.get("API_이름", "")
    if pd.notna(api_name) and str(api_name).strip():
        try:
            with st.spinner("Riot Data Dragon에서 챔피언 정보를 불러오는 중입니다..."):
                api_data = load_ddragon_champion(str(api_name).strip())
            show_ddragon_skills(api_data)
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            st.warning(
                "Data Dragon 정보를 불러오지 못해 로컬 CSV 정보로 표시합니다. "
                f"({type(error).__name__})"
            )
    if api_data is None:
        st.subheader(f"{selected['챔피언명']} · {selected['별명']}")

    c1, c2, c3 = st.columns(3)
    c1.info(f"주 직업군\n\n**{selected['주_직업군']}**")
    c2.info(f"보조 직업군\n\n**{selected['보조_직업군']}**")
    c3.info(f"추천 포지션\n\n**{', '.join(selected['추천_포지션_목록'])}**")
    skill_labels = ["패시브_스킬", "Q_스킬", "W_스킬", "E_스킬", "R_스킬"]
    with st.expander("로컬 CSV 스킬 정보", expanded=api_data is None):
        for label in skill_labels:
            st.markdown(
                f"**{label.replace('_스킬', '')}:** {format_skill(selected[label])}"
            )

    unique_info = info_df.drop_duplicates("챔피언명")
    recommended = unique_info.explode("추천_포지션_목록")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("전체 챔피언", f"{unique_info['챔피언명'].nunique()}명")
    m2.metric("주 직업군", f"{unique_info['주_직업군'].nunique()}개")
    m3.metric("추천 포지션", f"{recommended['추천_포지션_목록'].nunique()}개")
    m4.metric("보조 직업군 보유", f"{(unique_info['보조_직업군'] != '없음').sum()}명")

    left, right = st.columns(2)
    with left:
        st.subheader("주 직업군별 챔피언 수")
        role_count = (
            unique_info["주_직업군"].value_counts().sort_values(ascending=False)
        )
        st.bar_chart(
            role_count, horizontal=True, x_label="챔피언 수", y_label="주 직업군"
        )
    with right:
        st.subheader("추천 포지션별 챔피언 수")
        pos_count = (
            recommended["추천_포지션_목록"].value_counts().sort_values(ascending=False)
        )
        show_donut_chart(pos_count, f"총 {int(pos_count.sum())}건")
    top_role = role_count.index[0]
    st.success(
        f"가장 많은 주 직업군은 {top_role}이며 {role_count.iloc[0]}명의 챔피언이 있습니다."
    )
    st.caption(
        f"분석 데이터: {len(unique_info)}행 · 결측 스킬 값: {unique_info[skill_labels].isna().sum().sum()}개"
    )
    download_csv(unique_info, "champion_basic_information.csv", "download_basic")


def show_position_statistics(df: pd.DataFrame) -> None:
    st.header("2. 포지션별 챔피언 픽률·승률·밴률")
    st.write("실제 사용 포지션별 성능과 선택 지표를 비교합니다.")

    c1, c2, c3, c4, c5 = st.columns(5)
    positions = [p for p in POSITION_ORDER if p in df["포지션"].unique()]
    position = c1.selectbox("포지션", positions, key="pos_position")
    metric = c2.selectbox("분석 지표", ["픽률", "승률", "밴률"], key="pos_metric")
    top_option = c3.selectbox(
        "상위 데이터", [5, 10, 15, "전체"], index=1, key="pos_top"
    )
    min_pick = c4.number_input("최소 픽률(%)", 0.0, 100.0, 0.0, 0.5, key="pos_min_pick")
    ascending = (
        c5.selectbox("정렬", ["내림차순", "오름차순"], key="pos_sort") == "오름차순"
    )

    valid = df.dropna(subset=["포지션", "픽률", "승률", "밴률"])
    selected = valid[(valid["포지션"] == position) & (valid["픽률"] >= min_pick)].copy()
    if selected.empty:
        st.warning("선택한 조건에 맞는 데이터가 없습니다.")
        return
    selected = selected.sort_values(metric, ascending=ascending)
    result = selected if top_option == "전체" else selected.head(int(top_option))
    show_filter_summary(
        포지션=position,
        지표=metric,
        최소_픽률=f"{min_pick:.1f}%",
        정렬="오름차순" if ascending else "내림차순",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("챔피언", f"{selected['챔피언명'].nunique()}명")
    m2.metric("평균 픽률", f"{selected['픽률'].mean():.2f}%")
    m3.metric("평균 승률", f"{selected['승률'].mean():.2f}%")
    m4.metric("평균 밴률", f"{selected['밴률'].mean():.2f}%")

    position_means = (
        valid.groupby("포지션")[["픽률", "승률", "밴률"]].mean().reindex(positions)
    )
    tabs = st.tabs(["포지션 평균 픽률", "포지션 평균 승률", "포지션 평균 밴률"])
    with tabs[0]:
        st.bar_chart(position_means[["픽률"]], y_label="평균 픽률(%)", x_label="포지션")
    with tabs[1]:
        st.line_chart(position_means[["승률"]], y_label="평균 승률(%)", x_label="포지션")
    with tabs[2]:
        st.area_chart(position_means[["밴률"]], y_label="평균 밴률(%)", x_label="포지션")

    st.subheader(f"{position} · {metric} 순위")
    chart = result.assign(
        표시명=lambda x: x["챔피언명"] + " (" + x["포지션"] + ")"
    ).set_index("표시명")[[metric]]
    st.bar_chart(chart, horizontal=True, x_label=f"{metric}(%)", y_label="챔피언")
    percent_table(
        result, ["챔피언명", "포지션", "티어", "픽률", "승률", "밴률", "포지션_사용률"]
    )
    best = selected.loc[selected[metric].idxmax()]
    st.success(
        f"{position}에서 {metric}이 가장 높은 챔피언은 {best['챔피언명']}({best[metric]:.2f}%)입니다."
    )
    st.caption(
        f"분석 데이터: {len(selected)}행 · 제외된 결측 행: {len(df) - len(valid)}행"
    )
    download_csv(result, "position_statistics.csv", "download_position")


def show_position_match_analysis(df: pd.DataFrame) -> None:
    st.header("3. 추천 포지션과 실제 사용 포지션의 일치도")
    st.write(
        "정보 파일의 추천 포지션과 통계에서 관측된 실제 포지션이 얼마나 일치하는지 분석합니다."
    )
    basis = st.radio(
        "분석 기준",
        ["모든 포지션 행", "챔피언별 대표 포지션"],
        horizontal=True,
        key="match_basis",
    )
    analysis = df.copy() if basis == "모든 포지션 행" else get_main_position_data(df)
    analysis = analysis.dropna(
        subset=["포지션", "포지션_사용률", "픽률", "승률", "밴률"]
    )
    if analysis.empty:
        st.warning("일치도를 분석할 유효 데이터가 없습니다.")
        return

    counts = (
        analysis["포지션_일치"].value_counts().reindex(["일치", "불일치"], fill_value=0)
    )
    match_rate = counts["일치"] / len(analysis) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("일치", f"{counts['일치']}건")
    m2.metric("불일치", f"{counts['불일치']}건")
    m3.metric("전체 일치율", f"{match_rate:.2f}%")
    m4.metric("분석 챔피언", f"{analysis['챔피언명'].nunique()}명")

    left, right = st.columns(2)
    with left:
        st.subheader("일치 여부 구성비")
        show_donut_chart(counts, f"총 {len(analysis)}건", ["#4CAF50", "#EF5350"])
        st.subheader("일치 여부별 평균 지표")
        group_means = analysis.groupby("포지션_일치")[["픽률", "승률", "밴률"]].mean()
        st.bar_chart(group_means, y_label="평균(%)", x_label="일치 여부")
    with right:
        st.subheader("실제 포지션별 일치율")
        pos_match = (
            analysis.assign(일치값=analysis["포지션_일치"].eq("일치").astype(int))
            .groupby("포지션")["일치값"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        st.line_chart(pos_match, y_label="일치율(%)", x_label="실제 포지션")
        st.subheader("직업군별 일치율")
        role_match = (
            analysis.assign(일치값=analysis["포지션_일치"].eq("일치").astype(int))
            .groupby("주_직업군")["일치값"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        st.bar_chart(
            role_match, horizontal=True, x_label="일치율(%)", y_label="주 직업군"
        )

    mismatch = analysis.loc[analysis["포지션_일치"] == "불일치"]
    st.subheader("추천 포지션과 다른 위치에서 사용되는 챔피언")
    mismatch_columns = [
        "챔피언명",
        "주_직업군",
        "추천_포지션",
        "포지션",
        "포지션_사용률",
        "픽률",
        "승률",
        "밴률",
    ]
    percent_table(
        mismatch.sort_values("포지션_사용률", ascending=False), mismatch_columns
    )
    st.info(
        f"{basis} 기준으로 추천 포지션과 실제 포지션이 일치하는 비율은 {match_rate:.2f}%입니다."
    )
    st.caption(f"분석 데이터: {len(analysis)}행 · 불일치 데이터: {len(mismatch)}행")
    download_csv(mismatch[mismatch_columns], "position_mismatch.csv", "download_match")


def show_class_pick_analysis(df: pd.DataFrame) -> None:
    st.header("4. 챔피언 직업군별 선택률")
    st.write("주 직업군에 따라 챔피언 선택률의 평균·중앙값·합계를 비교합니다.")
    c1, c2, c3 = st.columns(3)
    basis = c1.selectbox(
        "집계 방식",
        ["포지션별 데이터", "챔피언별 픽률 합계", "대표 포지션"],
        key="role_basis",
    )
    positions = ["전체", *[p for p in POSITION_ORDER if p in df["포지션"].unique()]]
    position = c2.selectbox("포지션 필터", positions, key="role_position")
    role_options = ["전체", *sorted(df["주_직업군"].dropna().unique())]
    selected_role = c3.selectbox("상세 직업군", role_options, key="role_selected")

    base = df if position == "전체" else df[df["포지션"] == position]
    base = base.dropna(subset=["픽률", "주_직업군"])
    if basis == "챔피언별 픽률 합계":
        analysis = base.groupby(
            ["챔피언명", "주_직업군", "보조_직업군"], as_index=False
        ).agg({"픽률": "sum", "승률": "mean", "밴률": "mean"})
    elif basis == "대표 포지션":
        analysis = get_main_position_data(base)
    else:
        analysis = base.copy()
    if analysis.empty:
        st.warning("선택한 조건에 맞는 데이터가 없습니다.")
        return

    role_summary = (
        analysis.groupby("주_직업군")
        .agg(
            챔피언_수=("챔피언명", "nunique"),
            평균_픽률=("픽률", "mean"),
            픽률_중앙값=("픽률", "median"),
            픽률_합계=("픽률", "sum"),
        )
        .sort_values("평균_픽률", ascending=False)
    )
    support_pick = (
        analysis.assign(
            보조_직업군_유무=analysis["보조_직업군"]
            .ne("없음")
            .map({True: "있음", False: "없음"})
        )
        .groupby("보조_직업군_유무")["픽률"]
        .mean()
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("직업군", f"{analysis['주_직업군'].nunique()}개")
    m2.metric("챔피언", f"{analysis['챔피언명'].nunique()}명")
    m3.metric("전체 평균 픽률", f"{analysis['픽률'].mean():.2f}%")
    m4.metric("전체 중앙 픽률", f"{analysis['픽률'].median():.2f}%")
    show_filter_summary(집계_방식=basis, 포지션=position, 직업군=selected_role)

    tabs = st.tabs(["챔피언 수", "평균 픽률", "픽률 중앙값", "픽률 합계"])
    with tabs[0]:
        st.bar_chart(role_summary[["챔피언_수"]], horizontal=True, x_label="챔피언 수", y_label="주 직업군")
    with tabs[1]:
        st.line_chart(role_summary[["평균_픽률"]], y_label="평균 픽률(%)", x_label="주 직업군")
    with tabs[2]:
        st.area_chart(role_summary[["픽률_중앙값"]], y_label="픽률 중앙값(%)", x_label="주 직업군")
    with tabs[3]:
        st.bar_chart(role_summary[["픽률_합계"]], horizontal=True, x_label="픽률 합계(%)", y_label="주 직업군")

    st.subheader("보조 직업군 유무에 따른 평균 픽률")
    st.bar_chart(
        support_pick, horizontal=True, x_label="평균 픽률(%)", y_label="보조 직업군"
    )
    detail = (
        analysis
        if selected_role == "전체"
        else analysis[analysis["주_직업군"] == selected_role]
    )
    detail = detail.sort_values("픽률", ascending=False)
    st.subheader(f"{selected_role} 픽률 상위 챔피언")
    top = detail.head(15).assign(
        표시명=lambda x: x["챔피언명"]
        + (" (" + x["포지션"] + ")" if "포지션" in x else "")
    )
    st.bar_chart(
        top.set_index("표시명")[["픽률"]],
        horizontal=True,
        x_label="픽률(%)",
        y_label="챔피언",
    )
    shown = [
        column
        for column in [
            "챔피언명",
            "주_직업군",
            "보조_직업군",
            "포지션",
            "픽률",
            "승률",
            "밴률",
        ]
        if column in detail
    ]
    percent_table(detail, shown)
    winner = role_summary.index[0]
    st.success(
        f"평균 픽률이 가장 높은 주 직업군은 {winner}({role_summary.iloc[0]['평균_픽률']:.2f}%)입니다. 평균과 중앙값을 함께 비교하면 고픽률 챔피언의 영향을 확인할 수 있습니다."
    )
    st.caption(
        f"분석 데이터: {len(analysis)}행 · 픽률 결측 제외: {base['픽률'].isna().sum()}행"
    )
    download_csv(detail[shown], "class_pick_analysis.csv", "download_role")


def correlation_description(value: float) -> str:
    strength = abs(value)
    if strength < 0.2:
        level = "매우 약한"
    elif strength < 0.4:
        level = "약한"
    elif strength < 0.6:
        level = "보통"
    elif strength < 0.8:
        level = "강한"
    else:
        level = "매우 강한"
    direction = "양의" if value >= 0 else "음의"
    return f"{level} {direction} 관계"


def classify_champion_type(
    df: pd.DataFrame, pick_cut: float, win_cut: float, ban_cut: float
) -> pd.DataFrame:
    result = df.copy()
    result["챔피언_유형"] = "저활용 챔피언"
    result.loc[
        (result["픽률"] >= pick_cut) & (result["승률"] >= win_cut), "챔피언_유형"
    ] = "인기 강챔"
    result.loc[
        (result["픽률"] < pick_cut) & (result["승률"] >= win_cut), "챔피언_유형"
    ] = "숨은 강챔"
    result.loc[
        (result["픽률"] >= pick_cut) & (result["승률"] < win_cut), "챔피언_유형"
    ] = "인기 중심 챔피언"
    result["밴_유형"] = (
        result["밴률"].ge(ban_cut).map({True: "고밴 챔피언", False: "저밴 챔피언"})
    )
    return result


def show_correlation_and_type_analysis(df: pd.DataFrame) -> None:
    st.header("5. 픽률·승률·밴률 상관관계와 챔피언 유형")
    st.write("선택률과 성과의 관계를 확인하고 챔피언을 네 가지 유형으로 분류합니다.")
    c1, c2, c3 = st.columns(3)
    basis = c1.selectbox(
        "분석 데이터 기준",
        ["챔피언별 대표 포지션", "모든 챔피언-포지션 데이터", "특정 포지션"],
        key="corr_basis",
    )
    position = c2.selectbox(
        "특정 포지션",
        [p for p in POSITION_ORDER if p in df["포지션"].unique()],
        disabled=basis != "특정 포지션",
        key="corr_position",
    )
    highlight = c3.selectbox(
        "강조할 챔피언",
        ["없음", *sorted(df["챔피언명"].unique())],
        key="corr_highlight",
    )

    if basis == "챔피언별 대표 포지션":
        analysis = get_main_position_data(df)
    elif basis == "특정 포지션":
        analysis = df[df["포지션"] == position].copy()
    else:
        analysis = df.copy()
    analysis = analysis.dropna(subset=["픽률", "승률", "밴률", "포지션_사용률"])
    if analysis.empty:
        st.warning("상관관계를 계산할 유효 데이터가 없습니다.")
        return

    st.subheader("상관관계")
    corr = analysis[["픽률", "승률", "밴률"]].corr()
    c1, c2, c3 = st.columns(3)
    pairs = [("픽률", "승률"), ("픽률", "밴률"), ("승률", "밴률")]
    for container, (first, second) in zip([c1, c2, c3], pairs):
        value = corr.loc[first, second]
        container.metric(f"{first} ↔ {second}", f"{value:.3f}")
        container.caption(correlation_description(value))
    st.dataframe(
        corr.style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
        width="stretch",
    )

    scatter_tabs = st.tabs(["픽률 ↔ 승률", "픽률 ↔ 밴률", "승률 ↔ 밴률"])
    for tab, (x, y) in zip(scatter_tabs, pairs):
        with tab:
            st.scatter_chart(
                analysis,
                x=x,
                y=y,
                size="포지션_사용률",
                color="포지션",
                x_label=f"{x}(%)",
                y_label=f"{y}(%)",
            )
            if highlight != "없음":
                point = analysis[analysis["챔피언명"] == highlight]
                if not point.empty:
                    st.dataframe(
                        point[["챔피언명", "포지션", x, y]],
                        hide_index=True,
                        width="stretch",
                        column_config=PERCENT_CONFIG,
                    )
    st.warning(
        "상관관계는 변수들이 함께 움직이는 정도를 나타내며 인과관계를 의미하지 않습니다."
    )

    st.subheader("챔피언 유형 분류")
    mode = st.radio(
        "분류 기준",
        ["전체 평균", "중앙값", "직접 입력"],
        horizontal=True,
        key="type_mode",
    )
    default_pick = (
        analysis["픽률"].mean() if mode == "전체 평균" else analysis["픽률"].median()
    )
    default_win = (
        analysis["승률"].mean() if mode == "전체 평균" else analysis["승률"].median()
    )
    default_ban = (
        analysis["밴률"].mean() if mode == "전체 평균" else analysis["밴률"].median()
    )
    t1, t2, t3, t4 = st.columns(4)
    pick_cut = t1.number_input(
        "픽률 기준(%)",
        0.0,
        100.0,
        float(default_pick),
        0.1,
        disabled=mode != "직접 입력",
        key="type_pick",
    )
    win_cut = t2.number_input(
        "승률 기준(%)",
        0.0,
        100.0,
        float(default_win),
        0.1,
        disabled=mode != "직접 입력",
        key="type_win",
    )
    min_pick = t3.number_input(
        "최소 픽률(%)", 0.0, 100.0, 0.0, 0.1, key="type_min_pick"
    )
    min_usage = t4.number_input(
        "최소 포지션 사용률(%)", 0.0, 100.0, 0.0, 1.0, key="type_min_usage"
    )
    if mode != "직접 입력":
        pick_cut, win_cut = float(default_pick), float(default_win)
    classified_base = analysis[
        (analysis["픽률"] >= min_pick) & (analysis["포지션_사용률"] >= min_usage)
    ]
    if classified_base.empty:
        st.warning("유형 분류 조건을 만족하는 데이터가 없습니다.")
        return
    ban_cut = float(default_ban)
    classified = classify_champion_type(classified_base, pick_cut, win_cut, ban_cut)
    type_order = ["인기 강챔", "숨은 강챔", "인기 중심 챔피언", "저활용 챔피언"]
    type_counts = (
        classified["챔피언_유형"].value_counts().reindex(type_order, fill_value=0)
    )
    type_means = (
        classified.groupby("챔피언_유형")[["픽률", "승률", "밴률"]]
        .mean()
        .reindex(type_order)
    )
    left, right = st.columns(2)
    with left:
        st.subheader("유형별 챔피언 구성비")
        show_donut_chart(type_counts, f"총 {int(type_counts.sum())}건")
    with right:
        st.subheader("유형별 평균 지표")
        st.bar_chart(type_means, y_label="평균(%)", x_label="유형")
    st.scatter_chart(
        classified,
        x="픽률",
        y="승률",
        color="챔피언_유형",
        size="밴률",
        x_label="픽률(%)",
        y_label="승률(%)",
    )

    selected_type = st.selectbox(
        "확인할 유형", ["전체", *type_order], key="selected_type"
    )
    detail = (
        classified
        if selected_type == "전체"
        else classified[classified["챔피언_유형"] == selected_type]
    )
    detail_columns = [
        "챔피언명",
        "주_직업군",
        "포지션",
        "픽률",
        "승률",
        "밴률",
        "챔피언_유형",
        "밴_유형",
    ]
    percent_table(detail.sort_values(["승률", "픽률"], ascending=False), detail_columns)
    hidden_count = type_counts["숨은 강챔"]
    pick_ban_corr = corr.loc["픽률", "밴률"]
    st.info(
        f"픽률과 밴률의 상관계수는 {pick_ban_corr:.3f}로 {correlation_description(pick_ban_corr)}입니다. 승률 기준 이상이면서 픽률 기준 미만인 숨은 강챔은 {hidden_count}개입니다."
    )
    show_filter_summary(
        분석_기준=basis,
        분류_기준=mode,
        픽률_기준=f"{pick_cut:.2f}%",
        승률_기준=f"{win_cut:.2f}%",
        밴률_기준=f"{ban_cut:.2f}%",
    )
    st.caption(
        f"분석 데이터: {len(classified)}행 · 필터 전 유효 데이터: {len(analysis)}행"
    )
    download_csv(detail[detail_columns], "champion_type_analysis.csv", "download_type")


try:
    info_df, rate_df, merged_df, audit = load_data()
except FileNotFoundError as error:
    st.error(f"데이터 파일을 찾을 수 없습니다: {error.filename}")
    st.stop()
except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as error:
    st.error(f"CSV 파일을 읽거나 전처리하는 중 오류가 발생했습니다: {error}")
    st.stop()
except Exception as error:
    st.error(f"예상하지 못한 오류가 발생했습니다: {error}")
    st.stop()

st.title("🎮 리그 오브 레전드 챔피언 데이터 분석")
st.write("로컬 CSV를 이용해 챔피언 정보와 포지션별 픽률·승률·밴률을 분석합니다.")

topic = st.sidebar.radio(
    "분석 소주제",
    [
        "1. 챔피언 기본 정보",
        "2. 포지션별 챔피언 픽률·승률·밴률",
        "3. 추천 포지션과 실제 사용 포지션의 일치도",
        "4. 챔피언 직업군별 선택률",
        "5. 픽률·승률·밴률 상관관계와 챔피언 유형",
    ],
)

with st.sidebar.expander("데이터 로딩 및 품질"):
    st.write(f"정보 CSV: {audit['정보_행']}행 / {audit['정보_챔피언']}명")
    st.write(f"통계 CSV: {audit['통계_행']}행 / {audit['통계_챔피언']}명")
    st.write(f"병합 결과: {audit['병합_행']}행 / {audit['병합_챔피언']}명")
    st.write("통계 결측값", audit["통계_결측"])

if topic.startswith("1."):
    show_basic_information(info_df)
elif topic.startswith("2."):
    show_position_statistics(merged_df)
elif topic.startswith("3."):
    show_position_match_analysis(merged_df)
elif topic.startswith("4."):
    show_class_pick_analysis(merged_df)
else:
    show_correlation_and_type_analysis(merged_df)

st.divider()
st.caption("데이터 출처: Kaggle.com")
