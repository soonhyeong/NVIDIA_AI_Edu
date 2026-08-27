import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
import numpy as np

info_df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\Champions_info.csv",
    encoding="utf-8",
)

info_df["보조_직업군"] = info_df["보조_직업군"].fillna("")

info_df = info_df.drop(
    columns=[
        "챔피언_ID",
        "API_이름",
        "난이도",
        "사용_자원",
        "기본_능력치",
        "공격_형태",
        "출시일",
        "출시_패치",
        "최근_변경_패치",
        "세부_역할군",
        "실제_활용_포지션",
        "피해량",
        "내구력",
        "군중_제어",
        "기동력",
        "보조_능력",
        "공격_성향",
        "적응형_능력치",
        "파랑_정수_가격",
        "RP_가격",
        "전체_스킬",
        "본명",
        "별칭",
    ]
)

rate_df = pd.read_csv(
    "C:\\Users\\Admin\\NVIDIA\\1Week\\csv\\Champions_rate.csv",
    encoding="utf-8",
)

# % 문자를 제거한 뒤 숫자로 변환한다. 변환할 수 없는 값은 NaN이 된다.
float_columns = ["승률", "픽률", "밴률", "포지션_사용률"]
for column in float_columns:
    rate_df[column] = pd.to_numeric(
        rate_df[column].astype("string").str.rstrip("%"),
        errors="coerce",
    ).astype(float)

rate_df = rate_df.drop(columns=["직업군", "티어", "종합_점수", "변화량", "KDA"])

tot_df = pd.merge(info_df, rate_df, on="챔피언명")
main_warrior = tot_df[tot_df["주_직업군"] == "전사"]
main_mage = tot_df[tot_df["주_직업군"] == "마법사"]
main_assassin = tot_df[tot_df["주_직업군"] == "암살자"]
main_tank = tot_df[tot_df["주_직업군"] == "탱커"]
main_adc = tot_df[tot_df["주_직업군"] == "원거리 딜러"]
main_support = tot_df[tot_df["주_직업군"] == "서포터"]

# ===========================================================================================
# 주 직업군 별 세부 분석
# ===========================================================================================

# 분석할 직업군
class_list = [
    "전사",
    "마법사",
    "암살자",
    "탱커",
    "원거리 딜러",
    "서포터",
]

# 추천 포지션
positions = [
    "탑",
    "미드",
    "정글",
    "원거리 딜러",
    "서포터",
]


# 히트맵을 그리는 함수
def draw_heatmap(data, ax, title):
    # 실제 수치 뒤에 % 표시
    annotation = data.map(lambda value: f"{value:.2f}%")

    sns.heatmap(
        data,
        annot=annotation,
        fmt="",
        annot_kws={"fontsize": 6},
        linewidths=0.5,
        cbar=False,
        yticklabels=False,
        ax=ax,
    )

    # 데이터 칸마다 챔피언명을 전부 표시
    tick_positions = np.arange(len(data)) + 0.5

    ax.set_yticks(tick_positions)

    ax.set_yticklabels(
        data.index.tolist(),
        rotation=0,
        fontsize=6,
    )

    # 처음이나 마지막 챔피언이 잘리는 현상 방지
    ax.set_ylim(
        len(data),
        0,
    )

    ax.set_title(
        title,
        fontsize=12,
    )

    ax.set_xlabel("")
    ax.set_ylabel("챔피언")


# 모든 직업군 그래프 생성
for class_name in class_list:

    # 현재 직업군만 선택
    class_df = tot_df[tot_df["주_직업군"] == class_name].copy()

    # 해당 직업군의 데이터가 없으면 건너뛰기
    if class_df.empty:
        continue

    # 동일 챔피언이 여러 포지션에 있는 경우 하나로 집계
    class_rate = class_df.groupby(
        "챔피언명",
        as_index=False,
    ).agg(
        # 포지션별 승률의 단순 평균
        승률=("승률", "mean"),
        # 포지션별 픽률 합계
        픽률=("픽률", "sum"),
        # 포지션별 밴률의 단순 평균
        밴률=("밴률", "mean"),
    )

    # 결측값 제거
    class_rate = class_rate.dropna(
        subset=[
            "승률",
            "픽률",
            "밴률",
        ]
    )

    # 승률 내림차순
    win_heatmap = class_rate.sort_values(
        "승률",
        ascending=False,
    ).set_index(
        "챔피언명"
    )[["승률"]]

    # 픽률 내림차순
    pick_heatmap = class_rate.sort_values(
        "픽률",
        ascending=False,
    ).set_index(
        "챔피언명"
    )[["픽률"]]

    # 밴률 내림차순
    ban_heatmap = class_rate.sort_values(
        "밴률",
        ascending=False,
    ).set_index(
        "챔피언명"
    )[["밴률"]]

    # 추천 포지션 집계에서는 챔피언 중복 제거
    class_position = class_df[
        [
            "챔피언명",
            "추천_포지션",
        ]
    ].drop_duplicates(subset="챔피언명")

    position_count = {}

    for position in positions:
        position_count[position] = (
            class_position["추천_포지션"]
            .astype(str)
            .str.contains(
                position,
                regex=False,
                na=False,
            )
            .sum()
        )

    position_count_df = pd.DataFrame(
        position_count.items(),
        columns=[
            "추천_포지션",
            "count",
        ],
    )

    # 개수가 0인 포지션은 원그래프에서 제외
    position_count_df = position_count_df[position_count_df["count"] > 0]

    # 챔피언 수에 따라 Figure 높이 자동 조절
    # 2행 구조이므로 전체 챔피언명이 보이도록 높이 확보
    figure_height = max(
        10,
        len(class_rate) * 0.36,
    )

    figure, (
        (ax1, ax2),
        (ax3, ax4),
    ) = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(11, figure_height),
    )

    figure.suptitle(
        f"{class_name} 챔피언 분석",
        fontsize=17,
        fontweight="bold",
    )

    # 승률 히트맵
    draw_heatmap(
        win_heatmap,
        ax1,
        "챔피언별 승률",
    )

    # 픽률 히트맵
    draw_heatmap(
        pick_heatmap,
        ax2,
        "챔피언별 픽률",
    )

    # 밴률 히트맵
    draw_heatmap(
        ban_heatmap,
        ax3,
        "챔피언별 밴률",
    )

    # 추천 포지션 원그래프
    position_count_df.plot.pie(
        y="count",
        labels=position_count_df["추천_포지션"],
        autopct="%.1f%%",
        startangle=90,
        counterclock=False,
        legend=False,
        textprops={"fontsize": 8},
        ax=ax4,
    )

    ax4.set_title(
        "추천 포지션 분포",
        fontsize=12,
    )

    ax4.set_ylabel("")
    ax4.set_aspect("equal")

    figure.tight_layout(rect=[0, 0, 1, 0.96])

    # 실제 표시되는 챔피언 수 확인
    print(f"{class_name}: " f"{len(class_rate)}명 표시")

# ===========================================================================================
# 전체 챔피언의 주 직업군 분포
# ===========================================================================================

class_count = (
    info_df[["챔피언명", "주_직업군"]]
    .drop_duplicates("챔피언명")["주_직업군"]
    .value_counts()
)


# 원그래프에 비율과 챔피언 수를 함께 표시
def show_class_count(percent):
    total = class_count.sum()
    count = round(percent * total / 100)

    return f"{percent:.1f}%\n({count}명)"


figure, ax = plt.subplots(figsize=(8, 8))

class_count.plot.pie(
    autopct=show_class_count,
    startangle=90,
    counterclock=False,
    legend=False,
    textprops={"fontsize": 9},
    ax=ax,
)

ax.set_title(
    "전체 챔피언 주 직업군 분포",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel("")
ax.set_aspect("equal")

figure.tight_layout()

# 반복문에서 모든 Figure를 생성한 후 한 번만 실행
plt.show()
