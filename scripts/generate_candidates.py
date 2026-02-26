import os
import sys
from pathlib import Path

import pandas as pd

try:
    import akshare as ak
except ImportError:
    print("akshare 未安装，请确认 requirements.txt 中包含 akshare")
    sys.exit(1)


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT = OUTPUT_DIR / "candidates.txt"
OUTPUT_CSV = OUTPUT_DIR / "candidates_debug.csv"


def safe_num(series: pd.Series) -> pd.Series:
    """尽量把字符串列安全转成数值列。"""
    return pd.to_numeric(series, errors="coerce")


def load_spot_data() -> pd.DataFrame:
    """
    获取 A 股实时/当日快照。
    优先使用东财接口快照。
    """
    df = ak.stock_zh_a_spot_em()
    if df is None or df.empty:
        raise RuntimeError("获取A股快照失败：stock_zh_a_spot_em 返回空数据")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容 AkShare 常见中文列名，整理出统一字段。
    """
    col_map = {}

    # 常见字段映射（按 AkShare 常见返回）
    for c in df.columns:
        if c in ("代码", "symbol"):
            col_map[c] = "code"
        elif c in ("名称", "name"):
            col_map[c] = "name"
        elif c in ("最新价", "close", "最新"):
            col_map[c] = "close"
        elif c in ("今开", "open"):
            col_map[c] = "open"
        elif c in ("最高", "high"):
            col_map[c] = "high"
        elif c in ("最低", "low"):
            col_map[c] = "low"
        elif c in ("涨跌幅", "pct_chg"):
            col_map[c] = "pct_change"
        elif c in ("涨跌额",):
            col_map[c] = "pct_amount"
        elif c in ("成交量", "volume"):
            col_map[c] = "volume"
        elif c in ("成交额", "amount", "turnover"):
            col_map[c] = "turnover"
        elif c in ("换手率", "turnover_rate"):
            col_map[c] = "turnover_rate"
        elif c in ("量比", "volume_ratio"):
            col_map[c] = "volume_ratio"
        elif c in ("市盈率-动态", "市盈率动态", "pe"):
            col_map[c] = "pe_dynamic"

    df = df.rename(columns=col_map)

    required = ["code", "name", "close", "pct_change", "turnover", "turnover_rate"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise RuntimeError(f"快照字段缺失，无法继续筛选。缺失字段: {missing}")

    # 数值化
    numeric_cols = [
        "close",
        "open",
        "high",
        "low",
        "pct_change",
        "pct_amount",
        "volume",
        "turnover",
        "turnover_rate",
        "volume_ratio",
        "pe_dynamic",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_num(df[col])

    return df


def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    用当日价格做一个近似版的短期趋势判断。
    由于这里只拿到快照数据，不直接有历史K线，
    为了在 GitHub Actions 里保持轻量，这里使用一个简化策略：
    - 仅使用现价 + 涨跌幅做粗筛
    - MA相关可以先留空，后续如果你愿意再升级成逐票历史K线版

    当前最小可用版：
    先用 close、涨跌幅、成交额、换手率、量比做第一轮。
    """
    # 先默认占位，便于后续升级
    df["ma5"] = pd.NA
    df["ma10"] = pd.NA
    df["ma20"] = pd.NA
    df["bias_ma5"] = pd.NA
    df["trend_ok"] = True  # 当前最小版先不做历史均线过滤，后续再升级
    return df


def filter_candidates(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    按你的风格做初筛。
    """
    data = df.copy()

    # 1) 基础过滤：去掉异常、ST、北交所（先简单处理）
    data = data.dropna(subset=["code", "name", "close", "pct_change", "turnover", "turnover_rate"])

    # 非 ST
    data = data[~data["name"].astype(str).str.contains("ST", na=False)]

    # 剔除北交所（以 8 / 4 开头的代码常见于北交所/新三板风格，先简单排）
    data = data[~data["code"].astype(str).str.startswith(("8", "4"))]

    # 2) 热度过滤
    # 成交额单位通常是元，这里按 5亿过滤
    data = data[data["turnover"] >= 5e8]

    # 换手率 >= 3%
    data = data[data["turnover_rate"] >= 3]

    # 3) 强度过滤
    # 涨幅 2% ~ 7%
    data = data[(data["pct_change"] >= 2) & (data["pct_change"] <= 7)]

    # 量比 >= 1.2（若无量比列，则放宽）
    if "volume_ratio" in data.columns:
        data = data[(data["volume_ratio"].isna()) | (data["volume_ratio"] >= 1.2)]

    # 4) 趋势过滤（当前最小版先用 trend_ok 占位，后续升级历史均线）
    if "trend_ok" in data.columns:
        data = data[data["trend_ok"] == True]  # noqa: E712

    # 5) 打分（先按你风格做轻量排序）
    # 更偏好：成交额大、涨幅适中、换手活跃、量比不低
    data["score"] = (
        data["pct_change"].clip(upper=7).fillna(0) * 8
        + data["turnover_rate"].clip(upper=20).fillna(0) * 1.5
        + data["turnover"].fillna(0).rank(pct=True) * 20
        + (data["volume_ratio"].fillna(1).clip(upper=3) * 5 if "volume_ratio" in data.columns else 0)
    )

    data = data.sort_values(by="score", ascending=False)

    # 6) 只取前 top_n
    result = data.head(top_n).copy()

    return result


def save_results(df: pd.DataFrame) -> None:
    """
    保存调试明细和最终候选代码列表。
    """
    if df.empty:
        OUTPUT_TXT.write_text("", encoding="utf-8")
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print("本次未筛出候选股。")
        return

    # 保存调试表
    debug_cols = [c for c in [
        "code", "name", "close", "pct_change", "turnover", "turnover_rate", "volume_ratio", "score"
    ] if c in df.columns]
    df[debug_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 保存代码列表（逗号分隔）
    codes = df["code"].astype(str).tolist()
    OUTPUT_TXT.write_text(",".join(codes), encoding="utf-8")

    print(f"已筛出 {len(codes)} 只候选股：")
    print(",".join(codes))


def main():
    print("开始生成今日候选池...")
    raw = load_spot_data()
    data = normalize_columns(raw)
    data = add_ma_features(data)
    result = filter_candidates(data, top_n=15)
    save_results(result)
    print(f"候选池文件已写入：{OUTPUT_TXT}")
    print(f"调试文件已写入：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
