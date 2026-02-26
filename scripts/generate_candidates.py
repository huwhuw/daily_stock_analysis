import os
import time
from pathlib import Path

import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

try:
    import tushare as ts
except ImportError:
    ts = None

try:
    import efinance as ef
except ImportError:
    ef = None


OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TXT = OUTPUT_DIR / "candidates.txt"
OUTPUT_CSV = OUTPUT_DIR / "candidates_debug.csv"


def safe_num(series: pd.Series) -> pd.Series:
    """尽量把字符串列安全转成数值列。"""
    return pd.to_numeric(series, errors="coerce")


def load_spot_data() -> pd.DataFrame:
    """
    多数据源兜底：
    1. AkShare 东方财富
    2. Tushare rt_k（需 TUSHARE_TOKEN 且有 rt_k 权限）
    3. efinance
    全部失败则返回空 DataFrame，不抛异常。
    """
    # 1) AkShare 东方财富
    if ak is not None:
        last_error = None
        for i in range(1):
            try:
                print(f"[数据源] 尝试 AkShare-东方财富，第 {i + 1} 次...")
                df = ak.stock_zh_a_spot_em()
                if df is not None and not df.empty:
                    print(f"[数据源] AkShare-东方财富 成功，记录数：{len(df)}")
                    df["_source"] = "akshare_em"
                    return df
            except Exception as e:
                last_error = e
                print(f"[数据源] AkShare-东方财富 失败：{e}")
                time.sleep(1)
        print(f"[数据源] AkShare-东方财富 最终失败：{last_error}")

    # 2) Tushare rt_k
    tushare_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if ts is not None and tushare_token:
        try:
            print("[数据源] 尝试 Tushare rt_k ...")
            pro = ts.pro_api(tushare_token)
            df = pro.rt_k(ts_code="3*.SZ,6*.SH,0*.SZ,9*.BJ")
            if df is not None and not df.empty:
                print(f"[数据源] Tushare rt_k 成功，记录数：{len(df)}")
                df["_source"] = "tushare_rt_k"
                return df
            print("[数据源] Tushare rt_k 返回空数据")
        except Exception as e:
            print(f"[数据源] Tushare rt_k 失败：{e}")

    # 3) efinance
    if ef is not None:
        try:
            print("[数据源] 尝试 efinance 实时行情 ...")
            df = ef.stock.get_realtime_quotes()
            if df is not None and not df.empty:
                print(f"[数据源] efinance 成功，记录数：{len(df)}")
                df["_source"] = "efinance"
                return df
            print("[数据源] efinance 返回空数据")
        except Exception as e:
            print(f"[数据源] efinance 失败：{e}")

    print("[数据源] 所有行情源都失败，返回空结果。")
    return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容 AkShare / Tushare / efinance 的列名，整理成统一字段：
    code, name, close, open, high, low, pct_change, turnover, turnover_rate, volume_ratio
    """
    if df is None or df.empty:
        print("[标准化] 输入为空，返回空 DataFrame")
        return pd.DataFrame()

    source = df["_source"].iloc[0] if "_source" in df.columns and not df.empty else "unknown"
    print(f"[标准化] 当前数据源：{source}")
    print(f"[标准化] 原始列名：{list(df.columns)}")

    data = df.copy()

    if source == "akshare_em":
        col_map = {
            "代码": "code",
            "名称": "name",
            "最新价": "close",
            "今开": "open",
            "最高": "high",
            "最低": "low",
            "涨跌幅": "pct_change",
            "涨跌额": "pct_amount",
            "成交量": "volume",
            "成交额": "turnover",
            "换手率": "turnover_rate",
            "量比": "volume_ratio",
            "市盈率-动态": "pe_dynamic",
            "市盈率动态": "pe_dynamic",
        }
        data = data.rename(columns=col_map)

    elif source == "tushare_rt_k":
        col_map = {
            "ts_code": "code",
            "name": "name",
            "close": "close",
            "open": "open",
            "high": "high",
            "low": "low",
            "vol": "volume",
            "amount": "turnover",
        }
        data = data.rename(columns=col_map)

        if "pre_close" in data.columns:
            data["pre_close"] = pd.to_numeric(data["pre_close"], errors="coerce")
            data["close"] = pd.to_numeric(data["close"], errors="coerce")
            data["pct_change"] = (data["close"] / data["pre_close"] - 1) * 100

        data["code"] = (
            data["code"].astype(str)
            .str.replace(".SH", "", regex=False)
            .str.replace(".SZ", "", regex=False)
            .str.replace(".BJ", "", regex=False)
        )

        data["turnover_rate"] = pd.NA
        data["volume_ratio"] = pd.NA
        data["pe_dynamic"] = pd.NA

    elif source == "efinance":
        col_map = {
            "股票代码": "code",
            "股票名称": "name",
            "最新价": "close",
            "今开": "open",
            "最高": "high",
            "最低": "low",
            "涨跌幅": "pct_change",
            "成交量": "volume",
            "成交额": "turnover",
            "换手率": "turnover_rate",
            "量比": "volume_ratio",
            "动态市盈率": "pe_dynamic",
        }
        data = data.rename(columns=col_map)

    else:
        raise RuntimeError(f"未知数据源，无法标准化：{source}")

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
        if col in data.columns:
            data[col] = safe_num(data[col])

    required = ["code", "name", "close", "pct_change", "turnover"]
    missing = [x for x in required if x not in data.columns]
    if missing:
        raise RuntimeError(f"标准化后字段缺失：{missing}")

    print(f"[标准化] 标准化后列名：{list(data.columns)}")
    print(f"[标准化] 标准化后记录数：{len(data)}")
    preview_cols = [c for c in ["code", "name", "close", "pct_change", "turnover", "turnover_rate", "volume_ratio"] if c in data.columns]
    print("[标准化] 前5条预览：")
    print(data[preview_cols].head(5).to_string(index=False))

    return data


def add_ma_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    当前最小可用版：先不做真实 MA 历史计算。
    先用占位字段，后续可升级成逐票历史K线版。
    """
    if df.empty:
        print("[趋势] 输入为空，跳过 MA 占位")
        return df

    df["ma5"] = pd.NA
    df["ma10"] = pd.NA
    df["ma20"] = pd.NA
    df["bias_ma5"] = pd.NA
    df["trend_ok"] = True
    print(f"[趋势] 已添加 MA 占位字段，记录数：{len(df)}")
    return df


def filter_candidates(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    按你的风格做初筛。
    """
    if df is None or df.empty:
        print("[筛选] 输入为空，直接返回空结果")
        return pd.DataFrame()

    data = df.copy()
    print(f"[筛选] 原始记录数：{len(data)}")

    # 基础过滤
    base_subset = ["code", "name", "close", "pct_change", "turnover"]
    data = data.dropna(subset=base_subset)
    print(f"[筛选] 去掉关键字段空值后：{len(data)}")

    # 非 ST
    data = data[~data["name"].astype(str).str.contains("ST", na=False)]
    print(f"[筛选] 去掉 ST 后：{len(data)}")

    # 剔除北交所（简单过滤）
    data = data[~data["code"].astype(str).str.startswith(("8", "4"))]
    print(f"[筛选] 去掉北交所后：{len(data)}")

    # 成交额 >= 5亿
    data = data[data["turnover"] >= 5e8]
    print(f"[筛选] 成交额>=5亿 后：{len(data)}")

    # 换手率 >= 3%（若无该列值，则放宽）
    if "turnover_rate" in data.columns:
        before = len(data)
        data = data[(data["turnover_rate"].isna()) | (data["turnover_rate"] >= 3)]
        print(f"[筛选] 换手率>=3%（空值放行）后：{len(data)}（剔除 {before - len(data)}）")
    else:
        print("[筛选] 无 turn_over_rate 列，跳过换手率过滤")

    # 涨幅 2% ~ 7%
    before = len(data)
    data = data[(data["pct_change"] >= 2) & (data["pct_change"] <= 7)]
    print(f"[筛选] 涨幅 2%~7% 后：{len(data)}（剔除 {before - len(data)}）")

    # 量比 >= 1.2（若无量比列，则放宽）
    if "volume_ratio" in data.columns:
        before = len(data)
        data = data[(data["volume_ratio"].isna()) | (data["volume_ratio"] >= 1.2)]
        print(f"[筛选] 量比>=1.2（空值放行）后：{len(data)}（剔除 {before - len(data)}）")
    else:
        print("[筛选] 无 volume_ratio 列，跳过量比过滤")

    # 趋势过滤（占位）
    if "trend_ok" in data.columns:
        before = len(data)
        data = data[data["trend_ok"] == True]  # noqa: E712
        print(f"[筛选] trend_ok 过滤后：{len(data)}（剔除 {before - len(data)}）")

    if data.empty:
        print("[筛选] 过滤后已无股票入围")
        return pd.DataFrame()

    # 打分
    turnover_rate_score = (
        data["turnover_rate"].fillna(3).clip(upper=20) * 1.5
        if "turnover_rate" in data.columns
        else 0
    )
    volume_ratio_score = (
        data["volume_ratio"].fillna(1).clip(upper=3) * 5
        if "volume_ratio" in data.columns
        else 0
    )

    data["score"] = (
        data["pct_change"].clip(upper=7).fillna(0) * 8
        + turnover_rate_score
        + data["turnover"].fillna(0).rank(pct=True) * 20
        + volume_ratio_score
    )

    print("[筛选] 打分后 Top 10 预览：")
    preview_cols = [c for c in ["code", "name", "pct_change", "turnover", "turnover_rate", "volume_ratio", "score"] if c in data.columns]
    print(data.sort_values(by="score", ascending=False)[preview_cols].head(10).to_string(index=False))

    data = data.sort_values(by="score", ascending=False)
    result = data.head(top_n).copy()

    print(f"[筛选] 最终入围 {len(result)} 只（top_n={top_n}）")
    if not result.empty:
        print("[筛选] 最终入围名单：")
        print(result[[c for c in ["code", "name", "score"] if c in result.columns]].to_string(index=False))

    return result


def save_results(df: pd.DataFrame) -> None:
    """
    保存调试明细和最终候选代码列表。
    """
    if df is None or df.empty:
        OUTPUT_TXT.write_text("", encoding="utf-8")
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print("[输出] 本次未筛出候选股。")
        return

    debug_cols = [
        c for c in [
            "code",
            "name",
            "close",
            "pct_change",
            "turnover",
            "turnover_rate",
            "volume_ratio",
            "score",
            "_source",
        ]
        if c in df.columns
    ]
    df[debug_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    codes = df["code"].astype(str).tolist()
    OUTPUT_TXT.write_text(",".join(codes), encoding="utf-8")

    print(f"[输出] 已筛出 {len(codes)} 只候选股：")
    print(",".join(codes))
    print(f"[输出] 候选池文件：{OUTPUT_TXT}")
    print(f"[输出] 调试文件：{OUTPUT_CSV}")


def main():
    print("开始生成今日候选池...")
    raw = load_spot_data()

    if raw is None or raw.empty:
        print("[主流程] 未获取到市场快照，本次候选池为空。")
        OUTPUT_TXT.write_text("", encoding="utf-8")
        pd.DataFrame().to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[主流程] 已写出空候选池文件：{OUTPUT_TXT}")
        print(f"[主流程] 已写出空调试文件：{OUTPUT_CSV}")
        return

    data = normalize_columns(raw)
    data = add_ma_features(data)
    result = filter_candidates(data, top_n=10)
    save_results(result)
    print("[主流程] 候选池生成完成。")


if __name__ == "__main__":
    main()
