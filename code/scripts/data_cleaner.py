import pandas as pd
import re
import os

# ================= 核心配置 =================
# 你需要清理的数据集文件列表
FILES_TO_CLEAN = [
    "../data/train_toxic_comments.jsonl",
    "../data/valid_toxic_comments.jsonl",
    "../data/test_toxic_comments.jsonl",
]


def remove_ai_artifacts(text):
    """清理大模型常见的“幻觉”前缀、机械性结尾和无用废话"""
    if not isinstance(text, str):
        return ""
    prefix_patterns = [
        r"^(好的|没问题|当然|以下是|为您生成|作为.*?(AI|语言模型)).*?[：,，]\s*",
        r"^(这是一条|这是一段).*?[：,，]\s*",
    ]
    for p in prefix_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE)

    suffix_patterns = [
        r"(综上所述|总而言之|总结来说).*$",
        r"希望(这个|这些)(评论|建议)对您有帮助。?$",
    ]
    for p in suffix_patterns:
        text = re.sub(p, "", text)

    return text.strip()


def clean_single_dataset(file_path):
    """对单个数据集执行完整的清洗流水线"""
    if not os.path.exists(file_path):
        print(f"⚠️ 跳过: 找不到文件 '{file_path}'")
        return None

    # 动态生成输出文件名 (例如: train_cleaned.jsonl 和 train_garbage.jsonl)
    base_name, ext = os.path.splitext(file_path)
    output_good = f"{base_name}_cleaned{ext}"
    output_bad = f"{base_name}_garbage{ext}"

    print(f"\n⏳ 正在清洗: {file_path} ...")
    df = pd.read_json(file_path, lines=True)
    initial_count = len(df)
    garbage_bins = []

    # 1. 基础清理与 AI 痕迹剥离
    df["original_text"] = df["text"]
    df["text"] = (
        df["text"].astype(str).str.replace(r"[\u200b\u202d\uFEFF]", "", regex=True)
    )
    df["text"] = df["text"].apply(remove_ai_artifacts)

    # 2. 极端长度过滤 (5 - 300字)
    valid_length_mask = (df["text"].str.len() >= 5) & (df["text"].str.len() <= 300)
    df_bad_len = df[~valid_length_mask].copy()
    if not df_bad_len.empty:
        df_bad_len["reject_reason"] = "长度异常 (<5 或 >300)"
        garbage_bins.append(df_bad_len)
    df = df[valid_length_mask]

    # 3. 强力去重
    # A. 完全精确去重
    exact_dup_mask = df.duplicated(subset=["text"], keep="first")
    df_bad_exact_dup = df[exact_dup_mask].copy()
    if not df_bad_exact_dup.empty:
        df_bad_exact_dup["reject_reason"] = "完全重复"
        garbage_bins.append(df_bad_exact_dup)
    df = df[~exact_dup_mask]

    # B. 归一化去重
    df["normalized_text"] = (
        df["text"]
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    norm_dup_mask = df.duplicated(subset=["normalized_text"], keep="first")
    df_bad_norm_dup = df[norm_dup_mask].copy()
    if not df_bad_norm_dup.empty:
        df_bad_norm_dup["reject_reason"] = "高度相似(归一化重复)"
        garbage_bins.append(df_bad_norm_dup)
    df = df[~norm_dup_mask].drop(columns=["normalized_text"])

    # 4. 保存文件
    df[["text", "label", "category"]].to_json(
        output_good, orient="records", lines=True, force_ascii=False
    )

    bad_count = 0
    if garbage_bins:
        df_garbage = pd.concat(garbage_bins)
        df_garbage[["original_text", "label", "category", "reject_reason"]].to_json(
            output_bad, orient="records", lines=True, force_ascii=False
        )
        bad_count = len(df_garbage)

    final_count = len(df)
    print(f"   ✅ 完成! 保留 {final_count} 条，剔除 {bad_count} 条。")

    return {
        "Dataset": file_path,
        "Original": initial_count,
        "Cleaned (保留)": final_count,
        "Garbage (剔除)": bad_count,
        "Survival Rate": f"{(final_count / initial_count * 100):.2f}%"
        if initial_count > 0
        else "0%",
    }


def main():
    print("🚀 开始执行批量数据清洗流水线...\n")
    print("=" * 60)

    stats_list = []
    for file_path in FILES_TO_CLEAN:
        stats = clean_single_dataset(file_path)
        if stats:
            stats_list.append(stats)

    if not stats_list:
        print("\n❌ 没有找到任何可清理的文件。")
        return

    # ================= 打印全局汇总统计 =================
    print("\n" + "=" * 60)
    print("✨ 批量清洗任务全部圆满完成！全局统计报告：\n")

    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False))

    total_original = stats_df["Original"].sum()
    total_cleaned = stats_df["Cleaned (保留)"].sum()
    total_garbage = stats_df["Garbage (剔除)"].sum()

    print("-" * 60)
    print(
        f"📦 汇总数据: 初始共 {total_original} 条 | 最终保留 {total_cleaned} 条 | 总计剔除 {total_garbage} 条"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
