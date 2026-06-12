import pandas as pd
import json
import os

# ================= 核心配置 =================
# 获取脚本所在目录，构建绝对路径（以 code 为项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 定义三个数据集的文件路径
DATASETS = {
    "Train ": os.path.join(DATA_DIR, "train_toxic_comments.jsonl"),
    "Valid ": os.path.join(DATA_DIR, "valid_toxic_comments.jsonl"),
    "Test  ": os.path.join(DATA_DIR, "test_toxic_comments.jsonl")
}

def load_data(file_path):
    """加载数据，如果文件不存在则返回 None"""
    if not os.path.exists(file_path):
        return None
    return pd.read_json(file_path, lines=True)

def main():
    print("🚀 开始加载数据集...\n")
    
    # 存储读取到的 DataFrame
    dfs = {}
    total_samples = 0
    
    for name, path in DATASETS.items():
        df = load_data(path)
        if df is not None:
            dfs[name] = df
            count = len(df)
            total_samples += count
            print(f"✅ 加载 {name} 成功: {count} 条")
        else:
            print(f"❌ 警告: 找不到文件 {path}，跳过此数据集")
            
    if not dfs:
        print("\n❌ 错误: 没有找到任何有效的数据集，程序退出。")
        return

    print(f"\n✅ 数据集加载完毕，总计 {len(dfs)} 个数据集，共 {total_samples} 条样本。")
    print("=" * 60)

    # ================= 1. 整体数量对比 =================
    print("\n📊 1. 数据集总体规模对比")
    size_stats = pd.DataFrame([{"数据集": name, "样本数": len(df)} for name, df in dfs.items()])
    # 计算比例
    size_stats['占比'] = (size_stats['样本数'] / total_samples * 100).round(2).astype(str) + '%'
    print("\n" + size_stats.to_string(index=False))
    print("-" * 60)

    # ================= 2. Label 横向对比 =================
    print("\n📊 2. 正负例分布 (Label 比例) 综合对比")
    # 收集每个数据集的 Label 比例
    label_ratios_list = []
    for name, df in dfs.items():
        # normalize=True 直接计算比例
        ratios = df['label'].value_counts(normalize=True) * 100
        ratios.name = name
        label_ratios_list.append(ratios)
        
    # 合并成一个 DataFrame
    label_stats = pd.concat(label_ratios_list, axis=1).fillna(0).round(2)
    # 重命名索引，使其更直观
    label_stats.index = label_stats.index.map({1: '🔴 投毒 (Label 1)', 0: '🟢 正常 (Label 0)'})
    # 加上 % 号
    label_stats = label_stats.astype(str) + '%'
    print("\n" + label_stats.to_string())
    print("-" * 60)

    # ================= 3. Category 横向对比 =================
    print("\n📊 3. 细分种类分布 (Category 比例) 综合对比")
    
    cat_ratios_list = []
    label_mapping_dict = {}
    
    for name, df in dfs.items():
        # 统计比例
        ratios = df['category'].value_counts(normalize=True) * 100
        ratios.name = name
        cat_ratios_list.append(ratios)
        
        # 提取类别对应的 Label 用于展示 (任意一个 df 即可，这里我们合并所有找到的 mapping)
        mapping = df.drop_duplicates('category').set_index('category')['label']
        for cat, lbl in mapping.items():
             label_mapping_dict[cat] = '🔴 投毒' if lbl == 1 else '🟢 正常'

    # 合并类别统计
    cat_stats = pd.concat(cat_ratios_list, axis=1).fillna(0).round(2)
    
    # 插入所属大类作为第一列
    cat_stats.insert(0, '所属大类', cat_stats.index.map(label_mapping_dict))
    
    # 将后面的数值列加上 % 号
    for col in cat_stats.columns[1:]:
        cat_stats[col] = cat_stats[col].astype(str) + '%'
        
    print("\n" + cat_stats.to_string())
    print("\n" + "=" * 60)
    print("✨ 统计完成！")

if __name__ == "__main__":
    main()