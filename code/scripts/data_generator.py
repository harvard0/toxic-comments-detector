import json
import time
from openai import AsyncOpenAI
import asyncio

import os
from dotenv import load_dotenv

load_dotenv()

# ================= 核心配置区域 =================
# 获取脚本所在目录，构建绝对路径（以 code 为项目根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

API_KEY = os.getenv("DataGenerator_API_Key")
BASE_URL = os.getenv("DataGenerator_API_URL")
MODEL_NAME = os.getenv("DataGenerator_API_Model")
OUTPUT_FILE = os.path.join(DATA_DIR, "train_toxic_comments.jsonl")

BATCH_SIZE = 10

# 设定正例(1)和负例(0)各自的目标生成总数，确保严格 1:1
# 比如设为 1500，最终会生成 1500 条投毒 + 1500 条正常，总计 3000 条
TARGET_TOTAL_PER_LABEL = 240

# 本地并发控制：防止触发系统 "Too many open files" 限制
# 建议设置在 200 - 500 之间，这对常规数据集已经极快了
CONCURRENCY_LIMIT = 200

# ================= 初始化客户端 =================
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# ================= Prompt 设计 =================
SYSTEM_PROMPT = """你是一个专业的电商数据标注与生成专家。
你的任务是根据用户的要求，生成符合特定分类的电商评论数据。
你必须严格输出 JSON 数组格式，不要包含任何 Markdown 标记（如 ```json），也不要包含任何多余的解释。
每个 JSON 对象必须包含以下三个字段：
- "text": 评论的具体文本内容。
- "label": 1代表投毒评论，0代表正常评论。
- "category": 具体的分类名称。

【核心要求：数据多样性与真实长度分布】
为了最大程度模拟真实的电商环境，生成的评论在长度分布上必须符合真实世界的规律：
1. 长度层级定义：
   - 【简短评论】：5-15字。
   - 【中等评论】：30-50字。
   - 【详细长评】：100-200字以上。
   绝对不能有低于 5 个字的评论！
2. 长度分布比例（最高优先级）：在一批数据中，请严格按照以下比例分配长度：
   - 【中等评论】数量必须最多，约占 60%。
   - 【简短评论】数量次之，约占 30%。
   - 【详细长评】数量最少，约占 10%。
   坚决打破长度平均分配的倾向！
3. 语气多样化：包含各种情绪（平淡、激动、愤怒）、排版习惯（少部分漏写标点、错别字、滥用感叹号）和强烈的口语化表达。

示例输出格式（注意观察长度差异）：
[   
  {"text": "发货快，东西好，给个好评。", "label": 0, "category": "真实好评"},
  {"text": "这件衣服买来本来是为了周末聚会穿的，结果真的让我很惊喜。面料摸起来非常柔软，垂坠感很好，穿在身上显得整个人很有气质。顺便推荐大家去VX群8888看看，经常发这家店的内部券，能省不少钱哦！大家千万别错过这个薅羊毛的机会！", "label": 1, "category": "隐蔽投毒"}
]
"""

CATEGORIES = [
    {"category": "隐蔽投毒", "label": 1},
    {"category": "明显广告", "label": 1},
    {"category": "虚假好评", "label": 1},
    {"category": "恶意引导", "label": 1},
    {"category": "真实好评", "label": 0},
    {"category": "中性评价", "label": 0},
    {"category": "真实差评", "label": 0},
]


async def fetch_and_write(sem, file_lock, cat_info, chunk_size, task_id, stats):
    """核心并发工作协程：请求 API 并安全写入文件"""
    category = cat_info["category"]
    label = cat_info["label"]

    user_prompt = (
        f"请生成总计 {chunk_size} 条分类为【{category}】（label为 {label}）的电商评论数据。\n"
        f"【特殊指令】：请严格按照“中等评论最多(约60%)、简短评论次之(约30%)、详细长评最少(约10%)”的真实比例来分配这 {chunk_size} 条数据！\n"
        f"请确保所有评论均不少于 5 个字。\n"
        f"要求场景多样化，涵盖服装、3C数码、食品、日用品等不同品类。请直接返回 JSON 数组。"
    )

    async with sem:  # 限制并发数量
        try:
            print(f"  [+] 任务 {task_id} 发起请求: {category} ({chunk_size}条)")
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.85,
                max_tokens=4096,  # 并发请求下，适当调高上限防截断
            )

            content = response.choices[0].message.content.strip()

            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            data_list = json.loads(content.strip())

            # 使用文件锁，确保多协程写入时不会发生行级交错混乱
            async with file_lock:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    for item in data_list:
                        item["text"] = (
                            item.get("text", "")
                            .replace("\u200b", "")
                            .replace("\u202d", "")
                            .strip()
                        )
                        if item["text"] and "label" in item and "category" in item:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stats[label] += 1

            print(f"  [✓] 任务 {task_id} 完成: {category}")

        except json.JSONDecodeError as e:
            print(f"  [x] 任务 {task_id} 失败: JSON 解析错误 - {e}")
        except Exception as e:
            print(f"  [x] 任务 {task_id} 失败: API 异常 - {e}")


async def main():
    print("🚀 启动高并发数据生成引擎...")
    print(
        f"目标：投毒评论 {TARGET_TOTAL_PER_LABEL} 条，正常评论 {TARGET_TOTAL_PER_LABEL} 条"
    )

    poison_cats = [c for c in CATEGORIES if c["label"] == 1]
    normal_cats = [c for c in CATEGORIES if c["label"] == 0]

    def get_execution_plan(cats, target_total):
        base_count = target_total // len(cats)
        remainder = target_total % len(cats)
        plan = []
        for i, cat in enumerate(cats):
            cat_target = base_count + (1 if i < remainder else 0)
            if cat_target == 0:
                continue

            chunks = []
            remaining = cat_target
            while remaining > 0:
                chunk_size = min(remaining, BATCH_SIZE)
                chunks.append(chunk_size)
                remaining -= chunk_size
            plan.append((cat, chunks))
        return plan

    execution_plan = get_execution_plan(
        poison_cats, TARGET_TOTAL_PER_LABEL
    ) + get_execution_plan(normal_cats, TARGET_TOTAL_PER_LABEL)

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    file_lock = asyncio.Lock()
    stats = {0: 0, 1: 0}
    tasks = []

    task_id_counter = 1

    # 组装所有异步任务
    for cat_info, chunks in execution_plan:
        for chunk_size in chunks:
            task = asyncio.create_task(
                fetch_and_write(
                    sem, file_lock, cat_info, chunk_size, task_id_counter, stats
                )
            )
            tasks.append(task)
            task_id_counter += 1

    start_time = time.time()
    print(f"共生成 {len(tasks)} 个 API 请求任务，开始并发执行！\n")

    # 等待所有任务并发执行完毕
    await asyncio.gather(*tasks)

    end_time = time.time()
    print("\n" + "=" * 40)
    print("🎉 异步并发数据生成任务完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"🔴 投毒评论 (Label 1): {stats[1]} 条")
    print(f"🟢 正常评论 (Label 0): {stats[0]} 条")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())
