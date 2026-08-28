# -*- coding: utf-8 -*-
"""
使用 Qwen3-VL（DashScope API）对滑坡遥感影像的 question/answer 文本做数据增强。

工作流程
--------
1. 扫描 text 文件夹中的 JSON 文件，读取 question / answer。
2. 按「文件名主名」在影像文件夹中匹配对应的 .png / .tif 影像。
3. 将影像 + 原始文本 + 增强提示词一起送入 Qwen3-VL。
4. 解析模型返回的增强后 question / answer，逐条写入 JSONL 训练集。

运行前准备
----------
    pip install openai pillow

环境变量
--------
    DASHSCOPE_API_KEY=sk-xxxxxxxx   # 阿里云百炼 API Key

用法
----
    python augment_qwen3vl.py
    python augment_qwen3vl.py --limit 5     # 先跑 5 条试一下
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time

from openai import OpenAI

# ============================================================================
# 配置区 —— 按你的实际情况修改
# ============================================================================

IMAGE_DIR = r"e:/笔记本备份/备份文件夹/我的科研/第三篇论文/colab/images"   # 影像文件夹
TEXT_DIR = r"e:/笔记本备份/备份文件夹/我的科研/第三篇论文/colab/texts"      # 文本(JSON)文件夹
OUTPUT_JSONL = r"e:/笔记本备份/备份文件夹/我的科研/第三篇论文/colab/augmented.jsonl"

MODEL = "qwen3-vl-plus"          # 也可改为 "qwen3-vl-235b-a22b-instruct" 或 "qwen-vl-max"
API_KEY_ENV = "DASHSCOPE_API_KEY"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 大图会等比缩放到该最大边长（像素），避免超长 token；滑坡细节需保留，建议 1024~2048
MAX_IMAGE_EDGE = 2048

# 请求间隔 / 重试设置
REQUEST_INTERVAL_SEC = 0.5
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 3.0

# 文本 JSON 中 question / answer 的字段名
QUESTION_KEY = "question"
ANSWER_KEY = "answer"

# ============================================================================
# 增强提示词（系统提示，原文保留）
# ============================================================================

SYSTEM_PROMPT = """You are a domain expert in geological hazard analysis and dataset engineering for Multimodal Large Models (MLLM).
- Your task is to rewrite a given question–answer pair into a standardized, machine-readable form that optimizes stable alignment between optical remote sensing imagery and textual descriptions of landslide features.
- The rewritten content is intended for multimodal model training. Prioritize structural consistency, lexical stability, and direct visual grounding over stylistic variation.

Semantic Preservation
- Strictly preserve all original factual content.
- Do NOT add, remove, infer, or reinterpret any information.
- Do NOT introduce causal, temporal, or predictive statements.

Visual Grounding
- Both the question and the answer MUST refer only to features that are directly observable in optical remote sensing imagery.
- Use only visually verifiable landslide characteristics, such as:
  scar zone, head scarp, debris accumulation, surface texture discontinuity, vegetation removal, displaced material, and landslide boundary.
- Do NOT reference non-visual factors (e.g., triggering mechanisms, material properties, or subsurface processes).

Structured Answer Format
- The answer MUST consist of concise declarative sentences.
- Each sentence MUST correspond to one semantic unit.
- Use the following fixed semantic order whenever applicable:
  Sentence 1: Existence or spatial position of the landslide feature.
  Sentence 2: Observable geomorphic components (e.g., scar zone, debris accumulation).
  Sentence 3: Surface texture characteristics and vegetation condition.
- If a semantic unit is not present in the original answer, omit that sentence entirely.
- Do NOT merge semantic units into a single sentence.

Lexical Standardization
- Use consistent, canonical terminology across all samples.
- Prefer the following standardized terms when applicable:
  “landslide”, “scar zone”, “head scarp”, “debris accumulation”,
  “surface texture discontinuity”, “exposed substrate”, “vegetation removal”.
- Avoid synonyms if a standard term exists.

Language Constraints
- Do NOT use first-person pronouns.
- Do NOT use modal verbs or speculative expressions.
- Use declarative, objective sentences only.
- Maintain a neutral, technical register suitable for scientific datasets.

Geographic Anonymity
- The imagery originates from a known study region.
- Do NOT mention or imply any geographic names, locations, or regional identifiers.

Output Format
- Return ONLY a valid JSON object with exactly two keys: "question" and "answer".
- The "question" value is the rewritten question; the "answer" value is the rewritten answer.
- Do NOT include any text outside the JSON object."""


# ============================================================================
# 工具函数
# ============================================================================

def read_image_as_base64_data_url(path):
    """读取影像并返回 (mime, data_url)。

    Qwen-VL 支持 PNG/JPEG/WEBP 的 base64；TIFF 等其它格式先用 Pillow 转成 PNG。
    大图会等比缩放到 MAX_IMAGE_EDGE。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg", ".webp") and not _need_downscale(path):
        mime = {".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp"}[ext]
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return mime, f"data:{mime};base64,{b64}"

    # 其它格式（含 .tif）或需要缩放：统一用 Pillow 转 PNG
    from PIL import Image
    img = Image.open(path)
    img = _maybe_downscale(img)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "image/png", f"data:image/png;base64,{b64}"


def _need_downscale(path):
    # 快速用 Pillow 判断尺寸是否需要缩放；仅当边长超过阈值才进入慢路径
    try:
        from PIL import Image
        with Image.open(path) as im:
            return max(im.size) > MAX_IMAGE_EDGE
    except Exception:
        return True  # 打不开就走 Pillow 统一处理，由它抛出更明确的报错


def _maybe_downscale(img):
    w, h = img.size
    if max(w, h) <= MAX_IMAGE_EDGE:
        return img
    scale = MAX_IMAGE_EDGE / float(max(w, h))
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size)


def build_image_index(image_dir):
    """返回 {文件主名: 影像完整路径}，支持 .png/.jpg/.jpeg/.webp/.tif/.tiff。"""
    exts = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
    index = {}
    if not os.path.isdir(image_dir):
        return index
    for name in os.listdir(image_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in exts:
            index.setdefault(stem, os.path.join(image_dir, name))  # 同名多格式取第一个
    return index


def extract_json(text):
    """从模型输出中稳健地提取 JSON 对象。"""
    if not text:
        return None
    text = text.strip()

    # 1) 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) 去掉 ```json ... ``` 代码围栏
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    # 3) 取首个 { 到最后一个 } 的片段
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass

    return None


def call_qwen3vl(client, mime, data_url, question, answer):
    """调用 Qwen3-VL，返回 (增强后 question, 增强后 answer)。失败抛异常。"""
    user_text = (
        "Rewrite the following question–answer pair according to the rules.\n\n"
        f"Original question:\n{question}\n\n"
        f"Original answer:\n{answer}"
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        temperature=0.1,  # 低温度，保证结构一致性
    )
    raw = completion.choices[0].message.content
    parsed = extract_json(raw)
    if not parsed:
        raise ValueError(f"无法解析模型输出为 JSON: {raw[:200]!r}")
    new_q = parsed.get(QUESTION_KEY, "").strip()
    new_a = parsed.get(ANSWER_KEY, "").strip()
    if not new_q or not new_a:
        raise ValueError(f"输出缺少 question/answer 字段: {raw[:200]!r}")
    return new_q, new_a


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 滑坡文本数据增强")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条（用于测试），默认全部")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已处理的样本，重新生成")
    args = parser.parse_args()

    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        sys.exit(f"[错误] 未找到环境变量 {API_KEY_ENV}，请先设置：\n"
                 f"  Windows PowerShell: $env:{API_KEY_ENV}='sk-xxx'\n"
                 f"  CMD: set {API_KEY_ENV}=sk-xxx")

    if not os.path.isdir(TEXT_DIR):
        sys.exit(f"[错误] 文本文件夹不存在: {TEXT_DIR}")
    if not os.path.isdir(IMAGE_DIR):
        sys.exit(f"[错误] 影像文件夹不存在: {IMAGE_DIR}")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    image_index = build_image_index(IMAGE_DIR)
    if not image_index:
        sys.exit(f"[错误] 影像文件夹中没有找到 .png/.jpg/.tif 等影像: {IMAGE_DIR}")

    # 收集文本 JSON 文件
    text_files = sorted(
        os.path.join(TEXT_DIR, n) for n in os.listdir(TEXT_DIR)
        if n.lower().endswith(".json")
    )
    if not text_files:
        sys.exit(f"[错误] 文本文件夹中没有找到 .json 文件: {TEXT_DIR}")
    if args.limit:
        text_files = text_files[: args.limit]

    # 断点续跑：读取已处理的文件主名
    done = set()
    if not args.no_resume and os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    stem = json.loads(line).get("stem")
                    if stem:
                        done.add(stem)
                except Exception:
                    pass
        print(f"[续跑] 已跳过 {len(done)} 个已处理样本")

    out_dir = os.path.dirname(OUTPUT_JSONL)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    ok, skip_no_img, fail = 0, 0, 0
    with open(OUTPUT_JSONL, "a", encoding="utf-8") as out_f:
        for i, tf in enumerate(text_files, 1):
            stem = os.path.splitext(os.path.basename(tf))[0]
            if stem in done:
                continue

            # 读文本
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                question = data[QUESTION_KEY].strip()
                answer = data[ANSWER_KEY].strip()
            except Exception as e:
                print(f"[{i}/{len(text_files)}] {stem}: 读取文本失败 ({e})，跳过")
                fail += 1
                continue

            # 匹配影像
            img_path = image_index.get(stem)
            if img_path is None:
                print(f"[{i}/{len(text_files)}] {stem}: 未找到同名影像，跳过")
                skip_no_img += 1
                continue

            # 调用模型（带重试）
            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    mime, data_url = read_image_as_base64_data_url(img_path)
                    new_q, new_a = call_qwen3vl(client, mime, data_url, question, answer)
                    result = (new_q, new_a)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        wait = RETRY_BACKOFF_SEC * attempt
                        print(f"  {stem}: 第 {attempt} 次失败 ({e})，{wait:.0f}s 后重试...")
                        time.sleep(wait)
                    else:
                        print(f"[{i}/{len(text_files)}] {stem}: 重试耗尽，失败 ({e})")
                        fail += 1

            if result is None:
                continue

            new_q, new_a = result
            record = {
                "stem": stem,
                "image": img_path,
                "question": new_q,
                "answer": new_a,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            ok += 1
            print(f"[{i}/{len(text_files)}] {stem}: 完成")

            time.sleep(REQUEST_INTERVAL_SEC)

    print("\n==== 完成 ====")
    print(f"成功: {ok}  无同名影像跳过: {skip_no_img}  失败: {fail}")
    print(f"输出文件: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
