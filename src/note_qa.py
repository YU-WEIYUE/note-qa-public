# src/note_qa.py —— W1：最小链路（读 md → 拼 prompt → 调 Ollama → 打印 JSON）
# 零依赖：只用标准库 urllib，不需要 venv / pip install。
import argparse
import json
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("corpus", nargs="?", type=Path)  # 兼容原来的位置参数
parser.add_argument("--corpus", dest="corpus_flag", type=Path)
args = parser.parse_args()
NOTE = args.corpus_flag or args.corpus or Path(__file__).resolve().parent.parent / "corpus" / "sample.md"
QUESTION = "手冲咖啡用什么水温？"                                      # ② 问它什么
MODEL = "qwen14b:latest"                                                 # ③ 本地模型
LINES = 500                                                            # ④ 只喂前 500 行

text = "\n".join(NOTE.read_text(encoding="utf-8").splitlines()[:LINES])  # ⑤ 取前 500 行

prompt = (                                                               # ⑥ 拼 prompt
    "你只能依据下面的笔记内容回答，不要使用你自己的知识。\n"
    f"笔记内容：\n{text}\n\n"
    f"用户问题：{QUESTION}\n\n"
    f'输出 JSON：{{"answer": "...", "sources": ["{NOTE.name}"]}}；'
    '笔记里找不到答案时，answer 写"笔记中没有相关信息"，sources 写 []。'
)

body = json.dumps({                                                      # ⑦ 调 Ollama
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "format": "json",        # 让模型只吐合法 JSON，而不是夹带解释文字
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:11434/api/generate",
    data=body,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=300) as r:
    raw = json.loads(r.read())["response"]   # ⑧ 模型原文藏在 response 字段里

out = json.loads(raw)                        # ⑨ 校验它确实是 JSON；不是就报错，别假装成功
print(json.dumps({"answer": out["answer"], "sources": out.get("sources", [])},
                 ensure_ascii=False))        # ⑩ 只打印那一行 JSON（这一行就是留档内容）
