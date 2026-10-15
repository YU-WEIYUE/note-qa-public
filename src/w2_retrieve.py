# src/w2_retrieve.py —— W2：接入检索（chunk → embed → top-k）
# 零依赖：只用标准库 urllib，不需要 venv / pip install。
# 与 W1 的差别：W1 把「整份 md 前 500 行」全塞进 prompt；W2 先只取**相关的几块**再喂模型。
#
# 前置：ollama pull shaw/dmeta-embedding-zh     <- 中文向量模型（408MB）
#       （nomic-embed-text 是英文模型，在中文上几乎不判别，已弃用）
# 本次实测结论（同一份库、同一批问题）：
#   ① 模型：nomic 换 dmeta —— 8 句无关中文的平均相似度 0.635 -> 0.355，查询才能命中
#   ② 块大小：MAX_CHARS=300 时正确块排第 1；500/800 时掉到第 24/29 名 —— 块太大 = 稀释
#   ③ 同一句答案单独成块得 0.7346，被包进 1200 字块后只剩 0.4209
import argparse
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # ① 防 Windows GBK 控制台报错

NOTE = Path(__file__).resolve().parent.parent / "corpus" / "sample.md"  # ② 默认公开样例语料
EMBED_MODEL = "shaw/dmeta-embedding-zh"                                 # ③ 中文向量模型（和生成模型是两个）
K = 3                                                                   # ④ 取回几块
QUESTIONS = [                                                           # ⑤ 三个样例问题：两有一无
    "手冲咖啡用什么水温？",                       # 样例库里有
    "冷萃需要冷藏多久？",                         # 样例库里有
    "泡茶水温是多少？",                           # 样例库里没有
]

# ── 两个实验开关（做对照实验就是改这两个值）────────────────────────────
USE_PREFIX = False     # ① nomic 系列要求给「文档 / 查询」分别加任务前缀；dmeta 不需要
MAX_CHARS = 300        # ② 一块最多多少字，优先按行切；写 0 = 不二次切分
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def embed(text: str, is_query: bool) -> list[float]:
    """调 Ollama 拿一段文字的向量。新版是 /api/embed，老版是 /api/embeddings，两个都试。"""
    if USE_PREFIX and "nomic" in EMBED_MODEL:            # ③ 任务前缀：文档与查询要用不同的前缀
        text = (QUERY_PREFIX if is_query else DOC_PREFIX) + text
    body = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["embeddings"][0]
    except urllib.error.HTTPError:                       # 老版 Ollama 回退
        body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
        req = urllib.request.Request("http://127.0.0.1:11434/api/embeddings", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["embedding"]


def cut_long(title: str, body: str) -> list[tuple[str, str]]:
    """优先按行切块；超长单行才在单元格或句子边界附近切开。"""
    if MAX_CHARS <= 0 or len(body) <= MAX_CHARS:
        return [(title, body)]

    pieces, buf = [], ""
    for line in body.splitlines(keepends=True):
        if len(line) <= MAX_CHARS:
            if buf and len(buf) + len(line) > MAX_CHARS:
                pieces.append(buf)
                buf = ""
            buf += line
            continue
        if buf:
            pieces.append(buf)
            buf = ""
        while len(line) > MAX_CHARS:
            stops = [i + 1 for i, ch in enumerate(line[:MAX_CHARS])
                     if ch in "|。；，、;,.!?！？ \t"]
            end = stops[-1] if stops and stops[-1] >= MAX_CHARS // 2 else MAX_CHARS
            pieces.append(line[:end])
            line = line[end:]
        buf = line
    if buf:
        pieces.append(buf)
    if len(pieces) == 1:
        return [(title, pieces[0])]
    return [(f"{title} #{i}", p) for i, p in enumerate(pieces, 1)]


def split_chunks(text: str) -> list[tuple[str, str]]:
    """把 md 按 '## ' 标题切块，再对超长块做二级切分 -> [(标题, 正文), ...]。块 = 检索的最小单位。"""
    raw, title, buf = [], "(开头)", []
    for line in text.splitlines():
        if line.startswith("## "):
            if buf:
                raw.append((title, "\n".join(buf)))
            title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    if buf:
        raw.append((title, "\n".join(buf)))
    chunks = []
    for t, b in raw:
        if b.strip():
            chunks.extend(cut_long(t, b.strip()))        # ⑧ 丢掉空块 + 二级切分
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)                               # ⑨ 两向量都非零，不用防零除


def main() -> None:
    ap = argparse.ArgumentParser(description="W2：chunk → embed → top-k 演示")
    ap.add_argument("--corpus", type=Path, default=NOTE,
                    help="资料库 md（默认=corpus/sample.md）")
    note = ap.parse_args().corpus
    chunks = split_chunks(note.read_text(encoding="utf-8"))
    lens = [len(b) for _, b in chunks]
    print(f"[库] {note.name} -> {len(chunks)} 块（{min(lens)}~{max(lens)} 字）"
          f" | 前缀={USE_PREFIX} MAX_CHARS={MAX_CHARS} 模型={EMBED_MODEL}")
    vecs = [embed(t + "\n" + b, is_query=False) for t, b in chunks]   # ⑩ 每块一个向量

    sims = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    print(f"[判别力] 块-块平均相似度 = {statistics.mean(sims):.4f}"     # ⑪ 越接近 1 = 越分不开
          f"（{min(sims):.4f}~{max(sims):.4f}）")

    for q in QUESTIONS:
        qv = embed(q, is_query=True)
        ranked = sorted(((cosine(qv, v), i) for i, v in enumerate(vecs)), reverse=True)
        print(f"\n[问] {q}")
        print("  全体得分: " + " ".join(f"{s:.3f}" for s, _ in ranked))
        for rank, (score, i) in enumerate(ranked[:K], 1):
            title, body = chunks[i]
            head = " ".join(body.split())[:40]           # ⑫ 只印前 40 字，看得清就行
            print(f"  top{rank}  {score:.4f}  【{title}】 {head}...")


if __name__ == "__main__":
    main()
