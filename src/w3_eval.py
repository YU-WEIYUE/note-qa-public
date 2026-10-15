# src/w3_eval.py —— W3：评测集（检索层 + 生成层，两把尺子）
# 零依赖：只用标准库 urllib（不需要 venv / pip），检索模型与生成模型都走本地 Ollama。
#
# 它量的是两件事（对应主课 11/10 的三分类表）：
#   ① 检索层：正确那块有没有进 top-k        -> recall@k
#   ② 生成层：答案里的 [Source N] 对不对     -> 引用准确率；库里没答案的题它认不认得出来 -> 拒答率
#
# 前置：ollama serve 在跑；模型 shaw/dmeta-embedding-zh（向量）+ qwen14b:latest（生成）都在。
# 跑法（留档就用这条）：python src/w3_eval.py | tee 留档-w3-YYYY-MM-DD.txt
#   换资料库（样例语料，任何人拿到仓库都能跑）：
#     python src/w3_eval.py --corpus corpus/sample.md --eval-set src/eval_set.sample.json
#   --no-gen   只跑检索层（快，约 10 秒）
#   --limit 3  只跑前 3 题（改代码后先冒烟测试用）
import argparse
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # ① 防 Windows GBK 控制台报错

HERE = Path(__file__).resolve().parent                     # ...\note-qa\src
CORPUS = HERE.parent / "corpus" / "sample.md"               # ② 公开样例语料（与 W1/W2 同源）
EVAL_SET = HERE / "eval_set.sample.json"                   # ③ 8 题公开样例集
EMBED_MODEL = "shaw/dmeta-embedding-zh"                    # ④ 向量模型（W2 实测选出来的中文模型）
GEN_MODEL = "qwen14b:latest"                               # ⑤ 生成模型
MAX_CHARS = 300                                            # ⑥ W2 定稿的块大小：300 时正确块排第 1
RECALL_KS = (1, 3, 5)                                      # ⑦ 报三个 k
GEN_TOP_K = 3                                              # ⑧ 喂给生成模型的块数
NO_ANS = "笔记中没有相关信息"                                 # ⑨ 库外题的期望回答（逐字契约）
MARKER = re.compile(r"\[Source (\d+)\]")                    # ⑩ 引用标记契约；下游靠这条正则数引用


def post(url: str, body: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def embed(text: str) -> list[float]:
    """调 Ollama 拿一段文字的向量。新版 /api/embed，老版 /api/embeddings 回退。"""
    try:
        return post("http://127.0.0.1:11434/api/embed",
                    {"model": EMBED_MODEL, "input": text}, 120)["embeddings"][0]
    except urllib.error.HTTPError:
        return post("http://127.0.0.1:11434/api/embeddings",
                    {"model": EMBED_MODEL, "prompt": text}, 120)["embedding"]


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
    """按 '## ' 标题切块 + 二级切分 -> [(标题, 正文), ...]。"""
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
            chunks.extend(cut_long(t, b.strip()))
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def wilson(hits: int, n: int) -> tuple[float, float]:
    """Wilson 95% 置信区间。n=20 太小，只报点估计就是自欺 —— 所以每个比例都配区间。"""
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(hits: int, n: int) -> str:
    lo, hi = wilson(hits, n)
    return f"{hits}/{n} ({100 * hits / n:.1f}%, 95%CI {100 * lo:.1f}%~{100 * hi:.1f}%)" if n else "n/a"


def gold_hits(chunks: list[tuple[str, str]], phrases: list[str]) -> set[int]:
    """哪些块里逐字含该题的全部 gold 短语 = 该题的『正确答案所在块』集合。"""
    return {i for i, (t, b) in enumerate(chunks) if all(p in b or p in t for p in phrases)}


def ask_model(question: str, ctx: list[tuple[int, str, str]]) -> dict:
    """生成层：把 top-k 块按 [Source N] 编号喂进去，要求答案自带同一个编号。"""
    blocks = "\n\n".join(f"[Source {n}] {body}" for n, _, body in ctx)
    prompt = (
        "你只能依据下面的资料片段回答，不要使用你自带的常识。\n"
        f"资料片段：\n{blocks}\n\n"
        f"问题：{question}\n\n"
        "要求：\n"
        "1. 答案里凡是用到某个片段，就在那句话末尾标出它的编号，格式固定为 [Source N]"
        "（英文方括号 + 英文 Source + 空格 + 数字）。\n"
        f"2. 资料片段里找不到答案时，answer 逐字写“{NO_ANS}”，sources 写 []。\n"
        '3. 只输出 JSON：{"answer": "...", "sources": ["Source 1"]}\n'
    )
    raw = post("http://127.0.0.1:11434/api/generate",
               {"model": GEN_MODEL, "prompt": prompt, "stream": False,
                "format": "json", "options": {"temperature": 0}}, 300)["response"]
    out = json.loads(raw)                                  # ⑪ 不是 JSON 就当场炸，别假装成功
    answer = str(out.get("answer", ""))
    marked = sorted({int(n) for n in MARKER.findall(answer)})        # ⑫ 标记：从答案原文里数（单一真相源）
    field = sorted({int(n) for n in MARKER.findall(" ".join(map(str, out.get("sources", []))))}
                   | {int(m) for m in re.findall(r"\d+", json.dumps(out.get("sources", []), ensure_ascii=False))})
    return {"answer": answer, "marked": marked, "field": field}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gen", action="store_true", help="只跑检索层")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    ap.add_argument("--corpus", type=Path, default=CORPUS, help="资料库 md（默认=corpus/sample.md）")
    ap.add_argument("--eval-set", type=Path, default=EVAL_SET, help="题目集 json（默认=8 题公开样例版）")
    args = ap.parse_args()
    corpus, eval_set = args.corpus, args.eval_set

    cases = json.loads(eval_set.read_text(encoding="utf-8"))["questions"]
    if args.limit:
        cases = cases[: args.limit]

    t0 = time.time()
    chunks = split_chunks(corpus.read_text(encoding="utf-8"))
    vecs = [embed(t + "\n" + b) for t, b in chunks]         # ⑬ 每块一个向量
    print(f"[库] {corpus.name} -> {len(chunks)} 块 | 向量模型={EMBED_MODEL} MAX_CHARS={MAX_CHARS} "
          f"| 建索引 {time.time() - t0:.1f}s")
    print(f"[题] {len(cases)} 题（有答案 {sum(c['answerable'] for c in cases)} / 库里没有 "
          f"{sum(not c['answerable'] for c in cases)}）\n")

    rows = []
    print("[检索层] 正确块排名 / recall@k")
    for c in cases:
        gold = gold_hits(chunks, c["gold_phrases"]) if c["answerable"] else set()
        qv = embed(c["question"])
        ranked = sorted(((cosine(qv, v), i) for i, v in enumerate(vecs)), reverse=True)
        order = [i for _, i in ranked]
        rank = next((r for r, i in enumerate(order, 1) if i in gold), None) if gold else None
        rec = {k: (rank is not None and rank <= k) for k in RECALL_KS}
        top1 = ranked[0]
        tag = "NOGOLD" if c["answerable"] and not gold else ("OK" if (not gold or rec[1]) else "MISS")
        gold_title = chunks[order[rank - 1]][0] if rank else "-"   # ⑯ 排名第 rank 的块是哪个，别用 rank-1 当块号
        print(f"  {c['id']}  [{tag:6}] top1={top1[0]:.4f} 排名={rank if rank else '-'} "
              f"命中@1={rec[1]}  gold块={gold_title}")
        rows.append({"case": c, "rank": rank, "rec": rec, "top1": top1[0],
                     "top1_title": chunks[top1[1]][0], "order": order, "gold": gold})

    no_gold = [r["case"]["id"] for r in rows if r["case"]["answerable"] and not r["gold"]]
    if no_gold:
        print(f"[WARN] 有答案题没有 gold 块：{no_gold}")
    ann = [r for r in rows if r["case"]["answerable"]]
    unann = [r for r in rows if not r["case"]["answerable"]]
    print("\n[检索层小结]")
    for k in RECALL_KS:
        print(f"  recall@{k}（有答案 {len(ann)} 题）= {pct(sum(r['rec'][k] for r in ann), len(ann))}")
    m_ann = statistics.mean(r["top1"] for r in ann) if ann else 0.0
    m_un = statistics.mean(r["top1"] for r in unann) if unann else 0.0
    print(f"  判别力余量：有答案题 top1 均值 {m_ann:.4f} - 库外题 top1 均值 {m_un:.4f} = {m_ann - m_un:+.4f}")

    cite_ok = cite_any = refuse = gen_ans = un_ans = n_error = 0
    error_ids = []
    mismatch = []
    if not args.no_gen:
        print(f"\n[生成层] 模型={GEN_MODEL} temperature=0，喂 top-{GEN_TOP_K}")
        for r in rows:
            ctx = [(n, chunks[i][0], chunks[i][1]) for n, i in enumerate(r["order"][:GEN_TOP_K], 1)]
            try:
                g = ask_model(r["case"]["question"], ctx)
            except Exception as e:                          # ⑭ 单题失败不拖垮整轮
                n_error += 1
                error_ids.append(r["case"]["id"])
                print(f"  {r['case']['id']}  [ERROR] {type(e).__name__}: {e}")
                continue
            cited = {ctx[n - 1][1]: n for n in g["marked"] if 1 <= n <= len(ctx)}
            cited_idx = {r["order"][n - 1] for n in g["marked"] if 1 <= n <= len(ctx)}
            showed = cited_idx & r["gold"]                          # ⑮ 引用的块里有没有 gold
            if r["case"]["answerable"]:
                gen_ans += 1
                cite_any += bool(g["marked"])
                cite_ok += bool(showed)
            else:
                un_ans += 1
                refuse += g["answer"] == NO_ANS
            if sorted(set(g["marked"])) != sorted(set(g["field"])):
                mismatch.append(r["case"]["id"])
            r["gen"] = g
            short = g["answer"].replace("\n", " ")[:52]
            print(f"  {r['case']['id']}  标记={g['marked']} 引用命中gold={bool(showed)} "
                  f"| {short}")
        print("\n[生成层小结]")
        print(f"  成功生成 {gen_ans + un_ans}/{len(rows)} 题；失败 {n_error} 题 {error_ids}；以下比例只以成功生成的题为分母")
        print(f"  引用准确率（有答案题成功生成 {gen_ans}/{len(ann)} 题）= {pct(cite_ok, gen_ans)}；"
              f"其中至少引了一处 = {pct(cite_any, gen_ans)}")
        print(f"  库里没有的题 {un_ans} 题，说出『{NO_ANS}』= {pct(refuse, un_ans)}")
        print(f"  标记与 sources 字段不一致 = {len(mismatch)} 题 {mismatch}")

    summary = {
        "corpus": corpus.name, "chunks": len(chunks), "embed_model": EMBED_MODEL,
        "gen_model": None if args.no_gen else GEN_MODEL, "max_chars": MAX_CHARS,
        "n_answerable": len(ann), "n_unanswerable": len(unann),
        "recall@1": sum(r["rec"][1] for r in ann), "recall@3": sum(r["rec"][3] for r in ann),
        "recall@5": sum(r["rec"][5] for r in ann),
        "top1_mean_answerable": round(m_ann, 4), "top1_mean_unanswerable": round(m_un, 4),
        "citation_accuracy": cite_ok, "refusal": refuse,
        "n_generated_answerable": gen_ans, "n_generated_unanswerable": un_ans,
        "n_error": n_error, "error_ids": error_ids, "n_no_gold": len(no_gold), "no_gold": no_gold,
        "marker_field_mismatch": mismatch, "seconds": round(time.time() - t0, 1),
    }
    print("\n[JSON] " + json.dumps(summary, ensure_ascii=False))
    if n_error or no_gold:
        print(f"[WARN] 评测未完整通过：生成失败 {n_error} 题，缺 gold 块 {len(no_gold)} 题")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
