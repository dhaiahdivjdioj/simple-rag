"""
simple_rag — 基于 pgvector + Ollama 的轻量 RAG 系统

    python simple_rag/simple.py                       # 一键初始化 + 扫描文件 + 向量化入库
    python simple_rag/simple.py query "问题"           # 单次 RAG 问答
    python simple_rag/simple.py chat                   # 交互式对话
    python simple_rag/simple.py init                   # 仅初始化数据库表
    python simple_rag/simple.py vectorize <文件路径>    # 单文件向量化入库
"""

import os
import sys
import re
import json
import argparse
import textwrap
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# 配置 — 可按需修改
# ---------------------------------------------------------------------------

# pgvector 数据库连接
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", 5432)),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", "example"),
}

# Ollama 服务地址
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 模型配置
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "bge-m3")   # 1024 维, BAAI 多语言
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen3-vl:8b")

# 分块参数 (中文按字符数)
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", 500))        # 每块中文字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))    # 块间重叠字符

# 批量 embedding 参数
EMBED_BATCH = int(os.getenv("EMBED_BATCH", 10))         # 每批发送条数

# 检索参数
TOP_K = int(os.getenv("TOP_K", 3))

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
ETL_DIR = ROOT / "etl_file"

# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

DDL_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

DDL_TABLE = """
CREATE TABLE IF NOT EXISTS simple_rag (
    id          SERIAL PRIMARY KEY,
    source      TEXT        NOT NULL,
    chunk_id    TEXT        NOT NULL,
    start_index INT         NOT NULL,
    end_index   INT         NOT NULL,
    created_at  TIMESTAMP   DEFAULT NOW(),
    updated_at  TIMESTAMP   DEFAULT NOW(),
    content     TEXT        NOT NULL,
    embedding   VECTOR(1024)
);
"""

# ---------------------------------------------------------------------------
# 数据库连接
# ---------------------------------------------------------------------------

_db_conn = None


def _check_imports():
    """诊断：检查依赖是否安装，返回错误信息或 None。"""
    missing = []
    for mod in ["psycopg", "pgvector", "requests"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if "psycopg" in missing:
        # psycopg2 作为备选
        try:
            __import__("psycopg2")
            missing.remove("psycopg")
        except ImportError:
            pass
    return missing


def get_db():
    """返回 pgvector 数据库连接（懒加载单例）。
    优先 psycopg (v3)，降级 psycopg2。
    """
    global _db_conn
    if _db_conn is not None:
        return _db_conn

    # 诊断
    missing = _check_imports()
    if missing:
        sys.exit(
            f"缺少依赖: {', '.join(missing)}\n"
            "请运行: pip install psycopg pgvector requests ollama"
        )

    # 尝试 psycopg (v3)
    try:
        import psycopg
        from pgvector.psycopg import register_vector
        _db_conn = psycopg.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
        register_vector(_db_conn)
        return _db_conn
    except ImportError:
        pass

    # 降级：psycopg2
    import psycopg2
    from pgvector.psycopg2 import register_vector
    _db_conn = psycopg2.connect(**DB_CONFIG)
    register_vector(_db_conn)
    return _db_conn


# ---------------------------------------------------------------------------
# HTML / 文本清洗
# ---------------------------------------------------------------------------

_HTML_RE = re.compile(r"<[^>]*>")


def strip_html(text: str) -> str:
    """去除 HTML 标签，合并多余空白。"""
    text = _HTML_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Ollama 调用
# ---------------------------------------------------------------------------

def ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    """Ollama 批量 embedding（bge-m3 支持 batch）。"""
    if not texts:
        return []
    # Ollama /api/embed 接收 "input": [str, ...] 作为批量输入
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    # 返回 {"embeddings": [[...], [...]]}
    return resp.json()["embeddings"]


def ollama_embed_single(text: str) -> list[float]:
    """单条 embedding（降级兜底）。"""
    resp = requests.post(
        f"{OLLAMA_BASE}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def ollama_embed(texts) -> list[list[float]]:
    """智能 embedding：单条或批量。"""
    if isinstance(texts, list):
        if len(texts) == 1:
            return [ollama_embed_single(texts[0])]
        # 分批：避免单次请求过大
        all_vecs = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            try:
                all_vecs.extend(ollama_embed_batch(batch))
            except Exception:
                # 降级为逐条
                for t in batch:
                    all_vecs.append(ollama_embed_single(t))
        return all_vecs
    else:
        return [ollama_embed_single(texts)]


def ollama_chat(prompt: str, system: str = "你是一个有帮助的AI助手。") -> str:
    """调用 Ollama chat 接口。"""
    # 限制 prompt 总长度，防止 8B 模型推理超时
    max_chars = 3000
    if len(prompt) > max_chars:
        prompt = prompt[:max_chars] + "\n\n...[内容已截断]"

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# 数据库初始化
# ---------------------------------------------------------------------------

def init_db():
    """创建 pgvector 扩展和数据表。"""
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute(DDL_EXTENSION)
        cur.execute(DDL_TABLE)
        db.commit()
        print("[OK] pgvector 扩展已启用，simple_rag 表已就绪。")
    except Exception as e:
        db.rollback()
        sys.exit(f"[FAIL] 初始化失败: {e}")
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# 中文文本分块 — 按句边界 + 固定字符数
# ---------------------------------------------------------------------------

# 中文断句：句号、问号、感叹号、分号、换行等
_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？；\n])\s*"
)


def chunk_text(text: str, source: str,
               chunk_size: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    将文本切分为带元数据的块。
    - 先按句边界（。！？；\\n）粗切为句子
    - 再将句子拼接至接近 chunk_size 字符
    - 块间保留 overlap 字符的重叠
    """
    # 按句边界切分
    raw_sentences = _SENT_SPLIT_RE.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    idx = 0
    i = 0
    char_offset = 0  # 在原文本中的累计偏移

    while i < len(sentences):
        # 拼接句子直到接近 chunk_size
        buf = sentences[i]
        j = i + 1
        while j < len(sentences) and len(buf) + len(sentences[j]) < chunk_size:
            buf += sentences[j]
            j += 1

        chunk_content = buf

        # 在原文本中定位 start/end
        pos = text.find(buf, char_offset)
        if pos == -1:
            pos = char_offset
        start_index = pos
        end_index = pos + len(buf)

        chunks.append({
            "source": source,
            "chunk_id": f"{source}::{idx:04d}",
            "start_index": start_index,
            "end_index": end_index,
            "content": chunk_content,
        })

        idx += 1

        # 下一个块：从当前块的倒数 overlap 字符处开始
        if j > i + 1:
            # 多句拼接的块：回退 overlap，从上一块的尾部开始
            overlap_text = chunk_content[-overlap:] if len(chunk_content) > overlap else chunk_content
            # 找到 overlap_text 对应的句子位置
            i = j - 1  # 先回退一句
            char_offset = end_index - len(overlap_text)
            i = max(i, i)  # 保持至少前进
        else:
            i = j
            char_offset = end_index - overlap

        if i >= len(sentences):
            break

    return chunks


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------

def load_file(filepath: Path) -> tuple[str, str]:
    """加载单个文件，返回 (relative_path, clean_text)。"""
    content = filepath.read_text(encoding="utf-8")
    cleaned = strip_html(content)
    rel = str(filepath.relative_to(ROOT))
    return rel, cleaned


def load_directory(directory: Path) -> list[tuple[str, str]]:
    """递归收集目录下所有文本文件，返回 [(path, content), ...]"""
    results = []
    if not directory.exists():
        return results
    for f in sorted(directory.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".txt", ".md", ".rst"}:
            try:
                results.append(load_file(f))
            except Exception:
                print(f"[WARN] 跳过不可读文件: {f}")
    return results


# ---------------------------------------------------------------------------
# 向量化入库
# ---------------------------------------------------------------------------

def vectorize_file(filepath: str):
    """
    向量化单个文件并存入 pgvector。
    用法: python simple_rag/simple.py vectorize etl_file/三国.md
    """
    fp = ROOT / filepath
    if not fp.exists():
        sys.exit(f"[FAIL] 文件不存在: {fp}")

    print(f"[LOAD] {filepath}")

    try:
        rel_path, text = load_file(fp)
    except Exception as e:
        sys.exit(f"[FAIL] 读取失败: {e}")

    print(f"[TEXT] 总字符: {len(text):,}")

    # 分块
    chunks = chunk_text(text, source=rel_path)
    print(f"[CHUNK] 共 {len(chunks)} 块 (每块≤{CHUNK_CHARS}字, 重叠{CHUNK_OVERLAP}字)")

    if not chunks:
        print("[INFO] 无可分块内容。")
        return

    # 检查 Ollama 模型是否就绪
    print(f"[MODEL] embedding: {EMBEDDING_MODEL}")
    try:
        tags_resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        models = {m["name"] for m in tags_resp.json().get("models", [])}
        if EMBEDDING_MODEL not in models and f"{EMBEDDING_MODEL}:latest" not in models:
            print(f"[WARN] 未找到模型 '{EMBEDDING_MODEL}'，请先执行: ollama pull {EMBEDDING_MODEL}")
    except Exception:
        print(f"[WARN] 无法连接 Ollama ({OLLAMA_BASE})，请确认服务已启动")

    # 批量 embedding
    print(f"[EMBED] 批量向量化中 (batch={EMBED_BATCH})...")
    contents = [ch["content"] for ch in chunks]
    all_vecs = ollama_embed(contents)

    # 写入数据库
    print(f"[DB] 写入 pgvector...")
    db = get_db()
    cur = db.cursor()

    inserted = 0
    for ch, vec in zip(chunks, all_vecs):
        cur.execute(
            """
            INSERT INTO simple_rag
                (source, chunk_id, start_index, end_index, content, embedding)
            VALUES
                (%s, %s, %s, %s, %s, %s)
            """,
            (ch["source"], ch["chunk_id"], ch["start_index"],
             ch["end_index"], ch["content"], vec),
        )
        inserted += 1

    db.commit()
    cur.close()
    print(f"[DONE] 向量化完成 → {inserted} 条记录已入库")


def ingest():
    """从 docs/ 和 etl_file/ 读取所有文件，分块、嵌入、入库。"""
    all_files = []
    for folder in [DOCS_DIR, ETL_DIR]:
        found = load_directory(folder)
        all_files.extend(found)
        print(f"[INFO] {folder.name}/ 下找到 {len(found)} 个文件")

    if not all_files:
        print("[INFO] 没有可导入的文档，请在 docs/ 或 etl_file/ 放入 .txt/.md 文件。")
        return

    db = get_db()
    total = 0

    for rel_path, text in all_files:
        chunks = chunk_text(text, source=rel_path)
        print(f"[CHUNK] {rel_path} → {len(chunks)} 块")

        contents = [ch["content"] for ch in chunks]
        all_vecs = ollama_embed(contents)

        cur = db.cursor()
        for ch, vec in zip(chunks, all_vecs):
            cur.execute(
                """
                INSERT INTO simple_rag
                    (source, chunk_id, start_index, end_index, content, embedding)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                """,
                (ch["source"], ch["chunk_id"], ch["start_index"],
                 ch["end_index"], ch["content"], vec),
            )
            total += 1
        cur.close()

    db.commit()
    print(f"\n[DONE] 共导入 {total} 个 chunk 到数据库。")


# ---------------------------------------------------------------------------
# 检索 + 生成
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """向量检索，返回 top_k 个最相似 chunk。"""
    query_vec = ollama_embed(query)[0]
    db = get_db()
    cur = db.cursor()
    cur.execute(
        """
        SELECT source, chunk_id, content,
               embedding <=> %s::vector AS distance
        FROM simple_rag
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (query_vec, query_vec, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    return [
        {"source": r[0], "chunk_id": r[1], "content": r[2], "distance": r[3]}
        for r in rows
    ]


def build_prompt(query: str, contexts: list[dict]) -> str:
    """用检索到的上下文构建 RAG prompt。"""
    ctx_blocks = []
    for i, c in enumerate(contexts, 1):
        ctx_blocks.append(
            f"[上下文 {i}] 来源: {c['source']} | 相似度: {1 - c['distance']:.4f}\n"
            f"{c['content']}"
        )
    context_str = "\n\n---\n\n".join(ctx_blocks)

    return f"""请基于以下上下文回答问题。如果上下文不足以回答问题，请如实说明。

{context_str}

---
问题: {query}

回答:"""


def query_rag(query: str, top_k: int = TOP_K) -> str:
    """完整 RAG 流程：检索 → 构建 prompt → 生成。"""
    contexts = retrieve(query, top_k=top_k)
    if not contexts:
        return "数据库中没有找到相关内容，请先向量化文档。"
    prompt = build_prompt(query, contexts)
    return ollama_chat(prompt)


# ---------------------------------------------------------------------------
# 交互式对话
# ---------------------------------------------------------------------------

def chat_loop():
    """交互式 RAG 对话。"""
    print("=" * 60)
    print("  simple_rag 交互对话 (pgvector + Ollama)")
    print("  输入 /quit 退出, /sources 查看来源")
    print("=" * 60)

    last_sources = []

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/quit", "/exit", "/q"}:
            print("再见！")
            break
        if user_input.lower() == "/sources":
            if last_sources:
                for s in last_sources:
                    print(
                        f"  [{s['chunk_id']}] {s['source']} "
                        f"(相似度: {1 - s['distance']:.4f})"
                    )
            else:
                print("  暂无来源信息。")
            continue

        contexts = retrieve(user_input)
        last_sources = contexts

        if not contexts:
            print("[!] 未找到相关内容。")
            continue

        prompt = build_prompt(user_input, contexts)
        answer = ollama_chat(prompt)
        print(f"\n{answer}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="simple_rag — pgvector + Ollama 轻量 RAG (bge-m3)"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="仅初始化 pgvector 扩展与数据表")

    vec_p = sub.add_parser("vectorize", help="单文件向量化入库")
    vec_p.add_argument("filepath", help="文件路径（相对于项目根目录）")

    sub.add_parser("query", help="单次 RAG 查询")
    sub.add_parser("chat", help="交互式 RAG 对话")

    args, unknown = parser.parse_known_args()

    if args.command == "init":
        init_db()
        return

    if args.command == "vectorize":
        vectorize_file(args.filepath)
        return

    if args.command == "query":
        question = " ".join(unknown) if unknown else input("问题: ").strip()
        if not question:
            sys.exit("请提供问题。")
        answer = query_rag(question)
        print(f"\n{answer}\n")
        return

    if args.command == "chat":
        chat_loop()
        return

    # ── 默认模式：无参数运行 → 全自动向量化 ──
    print("=" * 60)
    print("  simple_rag — 自动向量化模式")
    print(f"  embedding: {EMBEDDING_MODEL} | chat: {CHAT_MODEL}")
    print("=" * 60)

    # 1. 初始化数据库
    init_db()

    # 2. 扫描文件
    all_files = []
    for folder in [ETL_DIR, DOCS_DIR]:
        found = load_directory(folder)
        all_files.extend(found)
        print(f"[SCAN] {folder.name}/ → {len(found)} 个文件")

    if not all_files:
        print("[INFO] 未找到 .md/.txt 文件。请把文档放入 docs/ 或 etl_file/ 后重试。")
        return

    # 3. 向量化入库
    db = get_db()
    total = 0

    for rel_path, text in all_files:
        chunks = chunk_text(text, source=rel_path)
        print(f"[CHUNK] {rel_path} ({len(text):,} 字) → {len(chunks)} 块")

        contents = [ch["content"] for ch in chunks]
        print(f"[EMBED] 正在向量化...", end="", flush=True)
        all_vecs = ollama_embed(contents)
        print(" 完成")

        cur = db.cursor()
        for ch, vec in zip(chunks, all_vecs):
            cur.execute(
                """
                INSERT INTO simple_rag
                    (source, chunk_id, start_index, end_index, content, embedding)
                VALUES
                    (%s, %s, %s, %s, %s, %s)
                """,
                (ch["source"], ch["chunk_id"], ch["start_index"],
                 ch["end_index"], ch["content"], vec),
            )
            total += 1
        cur.close()

    db.commit()
    print(f"\n[DONE] 向量化完成: {total} 条记录已入库 (bge-m3 / 1024维)")

    # 4. 提示后续操作
    print(f"\n现在可以:")
    print(f"  python simple_rag/simple.py chat")
    print(f'  python simple_rag/simple.py query "你的问题"')


if __name__ == "__main__":
    main()
