"""
userui — simple_rag 的 Gradio 聊天界面

    python simple_rag/userui.py          # 启动 Web UI (默认 http://localhost:7860)
    python simple_rag/userui.py --port 8080
    python simple_rag/userui.py --share  # 生成公网链接
"""

import sys
from pathlib import Path

# 确保同目录下的 simple.py 可被 import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simple  # noqa: E402

# ---------------------------------------------------------------------------
# RAG 核心 — 直接从 simple 模块复用
# ---------------------------------------------------------------------------


def do_rag(query: str, top_k: int = 5, chat_model: str | None = None):
    """
    执行一次 RAG：检索 + 生成。
    返回 (answer, sources)，sources 是检索到的 chunks 列表。
    """
    contexts = simple.retrieve(query, top_k=top_k)
    if not contexts:
        return "数据库中没有找到相关内容，请先向量化文档。", []

    prompt = simple.build_prompt(query, contexts)

    if chat_model:
        original = simple.CHAT_MODEL
        simple.CHAT_MODEL = chat_model
        try:
            answer = simple.ollama_chat(prompt)
        finally:
            simple.CHAT_MODEL = original
    else:
        answer = simple.ollama_chat(prompt)

    sources = [
        {
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "similarity": round(1 - c["distance"], 4),
            "content": c["content"][:200] + ("..." if len(c["content"]) > 200 else ""),
        }
        for c in contexts
    ]
    return answer, sources


# ---------------------------------------------------------------------------
# Gradio UI (兼容 Gradio 5/6)
# ---------------------------------------------------------------------------

def build_ui():
    try:
        import gradio as gr
    except ImportError:
        sys.exit("请安装 gradio: pip install gradio")

    with gr.Blocks(title="simple_rag · 三国演义问答") as demo:
        gr.Markdown(
            "# 📖 simple_rag · 三国演义问答\n"
            "基于 **pgvector + Ollama** (qwen3:4b + bge-m3) 的本地 RAG 问答。"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话")

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="输入问题，例如：关羽温酒斩华雄是哪一回？",
                    )
                    send_btn = gr.Button("发送")

                with gr.Row():
                    clear_btn = gr.Button("清空对话")
                    top_k_slider = gr.Slider(
                        1, 20, value=5, step=1,
                        label="检索条数",
                    )

            with gr.Column(scale=2):
                sources_md = gr.Markdown("### 检索来源\n\n*等待提问...*")

        # ── 回调 ──
        def respond(message, history, top_k):
            if not message.strip():
                return history, "### 检索来源\n\n*请输入问题*"

            try:
                answer, sources = do_rag(message, top_k=top_k)
            except Exception as e:
                error_msg = f"**出错:** {e}"
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": error_msg})
                return history, "### 检索来源\n\n*查询失败*"

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})

            # 构建来源面板
            if sources:
                lines = ["### 📎 检索来源\n"]
                for i, s in enumerate(sources, 1):
                    lines.append(
                        f"**{i}.** `{s['chunk_id']}`  "
                        f"*相似度: {s['similarity']}*\n\n"
                        f"> {s['content']}\n"
                    )
                sources_text = "\n".join(lines)
            else:
                sources_text = "### 检索来源\n\n*未找到相关内容*"

            return history, sources_text

        def clear_chat():
            return [], "### 检索来源\n\n*等待提问...*"

        send_btn.click(
            respond,
            [msg, chatbot, top_k_slider],
            [chatbot, sources_md],
        ).then(lambda: "", None, msg)
        msg.submit(
            respond,
            [msg, chatbot, top_k_slider],
            [chatbot, sources_md],
        ).then(lambda: "", None, msg)
        clear_btn.click(clear_chat, None, [chatbot, sources_md])

    return demo


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="simple_rag Web UI (Gradio)")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--share", action="store_true", help="生成 Gradio 公网链接")
    args = parser.parse_args()

    print(f"模型: {simple.CHAT_MODEL}  |  embedding: {simple.EMBEDDING_MODEL}")
    print(f"启动地址: http://localhost:{args.port}")

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
