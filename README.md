# simple_rag

基于 **pgvector + Ollama** 的轻量 RAG（检索增强生成）系统。

## 技术栈

| 组件 | 方案 |
|---|---|
| 向量数据库 | PostgreSQL + pgvector (1024维) |
| Embedding | Ollama + bge-m3 |
| LLM | Ollama + qwen3-vl:8b |
| UI | Gradio |

## 快速开始

```bash
# 1. 虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. 依赖
pip install psycopg pgvector requests ollama gradio

# 3. 前置服务
#    - PostgreSQL 运行中 + pgvector 扩展
#    - Ollama 运行中，模型已 pull:
#      ollama pull bge-m3
#      ollama pull qwen3-vl:8b

# 4. 一键向量化
python simple_rag/simple.py

# 5. 启动 Web UI
python simple_rag/userui.py        # → http://localhost:7860
```

## 使用

```bash
python simple_rag/simple.py                        # 全自动：init → 扫描 → 向量化
python simple_rag/simple.py query "你的问题"        # 单次问答
python simple_rag/simple.py chat                    # 终端对话
python simple_rag/simple.py vectorize <file>        # 单文件向量化

python simple_rag/userui.py                         # 启动 Web UI
python simple_rag/userui.py --port 8080 --share     # 自定义端口 + 公网链接
```

## 项目结构

```
simple_rag/
├── simple.py          # 核心：建表、分块、向量化、检索、生成
├── userui.py          # Gradio Web 聊天界面
docs/                  # 待导入文档
etl_file/              # ETL 输出或结构化数据
pyproject.toml         # 依赖配置
```
