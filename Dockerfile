# 企业办公自动化 Agent - Dockerfile
# Python 3.12 slim（比 3.14 更稳定的生态兼容性，chromadb/torch 均有预编译轮子）
FROM python:3.12-slim

# 设置工作目录（纯英文路径，规避 Chroma HNSW 中文路径编码问题）
WORKDIR /app

# 系统依赖：chromadb 的 hnswlib 需要 gcc+g++ 编译
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖清单（利用 Docker 层缓存）
COPY pyproject.toml requirements.txt ./

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 安装项目本身（可编辑模式，不带依赖）
COPY src/ ./src/
RUN pip install --no-deps -e .

# 复制配置与数据
COPY config/ ./config/
COPY data/raw_docs/ ./data/raw_docs/

# 模型权重目录（由 volume 挂载，不入镜像）
RUN mkdir -p models data/chroma

# 环境变量默认值（可被 docker-compose/env 覆盖）
ENV OFFICE_AGENT_LOG_LEVEL=INFO \
    OFFICE_AGENT_LLM__MODEL=deepseek-chat \
    OFFICE_AGENT_LLM__BASE_URL=https://api.deepseek.com/v1

# 默认入口：健康检查
ENTRYPOINT ["python", "-m", "office_agent.main"]
CMD ["check"]
