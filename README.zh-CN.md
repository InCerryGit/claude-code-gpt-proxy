# Anthropic API 代理（OpenAI / Gemini / Azure）

这是一个面向 Azure 兼容性的改版/分支，主要用于提升对 Azure OpenAI 的兼容与可用性。

通过一个 FastAPI 代理（基于 LiteLLM），让 Anthropic 兼容客户端（含 Claude Code）可使用 OpenAI、Gemini、Azure OpenAI 或 Anthropic 后端。

## 特性

- Anthropic 兼容的 `/v1/messages` 与 `/v1/messages/count_tokens` 接口
- `haiku` / `sonnet` 自动映射到各家模型
- 支持 OpenAI、Gemini（AI Studio 或 Vertex ADC）、Azure OpenAI 及直连 Anthropic
- 支持流式与非流式
- 可选 OpenAI Base URL 覆盖

## 快速开始

### 前置条件

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- 你要使用的云厂商 API Key

### 源码运行

```bash
git clone https://github.com/InCerryGit/claude-code-gpt-proxy.git
cd claude-code-gpt-proxy
cp .env.example .env
```

编辑 `.env` 后启动服务：

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
```

### Docker 运行

先本地构建镜像（上游镜像已不可用）：

```bash
docker build -t claude-code-gpt-proxy:latest .
```

带端口映射运行：

```bash
docker run -d --name claude-code-gpt-proxy --env-file .env -p 8082:8082 claude-code-gpt-proxy:latest
```

## 与 Claude Code 一起使用

```bash
npm install -g @anthropic-ai/claude-code
```

编辑配置文件以使用代理：
```json
# 编辑或新增 `settings.json` 文件
# MacOS & Linux 为 `~/.claude/settings.json`
# Windows 为`用户目录/.claude/settings.json`
# 新增或修改里面的 env 字段
# 注意替换里面的 `your_anthropic_auth_token` 为您配置的 `ANTHROPIC_AUTH_TOKEN`
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "your_anthropic_auth_token",
    "ANTHROPIC_BASE_URL": "http://localhost:8082",
    "API_TIMEOUT_MS": "3000000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": 1
  }
}
# 再编辑或新增 `.claude.json` 文件
# MacOS & Linux 为 `~/.claude.json`
# Windows 为`用户目录/.claude.json`
# 新增 `hasCompletedOnboarding` 参数
{
  "hasCompletedOnboarding": true
}
```

然后运行 Claude Code：

```bash
claude
```

## 配置说明

可通过 `.env` 或环境变量配置。

### 核心 Key

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`（Google AI Studio）
- `ANTHROPIC_API_KEY`（仅直连 Anthropic 时需要）

### 入站鉴权（可选）

设置 `ANTHROPIC_AUTH_TOKEN` 后，所有 API 请求都必须携带 `Authorization: Bearer <token>`。如果未设置，则允许匿名访问。

示例请求：

```bash
curl http://localhost:8082/v1/messages \
  -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}'
```

### 供应商选择

- `PREFERRED_PROVIDER`：`openai`（默认）、`google`、`azure`、`anthropic`
- `BIG_MODEL`：对应 `sonnet` / `opus`（默认 `gpt-4.1`）
- `SMALL_MODEL`：对应 `haiku`（默认 `gpt-4.1-mini`）

### Gemini Vertex AI（ADC）

- `USE_VERTEX_AUTH=true`
- `VERTEX_PROJECT`
- `VERTEX_LOCATION`

### OpenAI Base URL

- `OPENAI_BASE_URL`（可选）

### Azure OpenAI

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_API_VERSION`

### 日志（可选）

启用 `LOG_FILE` 后会写入日志文件，并在每日午夜进行轮转。

- `LOG_LEVEL`（默认：`INFO`）
- `LOG_FILE`（设置后开启文件日志；支持 `strftime` 占位符，例如 `./logs/proxy-%Y%m%d.log`）
- `LOG_FILE_BACKUP_COUNT`（默认：`7`）
- `LOG_FILE_PATTERN`（轮转文件名的 strftime 模式，例如 `proxy-%Y%m%d.log`）

**说明**
- `LOG_FILE` 会在进程启动时按 `strftime` 渲染，因此 `%Y%m%d` 会展开为当天日期。
- 如需稳定的基础文件名并按日期轮转，可设置 `LOG_FILE=./logs/proxy.log` 与 `LOG_FILE_PATTERN=proxy-%Y%m%d.log`。

## 模型映射

- `haiku` → `SMALL_MODEL`
- `sonnet` / `opus` → `BIG_MODEL`

映射规则取决于 `PREFERRED_PROVIDER`：

- `openai`：自动加 `openai/` 前缀
- `google`：若在 Gemini 已知模型列表中则加 `gemini/`，否则回退到 OpenAI
- `azure`：加 `azure/`
- `anthropic`：加 `anthropic/`，不做重映射

### 模型前缀处理

自动补全前缀：

- OpenAI：`openai/`
- Gemini：`gemini/`
- Azure：`azure/`
- Anthropic：`anthropic/`

## 示例

### 优先 OpenAI（默认）

```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-gemini-key" # 可选回退
# PREFERRED_PROVIDER="openai"
# BIG_MODEL="gpt-4.1"
# SMALL_MODEL="gpt-4.1-mini"
```

### 优先 Gemini（AI Studio）

```dotenv
GEMINI_API_KEY="your-gemini-key"
OPENAI_API_KEY="your-openai-key" # 回退
PREFERRED_PROVIDER="google"
# BIG_MODEL="gemini-2.5-pro"
# SMALL_MODEL="gemini-2.5-flash"
```

### 优先 Gemini（Vertex ADC）

```dotenv
OPENAI_API_KEY="your-openai-key" # 回退
PREFERRED_PROVIDER="google"
USE_VERTEX_AUTH=true
VERTEX_PROJECT="your-gcp-project-id"
VERTEX_LOCATION="us-central1"
```

### 直连 Anthropic

```dotenv
ANTHROPIC_API_KEY="sk-ant-..."
PREFERRED_PROVIDER="anthropic"
```

### Azure OpenAI

```dotenv
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your-azure-key"
AZURE_API_VERSION="2024-10-01"
PREFERRED_PROVIDER="azure"
BIG_MODEL="gpt-4o"   # 部署名
SMALL_MODEL="gpt-4o-mini" # 部署名
```

## 测试

```bash
python tests.py
```

单项测试：

```bash
python tests.py --no-streaming
python tests.py --streaming-only
python tests.py --simple
python tests.py --tools
```

## 贡献

欢迎提交 PR。