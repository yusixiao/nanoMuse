# nanoMuse gateway

自托管网关：手机上的 nanoMuse 以 OpenAI 兼容协议连过来，这里翻译成 Vertex
Gemini 调用（凭据只存在服务器，不下发手机），并可选挂一个 telegram-mcp
sidecar 让 agent 读写 Telegram。

```
phone ──https──▶ Caddy ──▶ gateway:8000   (/v1/chat/completions …)
                      └──▶ telegram-mcp:8080 (/mcp, Bearer 门禁)
```

## 部署

```bash
# 1. 一台能拿 Let's Encrypt 证书的机器（真域名或 sslip.io 都行；裸 IP 的
#    自签证书会被 LibreSSL/BoringSSL 拒绝）
# 2. 仓库 clone 到服务器，进入 gateway/
cp env.example .env && $EDITOR .env     # GATEWAY_TOKEN 等
mkdir sa && cp <your-sa>.json sa/credentials.json
# 3. Caddyfile 里的站点地址换成你的域名
docker compose up -d --build
# 4. 手机：Settings → Cloud/relay → 填 https://<域名>，API key 填 GATEWAY_TOKEN
```

## 本地开发

```bash
python -m pytest gateway/nanomuse_gateway/tests -q
GATEWAY_SA_FILE=<sa.json> uvicorn nanomuse_gateway.server:build_app \
    --factory --port 8000
```

## 端点

| 路径 | 说明 |
|---|---|
| `POST /v1/chat/completions` | OpenAI 协议入站 → Vertex Gemini 出站（流式/工具调用）|
| `GET /healthz` | relay 探测形状（`ok`/`version`/`models`）|
| `GET /v1/hub` | 手机 hub：GET stub + websocket accept-hold |
| `POST /mcp` | telegram-mcp sidecar（Caddy 层 Bearer 鉴权）|

环境变量见 `env.example`。`GATEWAY_SA_FILE` 指向服务账号 JSON；
`GATEWAY_TOKEN` 是手机侧唯一凭据。
