# 贡献指南

感谢你帮助改进织语 WeaveTalk。

## 开始之前

1. 对较大的功能或行为变化，请先创建 Issue 说明场景和方案。
2. 安全漏洞请按 [SECURITY.md](SECURITY.md) 私密报告。
3. 不要提交真实客户数据、订单、聊天记录或 API 密钥。

## 本地开发

```bash
git clone https://github.com/Aleax9291/weavetalk.git
cd weavetalk
uv sync --locked
cp .env.example .env
```

创建功能分支后进行修改：

```bash
git switch -c feature/short-description
uv run ruff check .
uv run pytest
```

涉及 Agent 行为的改动还应运行相关评测集，并在 Pull Request 中注明模型、配置、采样轮数和结果变化。

## Pull Request 要求

- 一次 PR 聚焦一个目标，说明“改了什么、为什么、如何验证”。
- 新功能和缺陷修复应附带自动化测试。
- 数据集必须是合成数据，或附带清晰、兼容的授权与来源说明。
- 不要为了提高固定评测分数而加入只匹配单一问法的规则。
- 保持向后兼容；若必须破坏兼容性，请在 PR 中明确迁移方式。

提交 PR 即表示你同意按本仓库的 MIT License 提供贡献。
