# 源码快照说明

这里保存项目文档直接引用的 `milly` 分支实现，不再依赖 GitLab 链接。目录结构保持原仓库相对路径，便于文档内跳转和核对。

为避免把仍可使用的凭据提交到 GitHub，快照做了最小安全处理：

- `harvest_select.py`、`render_sft.py`、`run_rounds.py` 与 `upload_all.py` 的 Langfuse Basic Auth 改由环境变量读取。
- `plan_checkpoint_eval.py` 不再内置个人邮箱，调用时必须显式传入 `--user-email`。

除这些安全修改外，文件保留上传快照内容。它是证据快照，不保证可在仓库外独立运行；运行仍依赖原项目环境、内部服务和未纳入的公共模块。
