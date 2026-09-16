# Spike 实验离线核验与绘图

报告入口：[pc7 / v2 → pc8 / v2.1](../../docs/pro/spike-study/README.md)。这些是事后审计脚本，不是历史过滤脚本，也不会启动训练、联网或提交评测。

## 直接重绘公开图表

在仓库根目录运行，Python 3.10+：

```bash
python3 -m pip install -r scripts/spike/requirements.txt
python3 scripts/spike/plot_report.py \
  --artifacts docs/pro/spike-study/artifacts \
  --figures docs/pro/spike-study/figures
python3 -m unittest discover -s scripts/spike -p 'test_*.py' -v
```

只有绘图需要 matplotlib、numpy。核验脚本只依赖 Python 标准库。

## 重新核验本地私有证据

下面的 `/path/to/...` 是占位路径，需替换为本人有权限访问的真实文件。原始证据不随仓库分发，请将新输出放到仓库外临时目录，审查后再决定是否公开。

```bash
python3 scripts/spike/audit_data.py \
  --before /path/to/v2.jsonl.gz \
  --after /path/to/v2.1.jsonl.gz \
  --before-meta /path/to/data_prochain_v7/meta.json \
  --after-meta /path/to/data_prochain_v8/meta.json \
  --output /path/to/audit-output

python3 scripts/spike/audit_training.py \
  --before-log /path/to/out_prochain_v7_r16_98k_0826/train.log \
  --after-log /path/to/out_prochain_v8_r16_98k_0827/train.log \
  --before-samples 1102 --after-samples 1010 \
  --output /path/to/audit-output

python3 scripts/spike/audit_eval.py \
  --runs /path/to/eval/runs \
  --cache-root /path/to/dsv4_run \
  --output /path/to/audit-output
```

`--cache-root` 下应有历史 `harvest_cache` / `harvest_out`；脚本只读取既有判决，不触发 harvest。前期 checkpoint 和轮次在 `audit_eval.py` 中显式固定；同名多轮或判决冲突会报错，不自动挑最新/最高的一轮。

## 输出与口径

- `data_summary.json`：整条记录 canonical SHA-256 多重集合差分，真实工具调用覆盖、长度/调用统计及 meta 的 split 聚合值。
- `removed_fingerprints.csv`：被删记录在源文件中的 1-based 行号、整条 SHA-256、事后复建的 assistant MD5、调用数。没有原文或用户 ID；MD5 仅作定位，非安全校验。
- `before_training_metrics.csv` / `after_training_metrics.csv`：每步 loss、裁剪前梯度范数、学习率、epoch 和 NaN/跳步计数。
- `training_summary.json`：两次完整训练的数值摘要、日志文件指纹和目录名；它不是完整启动配置备份。
- `binding_summary.json`：人工交叉核对后的 meta / 启动脚本指纹、源数据指纹、日志中实际数据目录及数量；保留文件名和证据行号，不公开内部完整路径。此文件不是上述脚本的自动输出。
- `early_evaluation.json`：约 0.2–0.8 epoch 的同题配对结果、各域与综合分母、gains/losses、McNemar 精确 p 值、四组综合结果的 Holm 校正及 run 指纹。

源文件分布与训练 split 分布不是同一个分母。真实/合成来源比例未推断。评测缺失题不在共同已判分母中，四档均值不是独立重复实验结果。历史候选选择规则和 spike 到样本的完整映射仍未恢复。
