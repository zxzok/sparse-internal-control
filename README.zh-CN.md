<a href="https://sparsecontrol.xizhe.net"><img src="https://raw.githubusercontent.com/zxzok/sparse-internal-control/gh-pages/readme/banner.png" alt="Sparse Control：冻结模型中的稀疏干预，以及评价控制的六轴协议" width="100%"></a>

# Sparse Internal Control of Language Models

[![Website](https://img.shields.io/badge/website-sparsecontrol.xizhe.net-D7EE5B?style=flat-square&labelColor=111111)](https://sparsecontrol.xizhe.net)
[![Paper](https://img.shields.io/badge/paper-PDF-D7EE5B?style=flat-square&labelColor=111111)](https://sparsecontrol.xizhe.net/paper.pdf)
[![Theory](https://img.shields.io/badge/theory-mathematical%20model-D7EE5B?style=flat-square&labelColor=111111)](https://sparsecontrol.xizhe.net/theory.html)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-F4F3EE?style=flat-square&labelColor=111111)](https://sparsecontrol.xizhe.net)
[![Record replay](https://img.shields.io/github/actions/workflow/status/zxzok/sparse-internal-control/replay.yml?branch=main&style=flat-square&label=record%20replay&labelColor=111111)](https://github.com/zxzok/sparse-internal-control/actions/workflows/replay.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-F4F3EE?style=flat-square&labelColor=111111)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-F4F3EE?style=flat-square&labelColor=111111)](pyproject.toml)

**[项目网站](https://sparsecontrol.xizhe.net)** · [理论与数学模型](https://sparsecontrol.xizhe.net/theory.html) · [论文 PDF](https://sparsecontrol.xizhe.net/paper.pdf) · [English](README.md) · [复现范围](REPRODUCIBILITY.md) · [实验说明](docs/experiments.md)

论文 **Sparse Internal Control of Language Models** 的代码与测量记录，作者为 **Xizhe Zhang**。

> **项目网站：[sparsecontrol.xizhe.net](https://sparsecontrol.xizhe.net)**。包含控制框架、带交互动画的六轴评价协议，以及背后的[数学模型](https://sparsecontrol.xizhe.net/theory.html)。

## 概览

稀疏内部控制在推理时通过少数内部位点操控一个冻结的语言模型：干预在前向传播中给少量激活加上缩放后的方向，权重保持不变，强度取零即恢复原模型。位点的选择被表述为一个控制问题（选哪些驱动点、推多大、有何保证），所得控制在六个轴上分别评价，而不是用一个分数概括。

<a href="https://sparsecontrol.xizhe.net/#protocol"><img src="https://raw.githubusercontent.com/zxzok/sparse-internal-control/gh-pages/readme/six-axes.png" alt="六轴协议：可达性、符号反转、剂量响应、反馈可控性、非目标保持、留出复用" width="100%"></a>

| 轴 | 问题 | 通过条件 |
|---|---|---|
| 可达性 Reachability | 干预能把读出推到离目标多近？ | 留出读出上 $\mathrm{Rel}_{\mathrm{tar}}(1) \le 0.4$ |
| 符号反转 Signed reversal | 反向补丁能否把目标侧行为拉回 source？ | 对目标侧输入施加 $-\beta$ 后 $\mathrm{Reversal}(\beta) \ge 2$ |
| 剂量响应 Dose response | 目标进展如何随干预强度变化？ | 十个剂量上带符号 Spearman $\rho \ge 0.95$ |
| 反馈可控性 Feedback controllability | 反馈调整的剂量能否跟踪指定目标？ | 终点误差不超过 $\tau_{\mathrm{ctrl}}$：IOI 与 MCQA 为 0.3，Refusal 为 0.1 |
| 非目标保持 Off-target preservation | 非目标输出改变了多少？ | $\mathrm{KL} \le 0.001$，种子均值与每条提示均满足 |
| 留出复用 Held-out reuse | 选出的位点或完整干预能否迁移到未见提示？ | 以连续恢复率报告，区分重拟合强度与冻结强度 |

仓库包含论文实际使用的 Driver-OMP、Coverage-Driver、比较方法、任务构造和六轴评价代码。主要任务为 IOI、Sentiment、MMLU Humanities、MMLU STEM 和 Refusal。

## 内容

- 21 组实验配置及执行入口。
- 272 组完整 source/target/patched 任务读出，对应 39 个条件组和 195 项阈值分类。
- 3,072 条逐提示、逐种子的 KL 测量。
- 重建 13 张数值表、3 张定义/配置表、阈值及 Sentiment 零位移敏感性统计的代码。
- 输入、保存的干预、模型配置、数据说明、测试与引用信息。

论文图的底层测量记录和对应关系见 [experiments/figure_records.json](experiments/figure_records.json)。仓库不含绘图代码、图片、模型权重或完整词表 logits 数组。

## 无需 GPU 的结果复算

使用 Python 3.12，在仓库之外建立环境和输出目录：

```bash
git clone https://github.com/zxzok/sparse-internal-control.git
cd sparse-internal-control

python3.12 -m venv ../sic-replay-env
source ../sic-replay-env/bin/activate
python -m pip install -r requirements/replay.lock.txt
python -m pip install --no-deps -e .

python -m sparse_control check-release --manifest
python -m sparse_control reproduce --output ../paper-replay
python -m sparse_control verify --output ../paper-replay
python -m pytest -q
```

Windows PowerShell 使用 `..\sic-replay-env\Scripts\Activate.ps1` 激活环境。

`reproduce` 从测量记录生成表格与统计；`verify` 检查完整性、论文数值、表号与静态表定义、逐种子结果、均值、标准差及阈值分类，不需要另一个独立验证脚本。部分结果复算应在两个命令中使用相同的 `--artifacts` 参数。每次运行使用新的输出目录。

未安装模型依赖或缺少显式模型 fixture 的测试会标记为跳过。GitHub Actions 仅运行 CPU 记录复算，不启动语言模型实验。

## 运行模型实验

请阅读 [实验说明](docs/experiments.md)、[资源与环境配置](docs/compute_resources.md)和[复现范围](REPRODUCIBILITY.md)。完整命令示例见 [英文 README](README.md#running-model-experiments)。

模型实验需独立环境、相应依赖和模型权重。`--plan` 只打印配置；实际执行应指定新的 `--output` 目录。固定干预评价使用保存的支持与系数；完整拟合通过 `--stage full` 执行。两者不会自动覆盖用于论文的测量记录。

## 指标与范围

- `Rel_tar` 用 source 到 target 的位移归一化；`R_clean` 用 clean 输出范数归一化，不能直接互换。
- P 为 evaluation-pair 方向，F 为 training-mean 方向。F 反馈仍可调整剂量，与完全冻结的开环复用不同。
- Sentiment 分别评价非配对剂量响应与配对固定剂量复用，未单独评价反馈跟踪。
- 保持性表格由逐提示 KL 标量重建；重新计算每个 KL 测量需要运行模型实验，定义仍是全词表 KL。
- 保存记录的复算不等于完整模型复现；原始输入和运行身份的缺失情况见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 文档与引用

[数据字典](docs/data_dictionary.md) · [方法说明](docs/methods.md) · [实验配置](docs/experiments.md) · [第三方素材](docs/third_party_assets.md)

引用信息见 [CITATION.cff](CITATION.cff) 和 [英文 README](README.md#citation)。代码采用 [Apache License 2.0](LICENSE)，版权与许可证变更历史见 [NOTICE](NOTICE)（1.1.0 及以前版本为 MIT），专利相关承诺见 [PATENTS.md](PATENTS.md)；第三方数据和模型遵循各自许可证。

问题反馈请使用 [GitHub Issues](https://github.com/zxzok/sparse-internal-control/issues)，附实验 ID、命令、环境及错误或数值差异。
