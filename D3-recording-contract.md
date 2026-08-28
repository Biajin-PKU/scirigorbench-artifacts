# D3 · 记录合同（Recording Contract）v1.0

> 交付物 D3。RH topic **91**。上位口径：RH 仓库
> `docs/innovation-kb/two-paper-execution-plan-2026-08-25.md` §4.8。
>
> **一句话**：一个 agent 至少要记下哪些「声明↔工件」对，才**可能**被赋予任何误差率。
> 记不下来的，我们不是给它打低分——**是根本无法给它打分**，而这本身就是结论。

## 0. D3 有两面，缺一面 Φ 就没有精度

| 面 | 是什么 | 谁提供 | 何时冻结 |
|---|---|---|---|
| **A · 断言类型目录** | 哪些量算「可核查的声明」 | **我们，公开、固定、对所有臂相同** | 一次，随 bench 发布 |
| **B · 工件范围声明** | 这次跑的哪些输出文件是「论文可以引用的结果」 | **被测方声明**；没有就由我们逐臂手工声明并公开 | **每臂一次，在该臂的自然趟被审计之前** |

**为什么 B 面是必需的（实测教训，2026-08-25）**：不划范围时，一次跑的工件索引有
**299,345 个键，其中只有约 874 个是论文可能引用的聚合结果**，其余是逐样本预测与原始交互。
连试三个自动判据（行数上限 / 有无描述符 / 行标识词能否被稿件命名）想把逐样本行切掉，
**每一个都切错了东西**，两次把 432 行的主结果表整个切没。

> **词法匹配对着一棵没有边界的输出树，做不出精度。**
> 而 N 一旦被噪声污染，「它记得少」与「我们分不清」就混成了同一个数字——
> 正是 §4.8.7 明令禁止的混读。**所以范围必须被声明，不能被猜。**

## 1. A 面 · 断言类型目录

**目录不是我们发明的。** 它是**已发表、社区采用的报告规范里，可被机械核对的那个子集**。
若目录从 RH 的轨迹归纳，就只量得到 RH 碰巧记录的东西（§4.8.4）。

两个独立锚点，均为逐字引用：

- **[MLRC]** *The Machine Learning Reproducibility Checklist* **v2.0, Apr. 7 2020**
  （Pineau et al., *Improving Reproducibility in Machine Learning Research
  (A Report from the NeurIPS 2019 Reproducibility Program)*, arXiv:2003.12206；
  清单原件 `www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist.pdf`）
- **[NPC]** *NeurIPS Paper Checklist*（`neurips.cc/public/guides/PaperChecklist`）

| 类型 | 逐字锚点 | 声明侧长什么样 | 工件侧必须记下什么 |
|---|---|---|---|
| **`metric`** | [MLRC] *"A clear definition of the specific measure or statistics used to report results."* | "在 X 上取得 Y" | 该指标的计算结果，带可命名的身份（数据集/模型/指标） |
| **`stat_test`** | [MLRC] *"A description of results with central tendency (e.g. mean) & variation (e.g. error bars)."*；[NPC] Q6 *"Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?"* | "p < 0.01"、"±0.003"、"95% CI [a,b]" | 离散度/检验量本身，且与其中心量同属一条记录 |
| **`seeds`** | [MLRC] *"The exact number of training and evaluation runs."* | "跑了 5 个随机种子" | 实际执行次数 / 种子清单 |
| **`split`** | [MLRC] *"The details of train / validation / test splits."*；[NPC] Q7 *"…did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?"* | "在留出集上评估"、"80/10/10" | 各划分的实际规模或索引 |
| **`hyperparam`** | [MLRC] *"The range of hyper-parameters considered, method to select the best hyper-parameter configuration, and specification of all hyper-parameters used to generate results."* | "学习率 1e-3，训 50 轮" | 实际生效的超参值 |
| **`data_source`** | [MLRC] *"The relevant statistics, such as number of examples."* | "12,483 条样本" | 实际读入的规模 |
| **`compute`** | [MLRC] *"The average runtime for each result, or estimated energy cost."* + *"A description of the computing infrastructure used."*；[NPC] Q8 *"…sufficient information on the computer resources (type of compute workers, memory, time of execution)…"* | "单卡 4 小时" | 实测墙钟 / 显存 / 设备 |

### 1.1 明写在外的东西（范围诚实）

清单里**不进本目录**的条目：数学设定与假设的描述、定理的证明、依赖说明、训练/评测代码、
预训练权重、README。理由只有一条：**它们不是数值，没有"工件里那个对应的数"可比。**

> **D3 = 一份已发表报告清单中，可被机械核对的数值子集。**
> 我们没有发明标尺，我们取了公认的标尺，并只保留能与工件做确定性比对的刻度。

⚠️ **两个锚点都是 ML 领域的。** 跨学科臂（如天文的 Denario）需要各自领域的对应规范，
**不得把 ML 清单当成普适清单套上去**；未做此项之前，跨学科结论不得写。

## 2. B 面 · 工件范围声明

### 2.1 格式

跑的根目录下一个 `record_contract.json`：

```json
{
  "record_contract_version": "1.0",
  "reportable_outputs": [
    "analysis_outputs/results.json",
    "analysis_outputs/*_metrics.csv"
  ],
  "excluded_rationale": {
    "analysis_outputs/predictions.csv": "per-instance dump; no sentence cites a single row"
  }
}
```

**`reportable_outputs` 是唯一必需字段**：论文可以引用的结果文件（glob）。
`excluded_rationale` 可选，但**被排除的大文件建议写明理由**——它是可审计性的一部分。

### 2.2 没有声明的 agent 怎么办（这是常态，不是例外）

**不猜。** 由我们**逐臂手工声明一次**，并且：

1. 该声明**随 bench 公开**，任何人可复核；
2. 声明**在该臂的自然趟被审计之前冻结**（进 §4.8.5 第 4 步的 SHA-256 清单），
   **绝不允许看过结果再调**；
3. 论文里明写它是**实验方提供的输入**，不是自动推断的；
4. 报告该声明对 N 的影响。

> **不假装自动。** 三次自动判据都失败了；把手工声明伪装成算法，只会把
> 「我们划错了边界」变成一个看不见的误差。**公开的手工输入好过隐蔽的错误启发式。**

### 2.3 声明范围之外的东西

不在 `reportable_outputs` 里的输出**不进工件索引**，因此引用它们的声明**落不进宇宙**，
记为 `unresolved`。这是**我们的覆盖损失**（§4.8.7 第一行），逐臂单独报，
**不得计入该臂的过程保真度**。

## 3. 可否被赋予误差率：判定

一个臂在一次跑上**可被赋予误差率**，当且仅当：

| # | 条件 | 不满足的后果 |
|---|---|---|
| 1 | 有 `reportable_outputs` 声明（自带或我们冻结的） | 无法划边界 ⇒ 不可判定 |
| 2 | 声明的文件里，行/记录**带可命名的身份**（哪个数据集、哪个模型、哪个指标），而非仅位置下标 | 声明无从绑定 ⇒ N→0 |
| 3 | `N_run ≥ 10`（E8 冻结件 §2.1 的可评估阈值） | 记为 non-evaluable，单列报告 |
| 4 | 目录里至少两个类型有非空槽位 | 只测到单一类型 ⇒ 不做分层结论 |

**都不满足 ⇒ 该臂进榜单第 1 列，但第 2、3 列为「无法判断」。**
这不是四象限里的任何一格。**而今天所有的榜都会给这种 agent 一个正常的分数。**

## 4. 与 Φ 的接口

`phi.py` 的 `artifact_index(run_dir, nameable, contract)`：
`contract["reportable_outputs"]` 决定遍历哪些文件。**目录（A 面）固定编译进代码并公开；
范围（B 面）逐臂输入。**

**验收判据不变**：植入前后**槽位名集合完全相同**（`phi.py --self-check`）。
D3 改变的是 N 的大小与纯度，**不改变槽位身份的定义**——身份仍是
`(断言类型, 工件键, 出现序号)`，与值无关。

## 5. 待办（未做，不得当已做引用）

- [ ] 把 `reportable_outputs` 接进 `phi.py`，在划定范围后**重测 N**（够不够支撑 E1）
- [ ] 为 5 条臂各写一份冻结的范围声明
- [ ] 跨学科臂的领域规范锚点（本文件的两个锚点都只覆盖 ML）

---

## 4. ⭐ 第三个面 · 声明标记合同（2026-08-27 新增，由 E5 的失败逼出）

**v1.0 的两个面都在工件侧**：A 说哪些量算可核查声明，B 说哪些输出是可被引用的结果。
**没有一个约束稿件如何标记它自己的声明。**

E5 给 AI Scientist-v2 补齐了 B 面所要求的一切——把它已有的曲线终点、脚本字面量、
token 账本重新序列化成 D3 形态的命名记录，每臂 10–15 个键。**N 几乎不动，0/10 仍是 0/10。**

原因：**该臂的稿件是 PDF，抽取器分不开研究断言与散文数字**（节号、引用年份、列表序号）。
LaTeX 之所以能用，是因为 `\input` 结构与表格列规格**顺带**提供了这个标记——
**而那是 LaTeX 的性质，不是我们设计的**。

> **⇒ 一份只规定「记什么」的合同是不完整的。**

### 4.1 C 面 · 声明标记

**要求**：agent 必须发出一份**机器可读的声明清单**，把稿件中哪些数字是研究断言标出来。

可接受的形态（任一）：

| 形态 | 例 |
|---|---|
| **宏表** | 稿件用 `\acc{}` 之类的宏印数，宏名即槽位名（RH 走的就是这条） |
| **claim 清单** | 与稿件并列的 `claims.json`：`{位置/锚点 → 断言类型 → 工件键}` |
| **结构化稿件** | 结果表以带行标签与列表头的表格形式发出（Markdown 管道表即可，实测有效） |

**不可接受**：纯 PDF、无结构的散文稿。**不是因为难，是因为它不携带该信息。**

### 4.1.1 ⭐ 粒度必须是**单元**，不是文件（2026-08-27，由 ARA 实测逼出）

ARA 已经实现了 C 面的一个版本：每条 claim 带 `Evidence basis: Table 2 (...)`。
**我们去消费它，N 从 1 降到 0。**

原因是粒度：

```
工件键     table2_imagenet_plain_vs_residual.md[18 layers | plain] = 27.94
声明写的   plain-18 = 27.94%
```

**链接指到了表，没指到格。** 且两侧命名法不同——声明用 `plain-18`，表用行 `18 layers` × 列 `plain`。
收窄到正确的表**并不解决"是哪一个单元"**。

> **⇒ C 面必须要求 claim → 单元 的链接，不是 claim → 证据文件 的链接。**
> **分母需要单元级身份；文件级指针给不了它。**

可接受形态因此收紧为：

| 形态 | 是否够 |
|---|---|
| 宏表（宏名即槽位名） | ✅ 天然单元级 |
| `claims.json` 里 `{锚点 → 断言类型 → **工件键**}` | ✅ 只要工件键指到格 |
| `Evidence basis: Table 2` 式的**文件级**指针 | ❌ **不够**——ARA 实测 |
| 结构化结果表（行标签+列表头） | ⚠️ 工件侧够，**仍需声明侧指明是哪一格** |

**这一条同样不是设计时想到的**：是拿一个已存在的实现去消费、失败了才反推出来的。

### 4.2 三个面的分工

| 面 | 内容 | 侧 | 谁提供 |
|---|---|---|---|
| A · 断言类型目录 | 哪些量算可核查声明 | 工件 | **我们**，公开固定 |
| B · 工件范围声明 | 哪些输出可被引用 | 工件 | 被测方 / 我们逐臂声明 |
| **C · 声明标记** | **稿件里哪些数字是断言** | **声明** | **被测方** |

> **A 与 B 决定「有没有可比的东西」，C 决定「知不知道该比哪个」。**
> **三者缺一，误差率无从附着。**

### 4.3 记这一条是怎么来的

**不是设计时想到的。** 是 E5 拿 D3 去救一条臂、失败了，才反推出缺了哪一面。
处方被自己的实验证伪一次，然后被修正——**这个过程本身要写进论文**。
