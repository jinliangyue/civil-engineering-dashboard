# 环境锁定与工程边界（Environment Lock & Deployment Separation）

> 创建：2026-09-03（P0.9.5）
> 本文档把「研究环境」与「部署环境」分成两个独立工程对象。二者**不得互相覆盖**：
> 部署环境的任何变更都不得静默替换正式研究结果（见 §8 铁律与 `AI_PROJECT_CONTEXT.md` P0.9.5 章节）。

---

## §1 分层总览

| 层 | 文件 | 状态 | 职责 |
|---|---|---|---|
| 正式研究结果 | `docs/AI_PROJECT_CONTEXT.md` §3/§4 | **LOCKED** | P0.5 / P0.6 全部数字，唯一 baseline |
| 研究环境锁 | `requirements-research.txt` | **已锁 · 3.9 矩阵** | 复现 P0.5 / P0.6 的唯一已核验环境 |
| 部署环境说明 | `requirements-deploy.txt` | documented compatibility baseline · **Cloud 未核验** | 目标运行 `app/streamlit_app.py` 的依赖矩阵 |
| 平台配置 | `runtime.txt` | repository configuration only | `python-3.12`，**不等于** Cloud 实际版本（未独立确认） |
| 环境指纹 | `scripts/environment_fingerprint.py` | 可复现工具 | 任意环境下输出版本 + Prophet 二进制 MD5 |

---

## §2 Research 环境（唯一 reproduction baseline）

已锁环境 = 本机框架 Python（P0.5/P0.6 产出环境，2026-09-02 安装，2026-09-03 复核）：

| 包 | 版本 | 锁定理由 |
|---|---|---|
| Python | **3.9.13**（CPython, macOS arm64, framework build） | P0.5/P0.6 全部正式结果在此产出并核验 |
| pandas | 2.3.3 | 直接参与实验 |
| numpy | 2.0.2 | 直接参与实验 |
| scipy | 1.13.1 | sklearn 传递必需 + app `scipy.stats` |
| scikit-learn | 1.6.1 | XGBoost/LSTM 流程 |
| xgboost | 2.1.4 | 子进程隔离训练 |
| torch | 2.8.0 | LSTM（PyTorch） |
| prophet | 1.3.0 | 捆绑 CmdStan 2.37.0（见 §7） |
| cmdstanpy | 1.3.0 | Prophet 运行依赖 |
| stanio | 0.5.1 | cmdstanpy 依赖 |
| holidays | 0.83 | Prophet 假期特征 |
| streamlit | **未安装（验证过）** | 符合已知限制 #6，research 不需要 |
| plotly | **未安装（验证过）** | 同上 |

规则：
- 本环境是 P0.5/P0.6 **唯一**完整 reproduction baseline。禁止为部署需求修改；版本变更必须走独立的 lock 轮。
- 不追最新版。不因「统一环境」而把 3.9 强升 3.12——3.12 下已有实测漂移（R3：Prophet Final Test 11.5787% vs locked 12.0476%；Ensemble 0.3540% vs 0.3551%，6 个非 prophet 模型逐 bit 一致），reproducibility 优先级高于线上 demo 环境统一。

---

## §3 Deployment 环境（documented compatibility baseline）

- 锁文件：`requirements-deploy.txt`（10 个 pin + streamlit/plotly/akshare 宽松）。
- 核心 10 包与 research 同矩阵，**选择理由**：该 wheel 矩阵实测可装于 Python 3.12（pandas 2.3.3 / numpy 2.0.2 / prophet 1.3.0 均有 cp312 macOS arm64 wheel），与已核验 research 环境一致可最小化 live demo 指标漂移。这不意味着 app 复现正式结果——app 每次会话实时重训练（见 §8）。
- streamlit 不 pin：Streamlit Cloud 管理自身运行时版本。
- akshare 不 pin：仅 fetch 路径函数级懒加载（`src/ppi_monthly.py`），app 实际读已提交 CSV。
- **生效范围**：Cloud 当前实际安装的是旧版 unpinned `requirements.txt`。`requirements-deploy.txt` 是文档化 baseline，须在真实 Cloud runtime 验证后才能声称 production reproducibility，在此之前**不得**声称已生效。

---

## §4 Cloud runtime 状态：UNKNOWN / NOT OBSERVABLE

截至 2026-09-03，live app（https://civil-engineering-ppi.streamlit.app/）无法从公开面确认 Python 版本：

| 证据 | 观察值 | 可靠性 |
|---|---|---|
| live app HTTP | 303 → `/-/login`（登录墙）；`/_stcore/health` 亦 303 | 公开面不可观察 |
| `runtime.txt` | `python-3.12` | 仅 repository configuration，不是 Cloud confirmed runtime |
| GitHub Actions / deployments | 无 runs、无 deployment 记录（gh CLI 已登录 jinliangyue） | 说明部署不在 Actions 链路，不说明 Cloud runtime |
| commit 信息矛盾 | `d5d7a48`（9/2 09:50）称 Cloud 默认 3.14 有兼容问题、3.12 稳定；`3de3695`（13:05）称 runtime.txt 在新 Cloud 已废弃 | 仓库内叙述相互矛盾，不可作为证据 |

**结论**：Cloud Python 版本 = UNKNOWN。写死 `python-3.12` 或 `python-3.9.13` 均无证据。文档一律不写「Cloud = 3.12 / 3.9」，只写 repository configuration = 3.12。

---

## §5 Python 版本策略

- Research = 3.9.13（已锁，见 §2）。
- Cloud = UNKNOWN，由 Streamlit Cloud 控制；`runtime.txt` 保留现状（`python-3.12`），标记 repository configuration only，**不自动删改**。
- 若未来 Cloud 侧可观察（取得 Cloud 日志/设置页），再做 runtime 对齐轮：以 Cloud 实际版本为目标，用独立 venv 验证后更新本文件。

---

## §6 Environment Fingerprint

工具：`python3 scripts/environment_fingerprint.py`（stdlib only；缺包输出 NOT INSTALLED 不 crash；不输出用户名/绝对 home/token/secret；Prophet 二进制以 dist 相对路径 + MD5 报告）。

### 6.1 Research 指纹（3.9.13 · 2026-09-03 执行）

```
python_version = 3.9.13
platform       = macOS-26.6.2-arm64-arm-64bit
pandas=2.3.3 numpy=2.0.2 scipy=1.13.1 scikit-learn=1.6.1 xgboost=2.1.4
torch=2.8.0 prophet=1.3.0 cmdstanpy=1.3.0 stanio=0.5.1 holidays=0.83
streamlit = NOT INSTALLED / plotly = NOT INSTALLED
prophet_model_bin = prophet/stan_model/prophet_model.bin (md5 72d9ae8b8f399727c6c5b2f7cfeb98e5)
prophet_stan      = prophet/stan_model/prophet.stan      (md5 971f67167bc2ff8441d6678d5df3658b)
prophet_cmdstan_version = 2.37.0
```

### 6.2 Deployment 指纹（3.12 smoke venv · 2026-09-03 执行）

```
python_version = 3.12.14
platform       = macOS-26.6.2-arm64-arm-64bit
pandas=2.3.3 numpy=2.0.2 scipy=1.13.1 scikit-learn=1.6.1 xgboost=2.1.4
torch=2.8.0 prophet=1.3.0 cmdstanpy=1.3.0 stanio=0.5.1 holidays=0.83
streamlit=1.63.0 plotly=7.0.0 akshare=1.18.94 openpyxl=3.1.5 setuptools=80.10.2
prophet_model_bin = prophet/stan_model/prophet_model.bin (md5 72d9ae8b8f399727c6c5b2f7cfeb98e5)
prophet_stan      = prophet/stan_model/prophet.stan      (md5 971f67167bc2ff8441d6678d5df3658b)
prophet_cmdstan_version = 2.37.0
```

要点：legacy 矩阵在 Python 3.12 **真实安装成功**（非仅 wheel 存在性验证）；prophet 1.3.0 在 3.12 安装产出的 `prophet_model.bin` 与 3.9 逐字节一致（同 md5 72d9ae8b…）→ 相同版本矩阵下无 Prophet 二进制级漂移来源。注意 Cloud 实际装的仍是 unpinned `requirements.txt`，会解析到比本矩阵更新的版本（如 prophet 1.4.x），届时将出现 §7 所述二进制差异——这是把 Cloud 收敛到本矩阵的理由，不是本矩阵已验证到 Cloud 的证据。

---

## §7 Prophet 可复现性（二进制级证据）

| 项 | research 1.3.0 | 3.12 drift env 1.4.0 | 结论 |
|---|---|---|---|
| 捆绑 CmdStan | 2.37.0 | 2.37.0 | 同版 |
| `prophet.stan` md5 | 971f6716… | 971f6716…（相同） | 模型源码逐字一致 |
| `bin/stanc` md5 | 632d992a… | 632d992a…（相同） | 编译器一致 |
| `prophet_model.bin` md5 | **72d9ae8b8f399727c6c5b2f7cfeb98e5** | **32d8284bc5b81883f822a3c1a2b75372** | 采样二进制被重新编译 → 1.4.0 与 1.3.0 的 Prophet 输出存在二进制级差异来源 |
| bin 大小 | 3,058,256 B | 3,058,256 B | 相同 |

因此：Prophet 数值差异不能单变量归因（cmdstanpy/stanio/holidays 版本 + 编译工具链共同参与），但 1.4.0 重新编译了采样二进制是已确认事实。reproduction 必须使用 1.3.0 矩阵（§2）。R3 环境（3.12.14 + pandas 3.0.5/numpy 2.5.2/prophet 1.4.0）的 cmdstanpy/stanio/holidays 精确版本不可恢复——安装日志已删除，不猜测。

---

## §8 Research vs Live Demo（数字分层铁律）

- 正式（LOCKED，唯一可对外引用）：P0.5 Final Test Ensemble MAPE **0.3551%** / R² 0.5664；XGBoost 0.3558%；LSTM 0.4387%；P0.6 三模型 Mean MAPE（见 AI_PROJECT_CONTEXT §3/§4）。
- Live Demo（每次会话实时重训练，**不是**正式结果）：`app/streamlit_app.py` Tab 7（P0.9.6 重接后）用 `@st.cache_data` 包 `train_monthly_models(df_train_val, df_test, LSTM_BEST_PARAMS)`，内部调用 `src.analyzer.monthly_lstm.train_all_monthly_models()`——同正式实验的 108/24 切分（至 2023-12 / 自 2024-01）+ P0.3 锁定 LSTM 超参，一次完成 Prophet/XGBoost/LSTM 的 24 月滚动一步 OOS；页面评估表与 OOS 回测图全部来自该次运行，随环境（pandas/numpy/prophet 版本）漂移，属 demo 复跑值。每会话网格搜索与 3 模型集成重跑已移除（P0.9.6 方案 C），正式 7 模型集成权重与指标只出现在 Tab 5 静态区。
- App 内 Tab 5 静态引用正式数字（0.3551%/0.3558%/0.4387%/1.0192%/1.5958%/1.4087%）——这两组数字并存是**正确分层**，不是矛盾。
- 任何人在简历/文档中引用 0.3551% 时，必须注明它是「2024-2025 低波动区间的 locked research result」，不是 Cloud 每次运行的保证，也不是对未来 PPI 的预测承诺。
- 已废弃数字（不得再作为正式结果宣传）：0.24%/0.241%、0.283%、15%、旧 TF LSTM 数字、44 点手工估算、2026 fallback 预测值。

### 8.1 App 静态审计记录（P0.9.5 · 只审计未改代码）

10 点核查结论：Tab7 实时重训练 ✓（L427-467 `@st.cache_data`）；正式 locked 结果仅在 Tab5 静态引用且无歧义 ✓；metrics 全部实时计算（不经 `metrics.py` 正式模块路径之外的值）✓；数据 = repo 内官方 CSV，无 fallback ✓（P0.1 后无 `FALLBACK_PPI`）；无绝对路径 ✓（`sys.path.insert` + `Path(__file__)`）；无 runtime-specific 代码 ✓；app 运行时依赖均入 deploy baseline ✓；research-only 包（cmdstanpy/stanio/holidays 显式 import 无）不影响 app ✓。

待修记录（文案层，未改代码，留给后续文档轮）：
1. L396 错误文案「请检查 akshare 安装或网络」：实际数据来自 repo 内已提交 CSV，仅 CSV 缺失时才可能走 akshare fetch 路径——文案与代码行为不符。
2. L375/L622 数据来源叙述「akshare 从统计局抓取」：与 `load_monthly_ppi()` 本地 CSV 优先的实际路径不一致（CSV 才是演示数据源；akshare 是当初的抓取工具 + 可选刷新路径）。
3. L432/L466 spinner 时长文案（30-60 秒 / 2-3 分钟）与实测（research 3.9 全流程约 9.4s；Prophet fit 在 3.12 更慢）不符。
4. Tab7 L640-641 结论区硬编码 demo 运行数值（0.24% / 0.28% / 低 15%）且无「实时重跑 demo 值、非正式结果」的上下文注记——与 Tab5 正式数字并存时可能误读为矛盾。

5. 【实测缺陷 · P0.9.5 AppTest smoke 确认 · app 当前无法启动】app 顶层数据契约与 data/raw 实际内容不匹配，在任意环境（research 3.9 / deploy 3.12 / 任意 cwd）都失败：P0.1 commit `587f9c6` 删除了 4 个年度行业 fallback CSV 后，顶层 `load_data()`（L67-76）仍调用 `load_all_raw()`（`src/data_loader.py`，年度 schema 必填 date/price/material，L34/L87-89）——它拒绝 data/raw 中唯一剩余的月度文件（列 date/ppi_index/yoy_pct/ytd_index，无 price/material）→ 返回空 DataFrame → L81-84 双 st.error + `st.stop()`，Tab 1-7 全部不可达。已分别在 /tmp cwd 与 repo root cwd 用 AppTest 复现，失败逐字一致（非 cwd artifact、非环境差异）。附带发现：L82 错误文案「未找到官方月度 PPI 数据文件」与真实原因不符——文件存在，被拒的是年度 schema 校验。
   - Cloud 推演（不可观察）：origin/main=`3de3695` 的 data/raw 仍含这 4 个 fallback CSV（12 行/个，source 列自标「国家统计局（公开估算）」，即已删除的 44 点估算数据）→ 按代码路径 Cloud 顶层可启动，但 Tab 1-6 年度视图实际渲染的是该 fallback 数据。若未来 push 当前 HEAD 而不修此缺陷，live app 会从「可启动（用 fallback 年度数据）」变成「完全不可启动」。
   - 修复方向（未实施，超出 P0.9.5 只审计不改代码的范围）：顶层数据流改走 `src/ppi_monthly.load_monthly_ppi()`（132 点官方月度序列，Tab7 已在用），或按 Tab 拆分数据加载契约。需要独立 fix 轮 + 用户决策。

### 8.2 P0.9.6 修复状态（2026-09-03 · App Data Contract Repair）

P0.9.6 已按用户批准方案实施（commit 见 §9.3；代码 diff 只动 `app/streamlit_app.py` + `src/data_loader.py` 适配层 + `requirements-deploy.txt`，未触碰 src/analyzer / src/evaluation）：

| 8.1 待修项 | P0.9.6 处理 | 状态 |
|---|---|---|
| 5（顶层数据契约 / st.stop / Tab1-7 不可达） | 顶层 `load_data()` 改走月度 loader（data/raw 唯一真实 CSV），cwd 无关路径基于 `Path(__file__)`；仅文件缺失/格式错误才 st.stop，错误文案区分 missing / schema 不兼容 | 已修复 |
| 1（L396「检查 akshare 安装」文案不实） | Tab7 错误文案改为本地 CSV 读取异常说明，明确不联网、无 fallback | 已修复 |
| 2（数据来源叙述与本地 CSV 实际路径不符） | Tab7 方法论区改为「akshare 仅作初始抓取工具，运行期读本地 CSV」 | 已修复 |
| 4（0.24%/0.28%/15% 硬编码 demo 数字） | 已删除；方法论区重写为数字分层（Tab7 demo 复跑 vs Tab5 正式锁定） | 已修复 |
| 3（spinner 时长文案） | 新文案「首次约 30-90 秒」，3.9 venv 实测冷缓存含 LSTM 滚动约 1-3 分钟，仍属近似说明 | 部分（文案诚实化，非精确计时） |

Tab 7 设计决策（用户选择方案 C 轻量接线）：保留每会话三模型实时训练 demo（同切分同锁定超参，非正式实验复跑）；预测图改为 2024-01~2025-12 OOS 回测图（真实 test_pred vs test_actuals）；移除每会话 LSTM 网格搜索与 3 模型集成重跑，改为静态文字指向 Tab 5 正式结果；方法论修正 demo/正式数字分层。

---

## §9 变更规则

1. **改 research 锁**（requirements-research.txt 或 §2 矩阵）：必须先停下报告，开独立 lock 轮，重跑 P0.5/P0.6 全量核验后才能声称新 baseline。
2. **改 deploy 锁**：先在目标 Python（当前 3.12 smoke）验证安装 + `compileall` + 启动检查；若声称 Cloud 复现，必须先取得 Cloud runtime 证据。
3. **Cloud 部署链路**：本地 research 修复（P0.5/P0.6/path fix/P0.9.6，本地 main 领先 origin/main 15 个 commit）**尚未 push**；Cloud 代码基线停留在 `3de3695`。任何 push 都需用户明确指示——本阶段禁止 push。
4. 环境指纹必须用 `scripts/environment_fingerprint.py` 产生；版本信息不得手抄。

---

## §10 其他环境事实

- research 环境安装于 2026-09-02（base site-packages），P0.9.3 期间曾另建 `.venv-p093-legacy` 副本，该副本已于 9/3 删除；**现存唯一 research 安装 = 框架 Python 3.9 base**，勿再依赖 `.venv-p093*`。
- legacy（未接线）模块 `src/features/` `src/trend/` `src/seasonality/` `src/forecast/` `src/data_cleaner.py` 含 statsmodels import，不在正式路径，不进 research 锁；删除与否不影响 reproduction。
- `tests/` 为空目录；`src/evaluation/test_metrics.py` 是唯一单测（9/9）。
