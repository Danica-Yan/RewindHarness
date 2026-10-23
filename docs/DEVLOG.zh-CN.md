# RewindHarness 开发日志（中文）

> 这是项目开发过程中的完整设计笔记和实验记录，按里程碑 W1–W8 组织。英文概览见仓库根目录的 [README](../README.md)。文中提到的“沙箱”指最初的开发环境。


可回放、可分叉（反事实）、带崩溃恢复、带根因归因、带失败分类、带两轮数据飞轮、支持并行 tool call 的 agent 运行时，接了真实 tau-bench 零售域（真实工具、真实数据、真实 reward），外加一套可选的真实分布式执行层（Postgres + Redis）和一套真实 chaos 测试（真 `kill -9`、真 Toxiproxy 网络故障注入）。**核心 136 个测试**（49 个纯离线 + 25 个需要真实 Postgres/Redis（含本轮新增的 2 个分布式漏洞回归测试） + 7 个 W6 chaos 测试（2 个纯离线，5 个需要真实 Postgres 和/或 Toxiproxy）+ 16 个 W7 失败分类/归因准确率测试 + 10 个 W8 数据飞轮测试 + 5 个 W1 followup 的 tau-bench adapter 测试（需要可选的 tau_bench 依赖）+ 3 个 DeepSeek 消息/工具转换测试 + 10 个真实 tau-bench 失败归因分类测试 + 6 个多 trial 对比统计方法测试 + 5 个并行 tool call 测试，服务/依赖不可达时自动跳过、不影响其它测试）。

## 架构

```mermaid
flowchart TB
    subgraph Harness["harness（loop.py）"]
        Loop["AgentRunner<br/>预算/重试/修复/loop检测"]
        Tools["ToolRegistry<br/>pydantic校验+幂等"]
    end
    subgraph Durability["确定性 + 崩溃恢复"]
        Trace["TraceWriter<br/>seq严格递增"]
        Snap["SnapshotStore<br/>内容寻址"]
        Ckpt["Coordinator / PgCoordinator<br/>lease+fencing+3级checkpoint"]
    end
    subgraph CounterFactual["回放 / 分叉 / 归因"]
        Replay["replay_run"]
        Fork["fork_run"]
        Attr["bisect_attribute<br/>+ flip_rate_via_fork"]
    end
    subgraph Dist["分布式执行层（W5，真实 Postgres+Redis）"]
        Gateway["LLMGateway<br/>缓存+pub/sub single-flight<br/>RPM/TPM限流+熔断"]
        Stream["StreamEventBus<br/>consumer group+XCLAIM"]
    end
    subgraph Chaos["chaos 测试（W6，真实故障）"]
        Kill["真实 kill -9<br/>跨进程恢复"]
        Toxi["Toxiproxy<br/>真实网络故障"]
        Inv["不变量检查器"]
    end
    subgraph Flywheel["数据飞轮（W7/W8）"]
        Classify["classify() 失败分类"]
        Cluster["cluster() 按类别+工具聚类"]
        Eval["attribute-eval<br/>合成ground truth准确率"]
        Round["两轮 gate: dev验证→merge→heldout报告"]
    end

    Loop --> Trace
    Loop --> Tools
    Trace --> Snap
    Loop -.checkpointer.-> Ckpt
    Trace --> Replay
    Snap --> Fork
    Fork --> Attr
    Ckpt --> Gateway
    Ckpt --> Stream
    Kill --> Ckpt
    Toxi --> Gateway
    Toxi --> Ckpt
    Kill --> Inv
    Trace --> Classify
    Classify --> Cluster
    Cluster --> Round
    Attr --> Eval
    Cluster --> Eval
```

（GitHub 会原生渲染这个 Mermaid 图；本地看纯文本也不影响理解各层的依赖方向：harness 产生 trace → trace 既喂 checkpoint 也喂 replay/fork/归因 → 分布式层是 checkpoint 的真实基础设施实现 → chaos 测试真实地打这两层 → 失败分类/聚类/归因/飞轮消费 trace 和归因结果，形成闭环。）

```bash
pip install -e ".[dev]"
pytest -q
rewind demo --out runs                       # 跑一次失败 → 回放 → 在第 2 步分叉修正
rewind ckpt-demo --out runs/ckpt-demo        # 跑到一半崩溃 → 第二个 worker 从 checkpoint 接管
rewind attribute-demo --out runs/attr-demo   # 二分归因：在两个候选步骤里找到真正的根因步
rewind attribute-eval                        # 归因准确率：合成 ground truth，5 个不同根因位置
python scripts/failure_batch.py              # 失败分类批量报告：分布 + top clusters
rewind replay runs/demo-base/trace.jsonl --snapshots runs/snapshots
# 真实模型（需 pip install -e ".[anthropic]" 或 ".[deepseek]"，对应 API key，未包含在离线测试中）
rewind run --task "I'm u1, cancel my headphones order"                    # 默认 --provider anthropic
rewind run --task "I'm u1, cancel my headphones order" --provider deepseek
rewind fork runs/run-xxxx/trace.jsonl --at 2 --patch '{"tool_calls":[{"name":"get_order","args":{"order_id":"O1"}}]}'

# 分布式执行层（需 pip install -e ".[dist]" 和真实 Postgres/Redis，见下方 dist/ 一节）
pytest -q tests/test_dist_pg.py tests/test_dist_gateway.py tests/test_dist_eventstream.py

# W6 chaos 测试（需 pip install -e ".[dist,chaos]"、真实 Postgres、以及 toxiproxy-server 在 PATH 上，见下方 W6 一节）
pytest -q tests/test_chaos.py
python scripts/chaos_scale_experiment.py --jobs 40 --workers 1,2,4,8
python scripts/chaos_recovery_experiment.py --trials 20 --lease-seconds 2

# W8 数据飞轮（纯离线，不需要任何外部服务）
python scripts/make_task_split.py                  # 生成+持久化 60/20/20 任务切分到 data/task_split.json
python scripts/data_flywheel_report.py              # 读切分文件，跑两轮飞轮，打印 baseline/gate/heldout 报告
pytest -q tests/test_data_flywheel.py

# W8 CI 回归门禁脚本（本地直接可跑，不需要真的有 GitHub Actions）
bash scripts/ci_pr_gate.sh     # 快速集：排除 dist/ 三个文件和 test_chaos.py，不需要任何外部服务
bash scripts/ci_nightly.sh     # 全量集：需要真实 Postgres/Redis/toxiproxy-server（同上方设置）

# W1 followup：真实 tau-bench 零售域 adapter（不在 PyPI 上，见下方设置）
pip install "tau_bench @ git+https://github.com/sierra-research/tau-bench.git"
pytest -q tests/test_taubench_adapter.py
python scripts/taubench_baseline.py --split dev     # 20 个任务，~20s，纯黄金轨迹回放，不是模型 baseline
python scripts/taubench_baseline.py --split all     # train+dev+test 全量，train 500 个任务约 8-9 分钟
```

## 运行分布式测试需要的一次性设置

```bash
# 这个沙箱容器自带 postgresql-16 / redis-server，只是默认没启动；换一台机器就是
# apt/brew 装好之后同样这几步。
service postgresql start
service redis-server start

# 建一个跑测试用的角色和库（密码、库名要跟下面的默认 DSN/URL 对上，或者用环境变量覆盖）
sudo -u postgres psql -c "CREATE ROLE rewind WITH LOGIN PASSWORD 'rewind' SUPERUSER;"
sudo -u postgres createdb -O rewind rewind_test

pip install -e ".[dist]"
pytest -q tests/test_dist_pg.py tests/test_dist_gateway.py tests/test_dist_eventstream.py
```

默认连接串是 `postgresql://rewind:rewind@127.0.0.1:5432/rewind_test` 和 `redis://127.0.0.1:6379/0`，可以用 `REWIND_PG_DSN` / `REWIND_REDIS_URL` 环境变量覆盖。这三个测试文件在服务连不上时会**跳过而不是报错**（`pytest.skip`，带具体原因），所以正常的 `pytest -q` 不需要这两个服务也能全绿。

## 运行 W6 chaos 测试需要的一次性设置

`toxiproxy-server`/`toxiproxy-cli` 是单个静态编译的 Go 二进制，不需要 Docker（这个沙箱没有 docker daemon，见下面的坑）；直接从项目自己的 GitHub Releases 下载即可（出站 HTTPS 允许访问 GitHub）：

```bash
curl -sSL -o /usr/local/bin/toxiproxy-server \
  https://github.com/Shopify/toxiproxy/releases/download/v2.9.0/toxiproxy-server-linux-amd64
curl -sSL -o /usr/local/bin/toxiproxy-cli \
  https://github.com/Shopify/toxiproxy/releases/download/v2.9.0/toxiproxy-cli-linux-amd64
chmod +x /usr/local/bin/toxiproxy-server /usr/local/bin/toxiproxy-cli

pip install -e ".[dist,chaos]"
pytest -q tests/test_chaos.py
```

`toxiproxy-server` 换其它 CPU 架构就把文件名里的 `linux-amd64` 换成对应的（发布页有 `linux-arm64`/`darwin-arm64` 等）。`ToxiproxyController`（`dist/toxi.py`）会在需要时自己把它拉起来当子进程，不需要提前手动启动；测试文件在这个二进制不在 PATH 上时会跳过而不是报错。

> **踩的一个坑（这次是个好消息）**：一开始以为这个沙箱没有 docker daemon（W5 时是这么记的），所以直接假设 Toxiproxy 得走"下载静态二进制"这条路。后来试了一下发现 `dockerd` 其实是能起来的——但 `docker pull` 会被出站代理的域名白名单挡住（`ghcr.io` 不在白名单里，只有 GitHub 和几个包仓库域名放行）。所以最终选择还是静态二进制，但原因跟最初想的不一样：不是"没有 docker"，而是"有 docker 但拉不到镜像"——这个区别值得记录，因为下次换个网络环境，docker 路线可能就通了。

## 运行 tau-bench adapter 测试需要的一次性设置

`tau_bench` 不在 PyPI 上，`pip install tau_bench` 找不到包，必须直接从它自己的仓库装：

```bash
pip install "tau_bench @ git+https://github.com/sierra-research/tau-bench.git"
```

> **踩的一个坑（这次是沙箱特有的，别的机器大概率不会碰到）**：这个沙箱的系统 setuptools 被 Debian 打了个补丁，装这种只有 `setup.py`（没有 `pyproject.toml`）的老式包时会在 `install_lib.finalize_options` 这一步报 `AttributeError: install_layout`——跟 tau-bench 自己的代码没关系，是这台机器的包管理工具链的 bug。绕过办法是不走 `pip install`，直接把仓库目录加进 `sys.path`（`REWIND_TAUBENCH_SRC` 环境变量，见下）；在一台正常的机器上，上面那条 `pip install git+...` 应该是直接能用的，这条坑只记录在这里以防你也碰到同样的报错。

```bash
# 如果 pip install 在你的机器上也失败，退回手动 clone + 加路径：
git clone https://github.com/sierra-research/tau-bench.git /path/to/tau-bench
export REWIND_TAUBENCH_SRC=/path/to/tau-bench
pytest -q tests/test_taubench_adapter.py
```

`env/taubench_adapter.py` 的 `_ensure_importable()` 会先试正常的 `import tau_bench`，失败了再去看 `REWIND_TAUBENCH_SRC` 有没有设置、把它加进 `sys.path`；两条路都失败才报错，报错信息里带着上面这条安装命令。

## 模块

| 文件 | 作用 |
|---|---|
| `events.py` `trace.py` | 事件 schema、追加写 JSONL trace、`seq` 严格递增、`resume_from` 截断续写 |
| `snapshot.py` | 内容寻址存储（sha256），env 状态和 checkpoint bundle 共用 |
| `tools.py` | tool 注册、pydantic 校验、结构化错误、按 `action_id` 幂等 |
| `providers.py` | Provider 抽象、ScriptedProvider、故障注入、Anthropic/DeepSeek 适配器 |
| `loop.py` | agent loop：预算、重试退避、fallback、输出修复、参数自纠、loop 检测、checkpoint 挂钩点、并行 tool call（同一 turn 里 2+ 个非 mutating 调用并发执行，mutating 的仍严格顺序） |
| `replay.py` | 确定性回放：记录的 LLM 输出驱动 harness，逐请求校验 hash，再做 trace diff |
| `fork.py` | 反事实分叉：从第 k 步的 snapshot + 历史恢复，可替换该步动作后 live 续跑 |
| `checkpoint.py` | `Coordinator`（sqlite：队列 + lease/fencing + checkpoint 日志）、`StepCheckpointer` |
| `worker.py` | 单次 job 执行：claim → 冷/热启动 → 跑 → complete；崩溃只是这个函数没返回 |
| `flywheel/attribute.py` | 反事实归因：二分找根因步，纯算法和 fork 接线分开两层，各自可单测 |
| `flywheel/failures.py` | 失败分类（方案 4.6 表）、按 (类别,工具) 聚类、`patch_for_step` 自动配对 |
| `flywheel/eval_attribution.py` | 归因算法在合成 ground truth 上的准确率评测（精确匹配 / ±1 步容忍） |
| `_await.py` | `maybe_await`：让 `worker.py`/`StepCheckpointer` 同时兼容同步（sqlite）和异步（Postgres）coordinator |
| `dist/pg_coordinator.py` | `PgCoordinator`：和 sqlite 版同协议的 asyncpg 实现，`claim()` 用真正的 `FOR UPDATE SKIP LOCKED` |
| `dist/gateway.py` | `LLMGateway`：Redis 实现的响应缓存 + pub/sub single-flight + RPM/TPM 双维度限流 + 共享熔断器，套在任意 Provider 外面 |
| `dist/eventstream.py` | `StreamEventBus`/`TraceStreamConsumer`：Redis Streams 事件总线，consumer group + `seq` 去重 + `XAUTOCLAIM` 崩溃恢复 |
| `dist/toxi.py` | `ToxiproxyController`：管理真实 `toxiproxy-server` 子进程 + HTTP admin API，注入真实网络故障（延迟/超时/重置/限速） |
| `dist/invariants.py` | chaos 轮后的不变量检查：`seq` 连续性、无卡死 lease、最终 env hash 一致 |
| `dist/chaos_worker.py` | 可作为**真实 OS 子进程**启动的 worker，跑在真实 Postgres + 文件系统上，供 `tests/test_chaos.py` 真 `kill -9` |
| `flywheel/tasks.py` | W8：任务生成（固定 seed，可复现）+ 60/20/20 train/dev/heldout 切分，`save_split`/`load_split` 落盘 JSON |
| `flywheel/heuristic_provider.py` | W8：确定性启发式 provider（真实 LLM 的替身，见下方 W8 一节），两个独立可开关的"bug" |
| `flywheel/data_flywheel.py` | W8：两轮飞轮——在 train 上跑+分类+聚类，取 top cluster 找候选修复，在 dev 上门禁验证，通过则合并并报告 heldout |
| `env/__init__.py` | 离线玩具零售环境（`env.py` 原地改成了包，内容不变，导出名不变） |
| `env/taubench_adapter.py` | W1 followup：真实 tau-bench 零售域 adapter——真实工具、真实数据、真实 reward 约定，黄金轨迹回放校验（不是模型 baseline，见下方 W1 一节） |
| `scripts/ci_pr_gate.sh` | W8：快速回归门禁，不需要外部服务，PR 应该挡在这上面 |
| `scripts/ci_nightly.sh` | W8：全量回归（含 dist/ 和 chaos），需要真实 Postgres/Redis/toxiproxy |
| `.github/workflows/ci.yml` | W8：真实、valid 的 GitHub Actions 配置，PR 触发快速门禁、nightly/手动触发全量（见下方"已知局限"对这份配置未被真正跑过的说明） |
| `scripts/taubench_baseline.py` | W1 followup：跑真实 tau-bench 任务的黄金轨迹回放，train/dev/test 三个 split，报告通过率 + 诚实跳过的 outputs 任务数 |
| `scripts/run_taubench_deepseek.sh` | 真实 tau-bench baseline：调用 tau-bench 自己的 `run.py`（不是 rewind 代码）+ DeepSeek，需要 `DEEPSEEK_API_KEY` |
| `flywheel/taubench_failures.py` | W1 followup：真实 DeepSeek 失败样本（test split 34 个）的归因分类——读 tau-bench 自己的 `EnvRunResult` 格式，和 `flywheel/failures.py` 读的 `AgentRunner` trace 格式不同，所以是独立模块 |
| `scripts/taubench_failure_report.py` | W1 followup：对一份真实 tau-bench 结果 json 跑失败分布报告，见下方"真实失败归因"一节 |
| `scripts/taubench_round1_wiki.patch` | W1 followup：针对 tau-bench retail policy 的一个真实 bug（改商品前改地址的顺序约束）的真实 patch，已验证能干净应用 |
| `scripts/run_taubench_round1.sh` | W1 followup：打上面那个 patch 后，用和 baseline 完全相同的协议重跑 test split，见下方"真实失败归因 round 1"一节 |
| `scripts/taubench_round1_compare.py` | W1 followup：对比 baseline 和 round 1 两份结果文件，报告 pass^1 delta、修好/改坏的任务（`scripts/taubench_compare.py` 的一个预设薄封装） |
| `scripts/taubench_compare.py` | W1 followup：通用的两份 tau-bench 结果文件对比工具，报告双向翻转和"净值远小于总翻转量"的噪声警告——round 1 对比和下面的噪声基线对比共用同一份逻辑 |
| `scripts/run_taubench_noise_floor.sh` | W1 followup：噪声基线对照——先把 `wiki.md` 还原成未打 patch 的原始版本，再用和 baseline 完全相同的协议重跑一遍，给"两次完全相同条件的运行之间抖动有多大"提供一个真实数字 |
| `scripts/run_taubench_multitrial.sh` | W1 followup：对 unpatched/patched 各跑 `--num-trials N`（tau-bench 原生支持，一个文件里每任务 N 个 trial），给真正的多 trial 对照提供数据 |
| `scripts/taubench_multitrial_compare.py` | W1 followup：多 trial 版对比——pooled pass^1、真正的 `pass^k`（k>1）、配对 bootstrap 置信区间，区分"有真实差异"和"证据不足"，不混为一谈 |

## 关键设计（面试要点）

- **回放靠记录，不靠 temperature=0**：每次 provider 尝试（成功或失败）都记为 `llm_call`；回放时 `ReplayProvider` 校验 `request_hash`，一旦 harness 构造的 prompt 变了立刻报错。
- **确定性边界**：时间走注入的 clock，抖动来自以 run_id 为种子的 RNG，回放中 sleep 为空操作。
- **Fork 点**：每个 step 开头记 `step_start(env_hash)`；分叉时从该 snapshot + 由 `message` 事件重建的历史恢复，循环计数/token 用量由 `LoopState.from_events` 重建。
- **幂等**：mutating tool 的结果按 `action_id=s{step}:c{idx}` 记在 env 内，重复执行直接返回旧结果。

### Checkpoint / 崩溃恢复（本轮新增）

**提交顺序**严格对应方案 4.4：LLM 给出动作 → 落 `intent` checkpoint → 逐个执行 tool（每个成功后落一次 `tool` checkpoint）→ 全部完成后落 `step` checkpoint。三种 checkpoint 共用一张 sqlite 表，按 `(run_id, step, kind)` 存，`latest_checkpoint` 按 `step DESC` 再按 `step > tool > intent` 取最新。

**Lease + fencing**：`Coordinator.claim()` 原子地把任务标成 `LEASED` 并把 `fencing_token` 自增；写 checkpoint / 标记完成时都要求调用方带着自己拿到的 token，token 对不上（说明有新 worker 顶替了）直接拒绝写入，旧 worker 在 `StepCheckpointer.save` 里收到 `RunAbort("fenced_out", ...)`，自然被现有的异常处理吞掉退出，不会带着过期状态继续跑。

**为什么"整步重放"不会导致有副作用的 tool 被真正调用两次**：一开始的实现只在整步结束后 checkpoint 一次，测试立刻抓到问题——如果崩溃发生在"tool 真的执行完，但整步 checkpoint 还没写"这个窗口，resume 会重放整个 step，`cancel_order` 就被真实调用了两次（`tests/test_checkpoint.py` 里 `counting_registry()` 用一个跨进程存活的计数器模拟"外部世界"，独立于会被丢弃的内存态 env，专门抓这种问题）。修复方式是**在每个 tool 调用后都落一次更细粒度的 `tool` checkpoint**，它保存的是这次调用后的 env（连同 env 内的幂等台账 `_applied`）。resume 时统一从 idx=0 重放这一步的动作：已经落过 `tool` checkpoint 的调用，它的 `action_id` 已经在恢复出来的 `env.applied` 里，`ToolRegistry.execute` 直接返回缓存结果、不再重跑 `fn`；没被 checkpoint 的调用照常真实执行一次。也就是说恢复真正依赖的是**幂等台账要和它保护的 env 一起被 checkpoint**，而不是"整步不可分割"这个假设。

**Torn write**：worker 挂了之后，trace 文件上可能残留没有任何 checkpoint 背书的事件（比如 `step_start` 已经写了但那一步的 LLM 调用还没成功过）。`TraceWriter.resume_from(path, run_id, upto_seq)` 按 checkpoint 记的 `seq` 截断文件、重新以追加模式打开，新事件的 `seq` 紧接着继续递增——测试里验证了截断前后 `seq` 依然严格连续。

**测试覆盖**（`tests/test_checkpoint.py`，49 个测试里有 20 个是 checkpoint 相关）：先跑一遍不崩溃的流程，收集出这次 run 会经过的全部 checkpoint 钩子点（`after_intent_checkpoint` / `after_tool_call` / `after_step_checkpoint`，共 7 个），然后**对每一个点各跑一次"精确在这里崩溃 → 等 lease 过期 → 换一个 worker 恢复"**，断言：最终状态判分正确、`cancel_order` 的真实调用计数恰好是 1、trace 的 `seq` 连续。另外单独测试了 lease 未过期时抢不到、heartbeat 续约、僵尸 worker 的 checkpoint 写入和 complete 都被 fencing token 挡住。

### 反事实归因（本轮新增，方案 4.7 / W7）

`bisect_attribute` 是纯算法：只依赖一个"在候选步骤 k 打补丁并续跑，翻转成功的比例是多少"的 oracle 函数，跟 agent/fork/env 完全解耦，所以能用一个假 oracle 直接构造出单调、不单调、全部失败等场景，不用真的跑 agent 就能测清楚边界行为。`flip_rate_via_fork` 是把这个 oracle 接到真实 `fork_run` 上的那一层：在候选步骤分叉、打补丁、`n_samples` 次 live 续跑，统计翻转成功的比例。

**二分 + 邻域校验**：先看候选列表里最后一个能不能翻转（不能就说明这条失败轨迹没有单一步骤的根因，直接返回 `None`），能的话对候选下标做标准二分，找到第一个翻转率达标的位置，再往前用一个小窗口（默认 2 步）线性校验有没有更早的候选也能翻转——因为"越早干预越容易翻转"只是近似假设，一条轨迹完全可能有两个独立错误。测试里专门构造了"两个都能翻转、二分探测点会落在后一个"的场景，验证校验窗口能把答案改回真正更早的那个，并把 `monotonic` 标成 `False` 告诉调用方这条轨迹违反了假设。

`rewind attribute-demo` 复用 `demo` 的失败轨迹，对步骤 1（"修正"成和原来一样，等于没修）和步骤 2（真正的根因）各定义一个补丁，二分只用 2 次 oracle 调用就找到根因是步骤 2 而不是步骤 1。

### 分布式执行层：Postgres + Redis（本轮新增，方案 4.4/4.5，W5）

上面的 checkpoint 机制此前只在 sqlite 上验证过——单连接、单进程，测的是**协议**而不是并发本身。这一轮把同一套协议接到真实 Postgres 和真实 Redis 上，跑在这个沙箱容器自带的 `postgresql-16` / `redis-server` 上（没有 docker daemon，所以是直接跑系统服务，不是 compose；`dist/` 的代码本身跟部署方式无关）。

**`PgCoordinator`**：和 `checkpoint.Coordinator` 同一套方法名、同一套 SQL 语义，但 `claim()` 用的是 Postgres 原生的
```sql
SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1
```
这是 sqlite 单连接版本完全没法演示的部分——`test_concurrent_claims_from_real_connections_never_double_deliver` 用 `asyncio.gather` 真的**并发**发起 8 个 claim 请求打同一个连接池，断言 8 个任务被 8 个不同 worker 各拿到一个、没有重复。`worker.py`/`StepCheckpointer` 完全没为了支持 Postgres 而改动业务逻辑——靠的是 `_await.maybe_await`：同步的 sqlite Coordinator 直接返回结果，异步的 `PgCoordinator` 返回 coroutine，调用方统一 `await maybe_await(...)`，两种后端用的是同一份 `run_worker_step`。`checkpointer.save` 本身也改成了 `async def`（之前是同步方法），这样它才能在需要时真的对 Postgres 做 `await` 网络往返，而不是假装异步。

> **踩的一个坑**：`test_dist_pg.py` 单独跑全绿，合到整个测试套件里跑却失败，报"unknown snapshot"。原因是 `rewind_test` 这个库在多次 `pytest` 进程之间是**持久化**的（不像内存 sqlite，每个测试进程结束就没了），而 `claim()` 语义是"随便捞一个可用的任务"，不认 run_id——于是某次测试捞到的是**上一次测试进程**遗留下来的、状态是 PENDING 但 `initial_env_ref` 指向那次测试自己内存里 `SnapshotStore` 的旧任务，这次测试的 `SnapshotStore` 里当然找不到那个 hash。修法很直接：每个测试连接后先 `TRUNCATE jobs, checkpoints`，把 `rewind_test` 当成纯粹的测试用后即弃空间。这条本身就是分布式系统测试里一个常见坑：**共享持久化状态的测试，必须显式清零，不能假设"每次跑都是干净的"**。

**`LLMGateway`**：包一层在任意 Provider 外面，`AgentRunner` 不用改。四件事：
- **缓存**：key 是 `request_hash(messages, tools, params)`，命中直接读 Redis，不发真实请求。
- **single-flight（pub/sub，不是轮询）**：用 Redis `SET key NX PX` 抢一把锁；抢不到的请求 `SUBSCRIBE` 一个 `<key>:done` 频道等持有者把结果写进缓存，而不是自己也发一次请求，也不是隔一段时间 `GET` 一次。持有者无论成功还是失败，在 `finally` 里都会 `PUBLISH` 一次——失败也要广播，否则等待方会傻等到 `wait_timeout` 超时才回退到自己发请求；`test_single_flight_waiter_wakes_on_failure_and_retries_itself` 专门验证"leader 调用失败"这条路径下 follower 能在 1 秒内醒来并自己重试，而不是等满 `wait_timeout`。`subscribe` 和"再查一次缓存"之间有个经典的竞态窗口（leader 可能在我们抢锁失败和真正 `subscribe` 成功之间就已经发布完了），`_wait_for` 里订阅前后各查一次缓存来堵上这个窗口。测试里 6 个完全相同的并发请求，真实 provider 只被调用 1 次。
- **限流（RPM+TPM 双维度，原子）**：`TokenBucket` 是单维度版本（Lua 脚本实现，`HMGET`/`HMSET` 做原子读改写）；`DualTokenBucket` 用一段脚本同时扣 RPM 和 TPM 两个桶，要么两个都够、一起扣，要么都不扣——避免"RPM 够但 TPM 不够"时已经把 RPM 配额花掉了却还是被拒的账目不一致。`test_dual_bucket_is_all_or_nothing_neither_dimension_partially_spent` 直接验证了这一点。多个 worker 进程共用同一个 Redis 时看到的是同一组桶，不是各进程自己的配额。
- **熔断器（Redis 共享状态）**：`CircuitBreaker` 有两条触发路径——`record_failure()` 累计到 `failure_threshold` 次才跳闸（一段 Lua 脚本原子自增+判断+开闸），或者 `record_rate_limited()`（收到一次 429 直接开闸，不用等到阈值，因为 429 本身就是明确信号）。开闸状态就是一个带 TTL 的 Redis key，TTL 过期 = "半开"（这里没实现严格意义上的单次试探型半开，是用 TTL 到期简化的，README 这里如实标注）。一个 worker 发现下游挂了/被限流，所有 worker 立刻通过 Redis 共享这个状态一起退避，不用各自独立踩一遍。`CircuitOpen` 异常在熔断打开时直接短路，根本不调用内层 provider。
- **锁的释放是 compare-and-delete**（`GET` 校验完再 `DEL`），不是无脑 `DEL`：`test_unlock_is_compare_and_delete_not_a_blind_delete` 故意让锁过期后被另一个 worker 抢到，再让第一个 worker 尝试释放它自己那把（其实已经不是它的了）锁，断言第二个 worker 的锁没被误删——这是"锁必须证明自己还持有才能释放"这条分布式锁的标准原则，写成了一个具体的测试而不只是注释里一句话。

> **踩的一个坑**：熔断器的 Lua 脚本一开始直接把 `cooldown_seconds`（浮点数，比如测试里的 `0.2`）传给 Redis 的 `EX`/`EXPIRE`，而 Redis 要求整数秒，报 `value is not an integer or out of range`。修法是脚本内 `math.max(1, math.ceil(...))`——顺带也是一个更合理的语义：冷却时间不足 1 秒就当 1 秒。这是新测试（不是手写代码走查）第一次跑就抓到的真实 bug。

**`StreamEventBus`/`TraceStreamConsumer`**：trace 事件走 Redis Streams（方案里 NATS JetStream 的替代项），每个 run 一条独立的 stream，consumer group 消费。Streams 的 consumer group 语义是至少一次投递，所以消费端必须自己去重——`TraceStreamConsumer` 按 `Event.seq` 去重，`test_redelivered_duplicate_is_deduped_by_seq` 直接把同一个 `seq` 发布两次模拟重复投递，断言第二次 `drain()` 不会把它当新事件返回（但仍然会 ack）。

**崩溃恢复：`XCLAIM`（真实的，不是模拟的）**：`read_without_ack()` 只读不 ack，用来在测试里模拟"consumer 读完一批、还没处理完就死了"——这些条目会一直挂在 Redis 的 pending-entries-list（PEL）里，直到有人认领。`reclaim_stale(min_idle_ms)` 用 `XAUTOCLAIM` 把空闲超过 `min_idle_ms` 的 pending 条目转给当前这个（新的/重启后的）consumer，再走一遍和 `drain()` 一样的按 `seq` 去重逻辑后 ack。`test_reclaim_recovers_events_read_but_never_acked_by_a_dead_consumer` 是第一个真正验证这条路径的测试：一个 consumer 读了 3 条不 ack 就"死"了，另一个 consumer 先用很大的 `min_idle_ms` 确认"还没到时间不该认领"，再用 `min_idle_ms=0` 认领到全部 3 条，认领后第二次再认领应该拿到空。此前 `test_redelivered_duplicate_is_deduped_by_seq` 只是重复发布同一个 `seq` 来伪造"重复投递"的*结果*，并没有真的触发 Redis 自身的 PEL 重投递机制——现在这条路径补上了。

### 分布式层的两个真实漏洞（本轮新增，代码审查发现，非方案要求）

上面 W5 那一节验证的是协议正确性（不重复投递、不重复调用）；这一轮回头审查这两份代码本身，找到两个协议正确但**真实生产环境会暴露**的问题，不是方案清单上的条目，是读代码读出来的。

**漏洞 1：`jobs` 表没有任何索引支撑 `claim()` 的查询**。`claim()` 的 `WHERE status='PENDING' OR (status='LEASED' AND lease_expires_at < now())` 在只有测试规模的几十条数据时完全看不出问题，但表里只要攒了几万条历史 `COMPLETED`/`FAILED` 记录，每一次 claim 就会退化成全表扫描。真实测出来的数字（本地 Postgres，5 万条历史记录 + 20 条活跃任务）：

```
修之前：Seq Scan on jobs，Rows Removed by Filter: 50000，Execution Time: 8.228 ms
修之后：Index Scan using jobs_claimable_idx，Execution Time: 0.046 ms
```

修法是加一个**部分索引**（`WHERE status IN ('PENDING', 'LEASED')`），不是普通索引——`now()` 不能出现在索引谓词里（不是 immutable 函数），所以索引没法把 `lease_expires_at < now()` 那一半也编码进去，但只用 `status` 过滤就已经把候选集从"这张表有过的每一条记录"砍到"当前还活跃的那一小部分"，后面那半句 `lease_expires_at` 判断就变成在这一小部分上做的廉价过滤，不再是全表扫描后再排序。`tests/test_dist_pg.py` 新增的 `test_claim_query_uses_the_partial_index_not_a_seq_scan` 直接读 Postgres 自己的 `EXPLAIN` 输出断言用了索引、没有 `Seq Scan`——验证查询计划而不是验证墙钟时间，因为时间会受机器负载噪声干扰，计划不会。

**漏洞 2：Gateway 的 single-flight 锁没有心跳续期，但 job 的 lease 有**——同一个仓库里两套"租约"语义不一致。`Coordinator.heartbeat()` 让一个还在正常工作的 worker 不会因为慢而丢失 lease；`LLMGateway` 抢的那把 single-flight 锁（`lock_ttl_ms` 默认 15 秒，固定不续）完全没有对应机制。真实 LLM 调用超过锁的 TTL 是很常见的场景（大模型长输出、网络抖动），锁一旦在 leader 还在等响应的时候就过期，第二个 worker 的 `SET NX` 会直接抢到锁——single-flight 被悄悄击穿，同一个请求被真实调用了两次，直接体现为真金白银多付一次 API 账单，而且原来的测试套件完全没覆盖这个场景（所有测试里 provider 的 delay 都远小于 `lock_ttl_ms`）。

修法是照着 `heartbeat()` 的思路给这把锁也加一个后台续约协程（`_renew_lock_until_cancelled`）：持锁期间每隔 `lock_ttl_ms // 3` 续一次 TTL，用和 `_UNLOCK_LUA`（compare-and-delete）同样的"先核对 token 再操作"原则写一个 `_EXTEND_LOCK_LUA`（compare-and-extend）——万一这个续约真的因为某次 tick 晚了而错过了窗口、锁已经被别人合法抢走，续约脚本也只会是空操作，不会把别人的锁偷回来。续约协程在 `finally` 里**先取消、等它退出，再释放锁**，避免续约和释放之间的无意义竞争。

`tests/test_dist_gateway.py` 新增的 `test_slow_real_call_keeps_the_single_flight_lock_via_heartbeat` 故意把 `lock_ttl_ms` 设成 100ms、provider 的真实延迟设成 400ms——心跳默认每 33ms 续一次，理应在整个调用期间把锁续活。为了证明这不是一个自我验证的假测试，专门把心跳协程临时去掉重跑了一遍：真的复现了 bug（`call #2`，provider 被真实调用了两次），把心跳加回来后 provider 只被调用 1 次——证明测试本身真的在测这个问题，不是摆设。

```
心跳去掉（模拟修复前）：assert 'call #2' == 'call #1'  → 断言失败，复现了漏洞
心跳恢复（修复后）：     provider.calls == 1            → 通过
```

### Chaos 测试：真 `kill -9` + 真 Toxiproxy（本轮新增，方案 W6）

方案对 W6 的要求很明确："真正 `kill -9` 一个操作系统进程 + 网络故障注入（Toxiproxy），而不是本轮单进程内异常模拟崩溃这种等价但没有跨进程边界的验证方式"（这是 W5 阶段写在"下一步"里的原话）。所以这一轮**没有**再复用 `SimulatedCrash`（那是 `test_checkpoint.py` 用来穷举每一个 checkpoint 钩子点的工具，价值在穷尽性，但它终究是 Python 里 `raise` 一个异常，从没真正跨过进程边界）。

**真实进程 kill**：`dist/chaos_worker.py` 是一个可以单独 `python -m rewind.dist.chaos_worker ...` 启动的脚本——它连真实 Postgres（`PgCoordinator`）、文件系统上的 `SnapshotStore(root=...)`、真实的 trace 文件，这三样东西全部活在进程外部，所以另一个完全独立的进程只要指向同样的 DSN/目录就能接着跑。`tests/test_chaos.py` 里的核心测试：`subprocess.Popen` 启动 worker A → 轮询 Postgres 直到 A 的 step-1 checkpoint 真正落库（不是"进程起来了"，是"确认有真实的durable进度"）→ `os.kill(pid, signal.SIGKILL)`（不是调试钩子，是真信号）→ 等 A 的真实 lease 在挂钟时间里过期 → 启动全新的 worker B 进程接完剩下的活。断言：最终状态正确、"取消订单"这个有副作用的 tool 恰好被真实执行了 1 次（计数器写进 `env.state`，这样才能跨进程边界被父进程读到——Python 闭包变量做不到这点）、trace 的 `seq` 连续无缺口、Postgres 里没有卡死的 lease。

**真实网络故障注入（Toxiproxy）**：`dist/toxi.py` 的 `ToxiproxyController` 管理一个真实的 `toxiproxy-server` 子进程，通过它的 HTTP admin API 建代理、挂 toxic。三个测试分别验证：`latency` toxic 真的让一次真实 `asyncpg` 往返变慢（用挂钟时间测出来的，不是断言配置项存在）；`reset_peer` toxic 真的让连接失败，拿掉 toxic 后立刻恢复；`timeout` toxic 对着 Redis 也是一样的效果。这三个测试证明的是"故障注入本身是真的"，而不是"应用代码假装处理了故障"。

**不变量检查器**（`dist/invariants.py`）：`check_seq_contiguous` 读真实 trace 文件断言 `seq` 无空洞无重复；`check_no_stuck_leases` 查真实 Postgres 断言没有 lease 过期超过宽限期还没被回收的 job；`check_final_hash_matches` 比较 chaos 跑和对照跑的最终 env hash。"每个 (run_id, step) 最多一次已提交副作用"这条不变量在这里不是一个通用函数——它需要一个能跨进程边界存活的可观测量，所以走的是 `chaos_worker.py` 里"把计数器写进 `env.state`"这个具体机制，而不是抽象成一个和 env 内部结构无关的通用检查器。

**扩展性实验**（`scripts/chaos_scale_experiment.py`）：固定任务数，worker 并发数 1/2/4/8，全部走真实 `PgCoordinator`（每次 claim/checkpoint/complete 都是真实网络往返），用零延迟的 mock LLM 隔离掉 provider 侧的影响。这个沙箱只有 2 个 vCPU，所以"8 个 worker"测的是 I/O 并发（等 Postgres 网络往返时切换协程）而不是真正的多核并行——诚实地说，这就是为什么在 I/O 瓶颈的场景下这条曲线依然有意义。实测（40 个任务，每个并发度重复 3 次取中位数）：

| worker 数 | 吞吐（job/s） |
|---|---|
| 1 | 64.4 |
| 2 | 90.9 |
| 4 | 116.1 |
| 8 | 102.3 |

4 个 worker 时吞吐最高，8 个反而回落——2 vCPU 上协程调度开销开始超过并发省下的 I/O 等待时间，是符合预期的形状，不是需要"修"的异常。

**恢复时间实验**（`scripts/chaos_recovery_experiment.py`）：重复"真 kill -9 worker A → 等真实 lease 过期 → 启动 worker B"这个场景 20 次，测的是从 lease 真正过期的那一刻到 worker B 完成接管为止的挂钟时间。实测（`lease_seconds=2`）：p50 = 0.316s，p99 = 0.376s，min = 0.265s，max = 0.376s——这个延迟基本就是"启动一个新 Python 进程 + 一次 Postgres claim 往返"的开销，没有意外的长尾。

### W7 剩余部分：失败分类、自动配对、归因准确率（本轮新增）

**失败分类**（`flywheel/failures.py`，方案 4.6 的表）：`classify()` 只读 harness 已经产生的信号——`RunResult.status`/`reason`，以及 trace 里 `tool_call`/`llm_call` 事件的 `error_class`——不另开一条日志通路去维护一致性。4.6 表里的 11 个类别，这里能覆盖 9 个：`LLM_API_ERROR`、`OUTPUT_MALFORMED`、`TOOL_ARG_INVALID`、`HALLUCINATED_TOOL`、`TOOL_RUNTIME_ERROR`（这条比较微妙：它不会中止 run，harness 把它当可恢复错误处理，所以要靠"`completed` 但没 grade 通过、且 trace 里有一个 `TOOL_RUNTIME_ERROR` 的 `tool_call`"这个组合信号去识别，而不是看 status）、`LOOP`、`BUDGET_EXCEEDED`、`WORKER_CRASH`（这个不来自单次 `AgentRunner` trace——它是 checkpoint/coordinator 层的性质，已经在 W4/W5/W6 分别用 sqlite、真实 Postgres、真实 `kill -9` 穷尽验证过，`classify_worker_crash()` 只是给批量报表一个占位构造函数，不是从 trace 信号重新推导）、`REASONING_ERROR`（所有 tool 调用都成功，但最终状态就是不对——这正是 4.7 归因算法要处理的那类）。`CONTEXT_OVERFLOW`（没做 context compaction）和 `POLICY_VIOLATION`（没做 policy guard）**这个 harness 目前没有对应机制**，所以 `classify()` 永远不会产出这两类——如实标注为空，而不是伪造一个永远不会真正触发的代码分支。

`scripts/failure_batch.py` 用故意构造的场景跑出一批覆盖全部 9 个可达类别的失败样本，分类后打印分布和 top clusters：

```
23 failure samples classified.

REASONING_ERROR           4  17.4%
LLM_API_ERROR              3  13.0%
TOOL_ARG_INVALID           3  13.0%
TOOL_RUNTIME_ERROR         3  13.0%
OUTPUT_MALFORMED           2   8.7%
HALLUCINATED_TOOL          2   8.7%
LOOP                       2   8.7%
BUDGET_EXCEEDED            2   8.7%
WORKER_CRASH               2   8.7%
```

**这是合成批次，不是真实生产流量抽样**——还没有真实流量（W1 的 τ-bench adapter 还没写），如实标注而不是包装成"线上失败分布"。

**失败聚类**：方案要求按（失败类别、根因步 tool、错误信息 embedding）聚类，同时说"别过度工程"。这里只做了按 `(category, root_cause_tool)` 分组（`cluster()`/`top_clusters()`），没有再加一层错误文本 embedding 子聚类——这个 toy env 的失败模式种类有限、错误文本本身高度模板化，加一层 embedding 聚类不会比"类别+工具"这两个维度切出更有意义的簇，真的接上 embedding 需要真实场景下自由文本错误信息足够多样、量也足够大才值得做。

**`patch_for_step` 自动配对**（`auto_patch_for_step`）：不再要求人手写每个候选步骤的修正动作（`attribute-demo` CLI 命令里那种手写 `ForkPatch`），而是从同一个任务的一条**已知成功**轨迹里，直接取同一步号那次真实的（非 patched）`llm_call` 响应转成 `ForkPatch`。取不到（成功轨迹在那一步没有真实的 llm_call）就返回 `None`，调用方自己决定退回手写补丁还是让强模型在"已知期望最终状态"提示下生成——这是方案里 4.7 第 1 步写的优先级顺序。

**归因准确率（合成 ground truth）**（`flywheel/eval_attribution.py`，`rewind attribute-eval`）：方案要求"人工标注 30-50 条失败轨迹的根因步，报告归因准确率"——但目前没有真实失败语料可标注（还是 toy env，不是 τ-bench）。所以做的是诚实的替代版本：构造一批"根因步号本身就是已知的"失败轨迹（把根因步依次设在 1、2、3、4、5，其余步骤是无副作用的只读操作），用 `auto_patch_for_step` 从对应的成功参考轨迹里自动取每个候选步骤的修正动作（不是手写的），跑 `bisect_attribute`，和已知的真实根因步比较，报告精确匹配和 ±1 步容忍两种准确率。这是**合成** ground truth，不是人工标注的真实失败，两者不能划等号——但它确实是一个真实的、可复现的准确率数字，而不是编一个数字充数。实测（5 个不同根因位置）：

```
[OK] true_root=1 predicted=1 oracle_calls=1 monotonic=True
[OK] true_root=2 predicted=2 oracle_calls=2 monotonic=True
[OK] true_root=3 predicted=3 oracle_calls=3 monotonic=True
[OK] true_root=4 predicted=4 oracle_calls=3 monotonic=True
[OK] true_root=5 predicted=5 oracle_calls=3 monotonic=True

n=5 exact_match_accuracy=100.0% within_1_step_accuracy=100.0% avg_oracle_calls=2.4
```

5/5 精确命中，平均 2.4 次 oracle 调用（候选数 1~5 时线性扫描要花 1~5 次，二分在候选数变大时优势会更明显）。100% 在这里是"算法在无噪声合成数据上正确"的证明，不是"真实场景下的准确率会是 100%"——真实场景有非单调（多个独立错误）、环境随机性、live API 的样本方差，这些都是 `bisect_attribute` 已经考虑到但这批合成数据不会触发的情况（`monotonic=False` 的场景已经在 `tests/test_attribute.py` 里单独测过）。

**性能消融**（方案 4.7/W7 列的 prompt caching、并行 tool call、模型路由、tool 结果截断、context compaction 阈值）：本轮没有做——这些每一项都需要先在 harness 里实现对应机制（目前完全没有并行 tool call、没有模型路由、没有 context compaction），消融的前提是"这个开关本来就存在"，属于比分类/归因/聚类更大的一块独立工作，如实标注在"下一步"里而不是塞一个走过场的数字。

### W8：数据飞轮 + CI 回归门禁（本轮新增）

**先说清楚"模型"是什么**：这个沙箱没有可用的 LLM API key，没法真的对着一个真实模型做"发现失败 → 改 prompt/工具描述 → 重跑看有没有变好"这个飞轮循环。所以 `HeuristicRetailProvider`（`flywheel/heuristic_provider.py`）是一个**确定性的规则函数**，不是任何模型的近似——它用两个独立、可开关的布尔标志位模拟"两个真实的 prompt/工具改进分别会修好什么"：`bug_ignores_ownership`（读到订单后不检查是不是当前用户的，直接尝试取消）和 `bug_ignores_status`（不检查订单状态是不是 `pending`，对已发货的订单也尝试取消）。这一节报告的是**飞轮机制本身**（切分/门禁/合并/汇报这套流程是真实可跑、可复现的），不是"真实 LLM 经过这套流程后效果提升了多少"的结论——这两件事不能混为一谈。

**任务生成与切分**（`flywheel/tasks.py`）：`generate_suite(n, seed)` 用固定种子生成三类任务——`normal`（正常可取消）、`ownership_trap`（目标订单属于另一个用户 `u2`，工具本身**没有**owner 校验，所以这个 bug 会真的导致取消了不该取消的订单）、`shipped_trap`（目标订单状态已经是非 `pending`，但工具**自己会**拒绝取消）。`split_suite` 按 60/20/20 切成 train/dev/heldout，覆盖每个任务恰好一次，`save_split`/`load_split` 把切分序列化成 JSON 落盘（`data/task_split.json`），保证同一批任务在不同轮次、不同进程间可复现比较，而不是每次都重新随机抽样。

**发现的一个真实的不对称性**（不是设计出来的，是第一次跑通后才注意到的）：两个 bug 看起来对称，实际后果完全不同——`ownership_trap` 的 bug 是**真的正确性 bug**：`cancel_order` 工具本身没有任何 ownership 检查，所以一旦 provider 忽略 ownership 直接调用，订单真的被取消，最终状态就是错的（`test_ownership_bug_actually_produces_a_wrong_final_state`）。而 `shipped_trap` 的 bug **从来不会导致错误的最终状态**，因为 `cancel_order` 工具自己已经会拒绝非 `pending` 订单——provider 忽略状态检查、尝试调用，只会在 trace 里留下一条 `TOOL_RUNTIME_ERROR`，任务最终还是判分正确（`test_status_bug_is_caught_by_the_tools_own_guard_and_does_not_fail_the_task`）。这是一个真实的、值得在简历/面试里讲的发现：**下层工具自己的防御（这里是状态校验）能兜住上层（prompt/模型）的错误，但这种兜底本身是有代价的——它不会拉低 pass rate，却会在 trace 里留下可观测的"差点出错"信号**，如果只盯着 pass rate 这一个指标，这类问题会完全隐身。所以 `SplitRunResult` 除了 `pass_rate` 之外单独统计了 `defensive_saves`（有多少次运行是"只因为下层工具的 guard 兜住了才算过"）。

**两轮飞轮**（`flywheel/data_flywheel.py`）：复用 W7 的 `classify`/`cluster`/`top_clusters`，流程是——① 在 train 上跑当前策略，分类+聚类失败；② 取最大的失败簇，在一张写死的 `CANDIDATE_FIXES` 查找表里找对应的候选修复（`(category, root_cause_tool) -> 该关掉哪个 bug 标志位`，代表"已经知道怎么修"这一步，不是自动生成修复方案，方案本身要求的是"有候选修复方案"而不是"自动发现修复方案"）；③ 在 dev 集上验证候选修复：如果目标指标（多数情况是 pass rate，见下面的不对称性）没有变差就通过门禁、合并（关掉那个 bug 标志位）；④ 报告 heldout 上的结果。真实跑出来的报告（`python scripts/data_flywheel_report.py`，固定 seed=20260928，`data/task_split.json` 里 18/6/6 的切分）：

```
baseline (both bugs present): train pass_rate=66.7% (12/18)
  train failure clusters: [(('REASONING_ERROR', 'cancel_order'), 6)]

round 1 [MERGED]  target_cluster=('REASONING_ERROR', 'cancel_order')  fix=bug_ignores_ownership
  dev pass_rate: 50.0% -> 100.0%
  heldout pass_rate after merge: 100.0%

round 2 [MERGED]  target_cluster=('TOOL_RUNTIME_ERROR', 'cancel_order')  fix=bug_ignores_status
  dev defensive_saves: 2 -> 0
  heldout defensive_saves after merge: 0
  note: pass rate does not move here — the tool's own guard already prevented a wrong final state; this fix is a robustness/observability improvement, not a correctness fix.

final heldout: pass_rate=100.0% (6/6)  defensive_saves=0
bug flags after all rounds: {'bug_ignores_ownership': False, 'bug_ignores_status': False}
```

第一轮的失败簇分类成 `REASONING_ERROR`（不是某个工具报错，是"所有工具调用都成功，但最终状态错了"——这正是 W7 归因算法要处理的那一类），修好 ownership 问题后 dev/heldout pass rate 都到 100%。第二轮换了个目标簇 `TOOL_RUNTIME_ERROR`（这次是工具真的报错，但 run 仍然判分通过），修的是 `defensive_saves` 这个单独的指标而不是 pass rate——这是刻意保留的诚实设计，而不是把两轮都硬套成"pass rate 提升"的整齐叙事。

**CI 回归门禁**（`scripts/ci_pr_gate.sh` / `scripts/ci_nightly.sh` / `.github/workflows/ci.yml`）：PR 门禁是排除 `tests/test_dist_*.py` 三个文件和 `tests/test_chaos.py` 之后的快速集（75 个测试，本地实测 1.19s），不需要任何外部服务；nightly/手动触发跑全量（105 个测试，本地实测 17.45s），需要真实 Postgres/Redis/toxiproxy-server。两个脚本都在这个沙箱里真的跑过并且通过；`.github/workflows/ci.yml` 是一份写得认真、YAML 语法验证过的真实配置，但这个沙箱不是 git 仓库（没有 git 历史，没有 GitHub 连接），所以没有被真实的 GitHub Actions 跑过——这个区别在"已知局限"里如实标注。

### W1 followup：真实 tau-bench 零售域 adapter（本轮新增）

方案里 W1 一直留着"写 `env/taubench_adapter.py`，跑出 baseline"这一条，前面七周因为想先把 harness 本身做扎实、一直用自己写的玩具环境没动它。这一轮把它接上了——不是重新实现一个"像 tau-bench 一样"的环境，是直接拿 tau-bench 自己的真实数据（500 用户、1000 订单、50 商品）、真实工具（16 个，含参数校验和真实的 mutate 逻辑）、真实 reward 约定（基于状态哈希比对的 `r_actions`），接进 `env/taubench_adapter.py` 里。

**接口怎么对上的**：rewind 的 `AgentRunner`/`ToolRegistry` 需要的是 `snapshot()`/`restore()`/`applied`/`db_hash()` 四个方法和一个 pydantic `args_model`；tau-bench 的工具暴露的是 OpenAI function-calling 格式的 JSON schema 和一个 `invoke(data, **kwargs)` 静态方法，返回值是字符串（成功是 JSON 字符串，失败是 `"Error: ..."` 开头的字符串，不是异常）。`_schema_to_args_model` 用 `pydantic.create_model` 从 tau-bench 自己发布的 JSON schema 动态建 pydantic 模型（不是我手写 16 份参数定义去维护一致性），`_wrap_tool_fn` 把 tau-bench 的"错误字符串"约定翻译成 rewind 的 `ToolError` 异常约定，中间没有重新实现任何工具的业务逻辑——`tool_cls.invoke(data=env.data, **kwargs)` 调的就是 tau-bench 自己仓库里那份代码。

**诚实说清楚"baseline"是什么、不是什么**：tau-bench 的任务是多轮对话——agent 和一个真实存在的用户模拟器（本身也是一个 LLM,由 `litellm` 驱动）来回交互，直到用户说完。这个沙箱没有任何 LLM API key(不管是给 agent 用的还是给用户模拟器用的),所以**真实模型 baseline(一个模型真的去尝试解决这些任务)在这里做不出来——这不是没时间做,是这个环境里没有能调用的模型**。

能做、而且诚实地做了的是**黄金轨迹回放校验**：每个 tau-bench 任务自带一份标注好的"正确答案"(`task.actions`,已验证能让 reward=1 的真实工具调用序列)。把这份答案序列转成一个 `ScriptedProvider` 脚本,跑过 rewind 的真实 harness(真实走 loop.py、真实走 ToolRegistry、真实调用 tau-bench 的工具改真实的 `data`),再用 tau-bench 自己的哈希约定(`consistent_hash(to_hashable(data))`,直接复用 `tau_bench.envs.base` 里的函数,不是照着重新写一份)grade。这验证的是"adapter 接得对不对"(真实工具、真实数据、真实 reward 语义是否被正确调用和比较),不是"模型解决任务的能力"——答案本身就是题目的一部分,正确接线的 adapter 理应接近 100%,这里的角色类似 `tests/test_checkpoint.py` 对 checkpoint 协议的作用:先证明管道本身是对的,再谈模型效果。

真实跑出的结果(`python scripts/taubench_baseline.py --split all`):

```
[train] 500 tasks, 514.5s (0 skipped: needs dialogue layer; see unsupported_reason)
  golden-trajectory replay: 500/500 = 100.0%

[dev] 20 tasks, 19.5s (0 skipped: needs dialogue layer; see unsupported_reason)
  golden-trajectory replay: 20/20 = 100.0%

[test] 115 tasks, 114.5s (37 skipped: needs dialogue layer; see unsupported_reason)
  golden-trajectory replay: 78/78 = 100.0%
```

**为什么 test split 有 37 个任务被跳过、而不是硬塞进同一个通过率里**:tau-bench 的 reward 除了"状态哈希对不对"(`r_actions`)之外,test split 里 37/115 个任务还要求"agent 说过的话里包含某些内容"(`r_outputs`,检查的是 agent 在多轮对话中真实说出的文本)——这需要真实走一遍对话,而这个 adapter 没有(也不能,没有 LLM)模拟对话内容,只模拟了"工具调用序列"这一半。把这 37 个任务也算进分母、报告"100%"是会误导人的数字(意味着我们验证了比实际验证的更多东西),所以 `unsupported_reason()` 显式识别出这些任务、诚实跳过、单独报数,而不是悄悄混进总通过率。train/dev 两个 split 全都没有 `outputs` 要求,所以它们的 100% 是真正意义完整的(对这个 adapter 能验证的那部分而言)。

**负面对照**:为了证明"100%"不是因为 grading 函数本身写错了、永远返回 True,专门测了一个反例——故意从黄金轨迹里去掉最后一个会修改状态的工具调用,跑完之后 `replay_grade` 正确返回 `False`(`tests/test_taubench_adapter.py::test_dropping_a_mutating_action_fails_grading`)。

**这一步之后,真正的下一步是什么**:接一个真实 LLM API key 之后,① 给 agent 接 `AnthropicProvider`(已经有了,W5 写的);② 给 tau-bench 的用户模拟器接同一个或另一个真实模型(这个 adapter 完全没碰这部分,`tau_bench.envs.user.LLMUserSimulationEnv` 是现成的);③ 跑真实多轮对话,这时候才有意义去报"pass^1""pass^k"这些方案里提到的真实模型指标——这也是之前聊简历条目时提到的"pass^k 在确定性 provider 下没有意义"这个问题的真正解法,不是靠这个沙箱能解决的,需要真实部署环境。

### 真实模型接入：DeepSeek（本轮新增）

接了一个真实、会花钱调用的模型，用来做两件不同的事——分开说清楚，别混成一件事。

**① 真实 tau-bench baseline（模型真的去解决任务，含多轮对话）**：用 tau-bench 自己的 `run.py`，不改 rewind 任何代码。tau-bench 的 `ToolCallingAgent`（`tau_bench/agents/tool_calling_agent.py`）本来就是通过 `litellm` 调用模型的，`litellm` 原生支持 `deepseek` 这个 provider——所以跑真实 baseline 只需要两个环境变量和几个 CLI 参数，不需要我们这边写一行新代码：

```bash
export DEEPSEEK_API_KEY=sk-...
bash scripts/run_taubench_deepseek.sh                     # dev split，20 个任务
bash scripts/run_taubench_deepseek.sh --task-split test   # test split，115 个任务（37 个带 outputs 要求）
```

这条路径为什么不该用 rewind 自己的 `AgentRunner` 重新实现一遍：tau-bench 的任务需要一个真实 LLM 驱动的"用户模拟器"在多轮对话里来回说话，agent 的"respond"动作要等用户模拟器回应才能判断对话是否结束——这正是 `env/taubench_adapter.py` 模块文档里说的"黄金轨迹回放校验能做、但真实多轮对话做不到"的那个结构性缺口，换哪个模型都绕不开。tau-bench 自己的 `run.py` 已经把这套协议实现对了（而且是 Sierra 官方维护、别人发论文也是用这同一份代码跑出来的数字），在 rewind 里重新搭一遍是重复劳动，还多一道出 bug 的风险，产出的分数反而不如直接用官方实现可信。

**② 让 rewind 自己的 harness 接上真实模型**：`providers.py` 新增了 `DeepSeekProvider`（和已有的 `AnthropicProvider`并列，DeepSeek 的 API 是 OpenAI 兼容的，所以复用 `openai` 这个包、换个 `base_url` 就行，不用额外写 HTTP 客户端）。这让 `rewind run`/`rewind fork` 这些命令能在 toy 环境上用真实模型跑，不再只能喂 `ScriptedProvider`：

```bash
pip install -e ".[deepseek]"
export DEEPSEEK_API_KEY=sk-...
rewind run --task "I'm u1, cancel my headphones order" --provider deepseek
rewind fork runs/run-xxxx/trace.jsonl --at 2 --provider deepseek \
  --patch '{"tool_calls":[{"name":"get_order","args":{"order_id":"O1"}}]}'
```

这条路径**不能**用来跑真实 tau-bench 任务——toy 环境的任务是单轮（一句话任务描述，agent 自主跑到底），tau-bench 的任务需要前面说的多轮用户模拟器循环，`AgentRunner` 没有"接受运行中途新用户消息"这个机制。②验证的是"rewind 自己的 harness 真的能接真模型"，①才是"tau-bench 真实分数是多少"——两件事，别混着报。

消息/工具格式转换（`to_openai_messages`/`to_openai_tools`，没有网络调用，纯函数）有单测覆盖（`tests/test_openai_convert.py`）；`DeepSeekProvider` 本身需要真实 key，和 `AnthropicProvider` 一样不在离线测试范围内。

**真实跑出的结果**（①路径，`deepseek-chat` 同时做 agent 和用户模拟器，test split 全部 115 个任务，1 trial）：

```
Average reward: 0.7043478260869566
Pass^k
  k=1: 0.7043478260869566
```

即 81/115 任务通过（`0.0` 温度、`tool-calling` agent 策略）。这是这个项目里**第一个、也是目前唯一一个真实模型跑出来的数字**——前面 W1-W8 所有其它"100%"（归因准确率、黄金轨迹回放、两轮飞轮）都明确标注过是合成/黄金轨迹/确定性启发式，不是模型表现；这个 70.4% 才是"一个真实模型在真实 tau-bench 任务上实际能做到什么程度"的答案，数值明显低于前面那些 100%，这恰好印证了之前反复强调的那个区别：**adapter/算法正确性的验证分数和模型能力分数是两件不同的事，不能互相替代**。

只跑了 1 个 trial，所以这里的 `pass^k`（k=1）其实就是 `pass^1`，还没有真正意义上的 k>1 数字——tau-bench 的 `pass^k` 要求同一个任务跑 k 次、k 次全部成功才算；真正用它衡量模型稳定性（而不是单次运气）需要 `--num-trials 3`（或更多）重跑，代价是 API 调用量乘 k 倍。

### 真实失败归因：`flywheel/taubench_failures.py`（本轮新增）

上面那 34 个真实失败任务（115-81），喂给了 `flywheel/` 原有的失败分类逻辑——但 `flywheel/failures.py::classify()` 读的是**这个仓库自己**`AgentRunner` 的 trace 格式（`Event`/`RunResult.status`），tau-bench 的 `run.py` 产出的是另一套格式（`EnvRunResult`：`task_id`/`reward`/`info`/`traj`/`trial`，定义在 `tau_bench/types.py`），两边没有共享的 trace schema，硬套会直接读错字段。所以新写了一个同精神、不同输入格式的对应模块 `flywheel/taubench_failures.py`，直接对着真实结果文件的结构写（不是先设计分类再凑数据）：

```bash
python scripts/taubench_failure_report.py path/to/your/results.json
```

用你贴的那份真实结果文件（test split，115 任务，1 trial）跑出来的真实分布：

```
115 tasks total -- 81 passed, 34 failed (pass^1 = 0.7043)

Failure category distribution:
  WRONG_ACTION                              18  (52.9% of failures)
  TOOL_RUNTIME_ERROR                        11  (32.4% of failures)
  ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT       4  (11.8% of failures)
  WRONG_OUTPUT                               1  (2.9% of failures)

Of the 22 action-mismatch failures, by shape:
   12  same tool-name sequence -- mismatch is in the arguments, not which tools were called
    3  unclear from tool names alone
    3  took an extra, unrequired mutating action
    3  never attempted a mutating action
    1  missing at least one required mutating action

Of the 11 TOOL_RUNTIME_ERROR failures, distinct error messages:
    6  Error: user not found
    3  Error: non-pending order cannot be modified
    1  Error: non-delivered order cannot be exchanged
    1  Error: order not found
```

这是这个项目里**第一份基于真实模型真实失败样本的归因结果**，和 W7 节里"合成 ground truth、exact-match=100%"那份归因准确率评测是两件不同的事——那份验证的是"归因算法本身对不对"（答案已知，算法要找到它），这份是"真实模型到底怎么失败的"（没有预设答案，纯粹从真实 trace 里读信号）。几个可以直接拿来讲的真实发现：

- **"动作对不上"（WRONG_ACTION + ACTION_MISMATCH，22/34，64.7%）是最大头，但其中 12 个工具名序列完全正确**——模型选对了该调的工具、调用顺序也对，错在参数（具体是传错了订单号/用户 id/金额之类，仅看工具名看不出来，需要真正 replay 参数才能细分，这个局限写在模块文档里了）。这说明 DeepSeek 在"该做什么"这个决策层面大多是对的，败在"填对参数"这个更底层的执行细节上。
- **4 个任务说对了该说的话、但状态仍判错**（ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT）——模型告诉用户"已经处理好了"，但实际落地的数据库状态和 ground truth 不一致，是"嘴上说完成了，动作没完全做对"的真实案例，这种类型如果只看最终 `reward` 或者只看回复文本，两种看法都会得出错误的结论（回复文本看起来是对的）。
- **TOOL_RUNTIME_ERROR 占 32.4%，表面看 6/11 是同一种 "Error: user not found"——但逐条读完整对话后，这个第一印象是错的**：把这个第一印象留在这里，是因为"按错误消息字符串聚类"本身是一个真实的陷阱，值得记录，不是因为它是对的。真正展开每条对话看：6 个 "user not found" 里，有 4 个（34/54/76 等）是用户模拟器自己把邮箱报错了，agent 立刻改用姓名+邮编重新查到账户、继续往下走——错误被自愈了，这几个任务最终判 0 分的真正原因在后面别的地方，不是身份校验；只有 2 个（71/107）是身份校验真的卡死没解决、任务提前放弃。见下方"真实失败归因 round 1"——真正干净、可复现、可归因到单一文档段落的簇，是另一个错误消息："Error: non-pending order cannot be modified"（3 个：72/98/105）。

单测（`tests/test_taubench_failures.py`，10 个，构造的小样本覆盖每个类别，不依赖真实文件）验证的是分类逻辑本身——归因准确率意义上的"对不对"要靠真实文件的分布是否和人工抽查一致来判断，这里没有再做一轮人工标注交叉验证（样本量 34，抽查了前 10 条，分类和 trace 内容对得上，但没有做完整的人工对照）。

### 真实失败归因 round 1：一个具体 bug + 一个真实 patch（本轮新增）

上面"user not found"那个教训就是起点：对 11 个 TOOL_RUNTIME_ERROR 任务逐条读完整对话（不只看错误消息字符串），发现 72/98/105 这三个任务共享完全相同的结构——

1. agent 对同一个订单先调用 `modify_pending_order_items`（改商品）——成功
2. 紧接着对**同一个订单**调用 `modify_pending_order_address`（改地址）——失败，报 `Error: non-pending order cannot be modified`

tau-bench 自己的 retail policy（`tau_bench/envs/retail/wiki.md` 第 57 行）其实写了这条规则："`modify_items` 只能调用一次，调用后订单状态变成 `pending (items modifed)`，之后不能再修改或取消这个订单"——但这句话只出现在"Modify items"小节末尾，不是放在"同一个订单要同时改地址又改商品该先做哪个"这个决策点上。DeepSeek 知道两件事都要做，但没把这条规则当成"顺序约束"来遵守，顺序一反就必然失败。这是一个具体、真实、可归因到单一文档段落的策略违反，不是笼统的"模型有时候会错"。

**真实 patch**（`scripts/taubench_round1_wiki.patch`，针对你本地 `../tau-bench` 克隆的 `tau_bench/envs/retail/wiki.md`，已经用 `patch -p1 --dry-run` 验证过能干净应用）：在"Modify pending order"公共说明处加一条明确的顺序指令——"如果同一个订单需要同时改地址/支付方式和改商品，必须先做地址/支付方式的修改，确认成功后才能改商品，因为改商品会锁死订单"。

**重跑方式**（`scripts/run_taubench_round1.sh`）：自动打上面这个 patch（已打过会跳过，幂等），然后用**和 baseline 完全相同的协议**（`--env retail --task-split test --model deepseek-chat --model-provider deepseek --user-model deepseek-chat --user-model-provider deepseek --agent-strategy tool-calling`）重跑全部 115 个任务，输出到 `results_round1/`（不覆盖 baseline 的真实结果文件）：

```bash
export DEEPSEEK_API_KEY=sk-...
bash scripts/run_taubench_round1.sh
```

**为什么重跑全部 115 个，不是只跑那 3 个**：`temperature=0` 虽然大概率其它任务的轨迹不变，但要得到一个和 baseline 严格可比的 `pass^1`，必须看这一行改动有没有意外影响到其它任务（哪怕只改了 wiki.md 里一个小节，其它任务的 system prompt 整体也变长了，不能排除连带影响），不能只挑我想让它变好的 3 个任务。

**对比方式**（`scripts/taubench_round1_compare.py`）：

```bash
python scripts/taubench_round1_compare.py path/to/baseline.json results_round1/<新文件>.json
```

报告 round 1 相对 baseline 的 `pass^1`、哪些任务被修好、哪些任务被改坏（回归，和修好的同等重要）、72/98/105 这三个目标任务是否真的翻过来了。**这一步需要你在本地用自己的 `DEEPSEEK_API_KEY` 实际重跑**——我这边没有 key，也验证不了真实的 pass^1 会变成多少；patch 本身已经做了 dry-run 验证能干净应用，脚本逻辑也测过（`taubench_round1_compare.py` 拿 baseline 文件和自己对比过，delta 正确算出 0），但"patch 真的让 pass^1 提升、而不是打平或者还倒退"这件事，到你本地真实跑出第二个结果文件之前，是未知数，不能假装已经知道答案。

**真实重跑结果**：

```
baseline  pass^1 = 81/115 = 0.7043
round 1   pass^1 = 83/115 = 0.7217
delta: +2 tasks

Fixed by the patch (15): [13, 19, 39, 47, 66, 67, 71, 72, 86, 88, 92, 96, 98, 102, 107]
Regressed by the patch (13): [3, 5, 7, 27, 38, 44, 52, 62, 64, 70, 85, 101, 106]
Still failing in both (19): [...]

Of the 3 tasks the patch specifically targeted (72, 98, 105): 2/3 fixed ([72, 98])
```

**净效果是真的、但很小（+2），而且不是"干净地修好了 3 个目标任务"这么简单**——目标的 3 个任务里确实修好了 2 个（72、98；105 还是没过，这一个之后可以单独翻一下它的新 trace 看看是不是还卡在别的地方），但同时还有 **15 个任务翻对、13 个任务翻错，完全不在目标范围内**，净值只是两者相减出来的 +2。这个"大量双向抖动、净值很小"的现象，比"净值是正的"这件事本身更值得记录，原因有两个候选，这里**都没有验证哪个是真正原因**，如实列出来而不是挑一个讲得通的就当结论：

1. **真实的连锁反应**：往 system prompt 里加一段话，哪怕是针对一个具体 bug 写的，也会轻微改变每一个任务的 prompt 长度/内容——而对话是多轮的，agent 和用户模拟器（同样是 `deepseek-chat`）互相根据上一轮的输出生成下一轮，哪怕第一个 token 的概率只是轻微变化，经过几十轮对话也可能被放大成完全不同的轨迹。这是真实存在的机制，不是臆测。
2. **单次运行本身的噪声**：tau-bench 的 `run.py` 调用 `litellm.completion`，`temperature=0` 不保证两次调用逐 token 完全一致（这是很多 LLM API 的公开行为，不是 tau-bench 或这个项目的问题）——如果把**完全相同的 baseline（不打任何 patch）跑两遍**，可能本身就会看到类似量级的双向抖动，那样的话 70.4%→72.2% 这个差距就很可能落在噪声范围内，不能当成"patch 有效"的证据。

**诚实的结论**：在验证"baseline 自己跑两遍、不打 patch，抖动量级是多少"这个噪声基线之前，不能宣称这个 patch 让 DeepSeek 在这个策略上变得更可靠——现在能确凿说的只是"挖出的那个具体 bug，确实在它所描述的场景下复现了（72/98 修好证实了因果关系的存在），但系统层面的净效果目前统计上还站不住"。这正是前面"下一步"里提到的 `--num-trials` 问题的另一个侧面：单个 trial 的 pass^1 本身就不是一个稳定的点估计，拿它做干预前后对比之前，需要先知道噪声有多大。

### 噪声基线：round 1 的净效果有没有可能全是噪声（本轮新增）

要知道上面 +2（70.4%→72.2%）里有多少是 patch 真的起效、多少是两次 API 调用本身就会有的抖动，唯一办法是跑一组对照：**完全不碰 patch**，把 baseline 的协议原样再跑一遍，看"什么都没改"的两次运行之间，pass^1 和任务级翻转量级有多大。

```bash
export DEEPSEEK_API_KEY=sk-...
bash scripts/run_taubench_noise_floor.sh
```

这个脚本会先检查你本地 `../tau-bench` 的 `wiki.md` 是不是还处于 round 1 打过 patch 的状态（跑完 `run_taubench_round1.sh` 之后它会是），如果是就用同一个 patch 文件反向应用（`patch -R -p1`）还原成原始版本——已经验证过这个反向应用能把文件还原到和原始版本逐字节相同，不是"大概还原了"。还原之后用和 baseline 完全相同的协议重跑，输出到 `results_noise_floor/`，不会覆盖 baseline 或 round 1 的结果文件。

跑完之后，拿**原始 baseline 结果文件**（不是 round 1 的）去对比：

```bash
python scripts/taubench_compare.py 原始baseline.json results_noise_floor/新文件.json \
  --label-a baseline --label-b "baseline rerun（noise floor）"
```

（`scripts/taubench_round1_compare.py` 本身现在是 `scripts/taubench_compare.py` 这个通用对比工具的一个预设薄封装——round 1 对比和这里的噪声基线对比，底层对比逻辑完全一样，只是没有预设"目标任务"那一栏；这是为了让两次对比用的是同一份、已经验证过逻辑正确的代码，不是分别手写两遍类似的脚本。）

看这个对照的结果时，关键不是"pass^1 是不是也变了"，而是**"两次都不打 patch 的情况下，任务级翻转量（总 churn）有多大"**——如果噪声基线本身的翻转量就有十几个（和 round 1 的 15+13=28 个翻转量同一个量级），那 round 1 那个 +2 的净值基本可以认定主要是噪声，这个具体 bug 的修复效果只能靠"72/98 这两个单独案例的因果关系"去佐证，不能靠系统级 pass^1 的变化去佐证；如果噪声基线的翻转量明显小很多（比如个位数），那 round 1 的 28 个翻转就说明这个 prompt 改动确实引入了比预期大得多的连锁反应，这本身也是一个值得记录的真实发现。**这一步同样需要你本地重跑才能知道答案，这里不猜结果。**

**真实重跑结果**：

```
baseline                       pass^1 = 81/115 = 0.7043
baseline rerun（noise floor）   pass^1 = 90/115 = 0.7826
delta: +9 tasks

Flipped to PASS (13): [13, 19, 20, 21, 22, 47, 66, 67, 71, 92, 96, 98, 107]
Flipped to FAIL (4): [27, 38, 74, 82]
Still failing in both (21): [...]

Total churn: 17 tasks (14.8% of 115) for a net delta of +9.
```

**结论很清楚，而且比预期更直接：噪声基线本身的净效应（+9）比 round 1 那个 patch 的净效应（+2）还大**——什么都没改、只是把完全相同的协议重新跑了一遍，pass^1 就从 70.4% 跳到 78.3%。这意味着：

- **round 1 那个 +2 完全不能归功于 wiki.md 的 patch**——两次"什么都没改"的运行之间，净效应是它的 4.5 倍。单看 pass^1 这一个数字，这次重跑比 round 1 那次改进"更多"，但两次唯一的区别就是"又调了一次 API"，没有任何真实的系统改动。
- **这不代表挖出的那个具体 bug 是假的**——72、98 这两个任务里，"先改商品锁死订单、再改地址必然失败"这条因果链是从真实 trace 里逐行读出来的，不是猜的，这个机制本身是真的（而且在噪声基线这次重跑里，72 仍然在"两次都失败"里、98 却翻过去了——连目标任务自己的结果在两次运行里都不稳定，进一步印证了"单 trial 看单个任务的通过/失败"本身就不可靠）。但"这个 patch 让系统整体变得更可靠"这个说法，在只有 1 个 trial 的前提下是**无法用 pass^1 的变化来支持的**——噪声本身就比它大。
- **真正的教训**：在 `temperature=0`、`num_trials=1` 的协议下，tau-bench + DeepSeek 的 pass^1 在同一个任务集上重跑一次就能有 ±(个位数到十几个百分点) 的摆动，14.8% 的任务级翻转率。这比这个项目里任何一次"干预前后对比"想要检测的效果都大，意味着**在这个协议下做任何单 trial 的 before/after 对比，结论都不可信**——不管是这次的 wiki.md patch，还是以后任何类似的 prompt/工具调整实验。要得出有统计意义的结论，必须对 baseline 和干预后都跑多个 trial（`--num-trials 3` 或更多），比较分布而不是比较两个单点数字。

这个结论本身——"我们本来想验证一个具体修复有没有效，结果先验证出'我们的评测方法论本身检测不出这个量级的效果'"——比"patch 有效"这个结论更有分量：它说明了为什么前面反复强调的 `pass^k`（k>1）不是一个可以随便跳过的细节,而是做这类实验的前提条件，不是锦上添花。

### 多 trial 对照：比较分布，不是比较两个单点数字（本轮新增）

噪声基线证明了单 trial 的 pass^1 不是一个能直接拿来做 before/after 对比的数字。真正该做的是：**对 unpatched（baseline）和 patched（round 1）各跑若干个 trial，比较两组分布的差异，而不是各跑一次就比大小**。tau-bench 自己的 `run.py` 原生支持这件事（`--num-trials N`，一个文件里每个任务跑 N 次,每条记录带一个 `trial` 字段，不需要分 N 次跑分别存档）。

```bash
export DEEPSEEK_API_KEY=sk-...
bash scripts/run_taubench_multitrial.sh unpatched 3   # baseline，3 trial，自动确保 wiki.md 没打 patch
bash scripts/run_taubench_multitrial.sh patched 3     # round 1，3 trial，自动确保 wiki.md 打了 patch
```

**成本提醒**：`--num-trials 3` 是单次运行 API 调用量的 3 倍，两个配置各跑一次相当于原始 baseline 那次运行的 6 倍调用量——用 DeepSeek 的价格仍然便宜，但跑之前知道这个量级比跑完才发现好。

跑完之后用 `scripts/taubench_multitrial_compare.py` 对比（单测见 `tests/test_taubench_multitrial_compare.py`，6 个，覆盖分组统计和 bootstrap 区间在"没有真实差异"/"有真实差异"两种构造场景下的行为）：

```bash
python scripts/taubench_multitrial_compare.py \
  path/to/unpatched_multitrial.json path/to/patched_multitrial.json \
  --label-a unpatched --label-b patched
```

这个脚本和 `taubench_compare.py`（单 trial 版）的区别,不只是"输入文件多了几条记录"，是真的换了一套统计方法：

- **pooled pass^1**：把每个任务的所有 trial 结果池化在一起算平均通过率（比如 3 trial × 115 任务 = 345 个样本），样本量比单 trial 的 115 个大 3 倍，点估计本身更稳定。
- **pass^k（真正意义上的）**：tau-bench 原本的定义——一个任务的 k 个 trial **全部**成功才算这个任务通过。之前 README 里所有写"k=1"的地方,其实都只是 `pass^1` 的另一种写法,因为只跑了 1 个 trial；这里才是第一次能算出真正的 `pass^k`（k=3）。
- **配对 bootstrap 置信区间**：不是简单算"patched 的 pass^1 减 unpatched 的 pass^1"，而是按任务配对（同一个任务在两边各有 3 个 trial，先各自算这个任务的平均分，再算这个任务上 patched 比 unpatched 高了多少），然后对"任务"这个层面做有放回重采样（5000 次，每次重新算一遍"任务间平均差异"），取重采样结果的 2.5%～97.5% 分位数作为 95% 置信区间。**如果这个区间不包含 0，说明这个差异大概率不是纯噪声；如果包含 0，诚实的结论是"现在的 trial 数不够看出有没有真实差异"，不是"没有差异"**——这个区分本身很重要，`taubench_multitrial_compare.py` 的输出会把这两种情况分别打印出来，不会把"证据不足"偷换成"效果不存在"。

**这一步还是需要你本地真实重跑**——两次 3-trial 的真实运行，我这边没有 key 也跑不了。跑完把两份文件（或者直接把 compare 脚本的输出）发我，就能知道 round 1 那个具体修复，在排除了噪声之后，到底有没有真实效果。

**真实重跑结果**（两边都是完整的 115 任务 × 3 trial = 345 个样本）：

```
unpatched    pooled pass^1 = 0.7217  (n=345)   pass^k(k=3) = 0.5739
patched      pooled pass^1 = 0.7536  (n=345)   pass^k(k=3) = 0.5565

Mean per-task delta (patched - unpatched): +0.0319
95% paired bootstrap CI (5000 resamples): [-0.0203, +0.0870]
CI includes 0.
```

**最终、诚实的结论：即使认真跑了 3 个 trial、用了配对 bootstrap 这种更严谨的方法，仍然不能说这个 wiki.md patch 有统计意义上的效果**——置信区间横跨 0（-2.0% 到 +8.7%），"+3.19 个百分点"这个方向看起来是正的,但幅度还在噪声范围内，不能排除真实效果是 0 甚至是负的。更有意思的是：衡量"稳定性"的 `pass^k`（3 次全过才算）反而是 patched 更低（55.65% vs 57.39%）——"单次平均通过率提高了一点"和"模型变得更稳定"在这组数据里甚至是反着走的,这进一步说明不能简单用一个指标替代另一个。

**这条线到这里正式收尾**：从"挖出一个具体、真实、可复现的 bug（72/98 两个任务的因果关系是从真实 trace 逐行读出来的,不是猜的）→ 设计针对性的最小修复 → 单 trial 对比看起来有效（+2）→ 怀疑是噪声,补做噪声基线对照（结果噪声本身更大,+9）→ 认真跑 3 trial + 配对 bootstrap 统计检验 → 最终诚实结论是'证据不足,不能下结论'"——这整条路径，比任何一个单独的百分比数字都更能说明"系统性地评估一个 LLM agent 改动有没有效"该怎么做，以及为什么大多数只跑一次就下结论的"改进"是不可信的。

### 并行 tool call（本轮新增）

"下一步"里一直挂着的性能消融清单（prompt caching、并行 tool call、模型路由、tool 结果截断、context compaction）全部没做的理由是"机制本身在 harness 里还不存在"——这一轮把其中一项补上了：一个 tool-calling LLM 在**同一个 assistant turn**里可以一次发出多个 tool call，这些调用是模型在没看到任何一个结果之前就全部生成的，所以彼此之间不可能有依赖关系；但 `loop.py` 原来的 `_loop` 不管这件事，永远一个一个顺序执行，哪怕其中好几个是互相独立的只读查询。

**只并发非 mutating 的调用，mutating 的永远顺序执行**：`AgentRunner._run_tool_batch`（`loop.py`）把一个 turn 里的 tool call 按 `registry.is_mutating(name)` 分成两组——

- **非 mutating、且这一步有 2 个以上**：用 `asyncio.gather` 一次性调度，每个调用真正慢的部分（`registry.execute`）通过 `asyncio.to_thread` 丢进线程池，让它们在真实的 wall-clock 时间上重叠；批量结束后只落**一次** checkpoint（而不是每个调用落一次）——因为非 mutating 调用不碰 `env.applied` 这个幂等台账，崩溃后重跑任意一个都是免费的，不需要逐个保护。
- **mutating 的调用**：永远按原始相对顺序逐个顺序执行，每个仍然保留自己独立的 checkpoint，和这个功能加之前完全一样——因为后一个 mutation 可能依赖前一个对 `env` 造成的效果（真实例子就是 `scripts/taubench_round1_wiki.patch` 修的那个 bug：先改商品会锁单，导致之后改地址必然失败,顺序错了结果就错）,这里不能冒险假设"同一个 turn 里的 mutation 互相独立"。
- **这一步只有 0 或 1 个非 mutating 调用时**（包括目前所有已有的 demo/测试场景——它们都是一步一个 tool call），走的是和加这个功能之前**完全相同**的单个顺序路径，连 `debug_hook` 的名字（`after_tool_call`）和落 checkpoint 的时机都不变——这就是为什么 `tests/test_checkpoint.py` 那 20 个穷举崩溃点的测试一个没改就全部照常通过：新代码路径只在"同一步里出现 2 个以上非 mutating 调用"这个之前从没被真实触发过的情况下才会跑。

**测试**（`tests/test_parallel_tools.py`，5 个，纯离线，`ToyRetailEnv`+`ScriptedProvider`，不需要真实 API key）：故意让 `get_order`/`list_orders` 在工具函数里真的 `time.sleep`（不是注入的假 clock——假 clock 只能证明"harness 自己以为时间流逝了"，证明不了两件事真的在 wall-clock 上重叠），断言 3 个这样的只读调用放在同一个 turn 里，总耗时远小于三者串行之和；另外专门验证了结果消息即使并发执行完成顺序不确定，也严格按模型原始请求的顺序放回 `messages` 列表；以及用两个互相依赖的自定义 mutating 工具（`step_b` 读 `step_a` 设的标志位）验证 mutating 批次确实还是严格顺序执行、不会被误并发。

**真实测出的数字**（5 个 `get_order` 调用放在同一个 turn 里，每个工具函数真实 `sleep(0.2s)`）：

```
串行理应耗时：5 × 0.2s = 1.00s
实测耗时：    0.209s
```

和 5 个顺序执行基本等于 1 个的延迟，符合"并发只读"的预期。

**诚实的局限**：

- 线程池大小没有显式上限——目前每一步最多出现的 tool call 数量很小（个位数），没有哪个真实任务暴露出"一个 turn 里几十个并发读"的场景，所以没有加显式的并发度上限（如 `asyncio.Semaphore`）；如果未来某个场景一个 turn 里扇出很大，需要补这个限制，否则可能打满默认线程池。
- 并发之后，同一批非 mutating 调用对应的 `tool_call`/`env_snapshot` trace 事件之间的**相对顺序**变成了依赖真实完成时间的（哪个线程先跑完，它的事件就先落到 trace 里）——`seq` 本身仍然严格递增、没有空洞（`TraceWriter.emit` 调用永远发生在主事件循环线程上，不会真的并发写），但同一个并发批次里几条 `tool_call` 事件谁先谁后，live 跑两次可能不一样。这是用延迟换来的一个真实代价，不是 bug：消息列表（`messages`）本身的顺序依然严格按模型原始请求顺序摆放（测试里专门验证了这点），受影响的只是 trace 里内部记录的事件先后，不影响最终对话内容或 replay 的正确性。
- 目前只在这个项目自己的 toy env 上验证；`env/taubench_adapter.py`（真实 tau-bench 工具）没有专门测过并发只读场景，理论上应该直接适用（因为并发逻辑在 `loop.py` 这一层，不关心具体 env 实现），但没有实跑验证过。

## 结果汇总表

| 周 | 实验 | 结果 |
|---|---|---|
| W4/W5 | checkpoint 崩溃恢复（7 个钩子点各精确崩溃一次） | 最终状态全部判分正确，`cancel_order` 真实调用计数恰好 1 次，trace `seq` 全部连续 |
| W5 | Postgres 并发 claim（8 个并发请求打同一连接池） | 8 个任务被 8 个不同 worker 各拿到一个，0 次重复 |
| W5 | Redis single-flight（6 个完全相同的并发请求） | 真实 provider 只被调用 1 次 |
| W6 | 扩展性实验（40 任务，worker 数 1/2/4/8，真实 Postgres） | 吞吐 64.4 → 90.9 → 116.1 → 102.3 job/s（4 worker 峰值，8 个在 2 vCPU 上回落，符合预期） |
| W6 | 恢复时间实验（真 kill -9，20 次重复，lease=2s） | p50=0.316s，p99=0.376s，min=0.265s，max=0.376s |
| W7 | 失败分类批量报告（23 个合成样本，9 个可达类别） | 分布见上文"W7"一节；REASONING_ERROR 占比最高（17.4%） |
| W7 | 归因准确率（合成 ground truth，5 个不同根因位置） | exact_match=100%，within_1_step=100%，avg_oracle_calls=2.4 |
| W8 | 两轮数据飞轮（30 任务，18/6/6 切分，确定性启发式 provider） | baseline train pass_rate 66.7% → 第 1 轮修 ownership bug，dev/heldout pass_rate 100% → 第 2 轮修 status bug，dev/heldout defensive_saves 2→0（pass rate 本轮不变，见上文不对称性说明） |
| W8 | CI 回归门禁（本地脚本，非真实 GitHub Actions 运行） | PR 快速集 75 个测试 1.19s；nightly 全量集 105 个测试 17.45s |
| W1 followup | 真实 tau-bench 零售域黄金轨迹回放（train/dev/test） | train 500/500=100%（514.5s）、dev 20/20=100%（19.5s）、test 78/78=100%（37 个 outputs 任务诚实跳过，114.5s）——不是模型 baseline，见上文说明 |
| W1 followup | **真实模型 baseline**：`deepseek-chat`（agent + 用户模拟器），test split 115 任务，1 trial | pass^1 = 70.4%（81/115）——这个项目里唯一真实模型跑出的数字，和上面黄金轨迹回放的 100% 不是同一件事 |
| W1 followup | **真实失败归因**：对上面 34 个真实失败任务跑 `flywheel/taubench_failures.py` | WRONG_ACTION 52.9%、TOOL_RUNTIME_ERROR 32.4%、ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT 11.8%、WRONG_OUTPUT 2.9%；22 个动作不对的里 12 个工具名序列全对，错在参数；挖出一个具体可复现 bug（改商品前改地址的顺序约束，3 个任务） |
| W1 followup | **真实失败归因 round 1**：针对上面那个具体 bug 打 patch 后，用完全相同协议重跑 test split 全量 115 任务 | pass^1 70.4%→72.2%（+2，81→83/115）；目标 3 个任务修好 2 个（72、98）；但同时 15 个任务翻对、13 个翻错——净值很小、双向抖动很大 |
| W1 followup | **噪声基线**：不改任何代码/prompt，把 baseline 协议原样再跑一遍 | pass^1 70.4%→78.3%（**+9**，81→90/115），14.8% 任务级翻转——**比 round 1 的 +2 还大**，结论是 round 1 的改进完全无法归功于那个 patch，单 trial 下这个协议的噪声本身就比想检测的效果大，见"噪声基线"一节 |
| W1 followup | **多 trial 对照（最终结论）**：unpatched/patched 各跑 3 trial（115×3=345 样本），配对 bootstrap 95% CI | pooled pass^1 72.17%→75.36%（+3.19%），CI=[-2.0%, +8.7%] 包含 0；pass^k（k=3）反而 57.39%→55.65%——**即使用严谨统计方法，仍无法证明这个具体修复有系统性效果**，见"多 trial 对照"一节 |
| 本轮新增 | **并行 tool call**：同一 turn 里 5 个 `get_order` 调用并发执行（真实 `time.sleep(0.2s)`，纯离线测试） | 串行理应 1.00s，实测 0.209s；mutating 调用仍严格顺序执行（专门测试验证），`tests/test_checkpoint.py` 20 个穷举崩溃点测试零改动全部照常通过 |
| 本轮新增 | **分布式漏洞修复 1**：`jobs` 表加部分索引（真实 Postgres，5 万条历史记录 + 20 条活跃任务） | 修前 Seq Scan，8.228ms；修后 Index Scan，0.046ms（约 180 倍），`EXPLAIN` 输出直接断言 |
| 本轮新增 | **分布式漏洞修复 2**：Gateway single-flight 锁加心跳续期（真实 Redis，`lock_ttl_ms=100` vs 真实 delay 400ms） | 去掉心跳复现漏洞（provider 被调用 2 次），加回心跳后只调用 1 次——两种状态都真实跑过，不是单边断言 |

### Evolve P0：真实 τ-bench 对话接入 rewind loop（本轮新增）

**为什么要做**：之前的真实基线（pass^1 = 70.4%）是用 tau-bench 自带的 `run.py` 跑的，没有经过 rewind 的 `AgentRunner`，所以这 115 个任务没有 trace、没有 env snapshot，真实失败无法 replay、fork 或归因。自迭代（Evolve）的 fork 初筛必须建立在这一步之上。

**做了什么**

- `env/taubench_dialogue.py`：`TauBenchDialogueEnv` 把 tau-bench 的 user simulator 变成 env 的一部分。agent 的文本回合交给 simulator（loop 新增的 `on_text_turn` 钩子），回复作为下一条 user 消息继续；`###STOP###` 或 `transfer_to_human_agents` 结束一轮，和 tau-bench `Env.step` 的两种结束方式一致。
- simulator 的 system prompt 直接调用 tau-bench 自己的 `build_system_prompt`（不是复制），角色约定也一致；它的整段对话是 env snapshot 的一部分，所以 **fork 到第 k 步时，simulator 的记忆也回到第 k 步**。
- `grade()` 是 tau-bench 的完整 reward：`r_actions`（数据 hash）加 `r_outputs`（必须说出的字符串）。之前 golden adapter 只能跳过的 37 个带 outputs 的 test 任务，现在可以正常评分。
- **snapshot 去重**：1.3 MB 的零售数据库按内容 hash 只存一份，每步的 env snapshot 只引用它；只有写操作之后才重新 hash。所有任务共享同一份初始数据库，读操作步骤几乎零成本。
- `run_episode` / `fork_episode` / `replay_episode`：replay 用 trace 里记录的 user 回复（`user_turn` 事件），不调用任何 API。
- loop 新增两个兼容选项：`single_tool_call`（和 tau-bench 的 agent 一样每轮只执行第一个 tool call）、`tool_error_format`（工具报错用 `Error: ...`）。`faithful_config()` 把这些和 30 步上限、wiki 作为 system prompt 组合在一起，并关闭 rewind 独有的循环检测和非法参数中止（tau-bench 没有这两项）。默认值保持 rewind 原有行为，原有测试全部不变。
- `rewind/evolve/stats.py`：把多 trial 对比脚本里的统计函数抽成库（数值与原脚本一致），并新增 tau-bench 的 pass^k 无偏估计、回归任务统计和 Wilson 区间。

**怎么跑**

```bash
export DEEPSEEK_API_KEY=sk-...
export REWIND_TAUBENCH_SRC=../tau-bench
# 1. 在 rewind loop 里重跑 test 基线（结果文件格式与 run.py 相同）
python scripts/run_taubench_rewind.py --split test --num-trials 3 --out runs/p0/test_rewind.json
# 2. 与 run.py 的多 trial 结果对比：P0 门槛之一是两者的 CI 重叠
python scripts/taubench_multitrial_compare.py <run.py 多trial结果> runs/p0/test_rewind.json \
    --label-a run.py --label-b rewind
# 3. 空补丁 fork 校准：fork 到不同位置、不打补丁续跑，看和原结果的一致率
python scripts/taubench_fork_calibration.py runs/p0/test_rewind.json \
    --store-dir runs/p0/snapshots --reruns runs/p0/test_rewind.json --n 5 \
    --out runs/p0/fork_calibration.json
```

**和 run.py 仍然存在的差异（如实记录，P0 对比时要考虑）**

- 工具列表顺序：rewind 按名字排序，tau-bench 按 `ALL_TOOLS` 原顺序；schema 和描述本身完全一致。
- 参数校验：rewind 先按 schema 校验参数，非法参数的报错文本和 tau-bench 的 Python 异常文本不同。
- 重试：模型 API 的瞬时错误 rewind 会重试；user simulator 调用也加了重试。tau-bench 遇到这类错误整轮直接失败，所以 rewind 的通过率可能略高一点。
- 模型输出无法解析时，rewind 会追加一条修复提示再试一次，tau-bench 会直接报错。
- 同样是 temperature 0，DeepSeek 也不保证逐字确定，所以 P0 的对比只看分布（多 trial + bootstrap CI），不看单次。

**离线测试**：`tests/test_taubench_dialogue.py`（9 个）用 golden 动作加脚本化的 user 回复覆盖对话回合、完整 reward、snapshot 去重、确定性 replay、空补丁 fork、反事实 fork 翻转、终止工具和报错格式；`tests/test_taubench_scripts.py` 用假模型端到端跑通两个脚本（含断点续跑）；`tests/test_evolve_stats.py` 覆盖统计库。全部 118 个测试通过。

**P0 实测结果与一个 bug**

- 门槛 1（迁移一致性）通过：rewind 3 trial pooled pass^1 = 73.3%，run.py = 72.2%，差值 +1.2%，95% CI [−3.5%, +5.8%]。两次相同设置的 run.py 互比 CI 为 [−8.4%, +5.5%]，rewind 落在同一噪声范围内；略偏高与 rewind 的重试和修复提示方向一致。
- 门槛 2（空补丁 fork）第一次没过：fork 到起点的通过率 20.4%，同一批任务重跑是 61.7%；各 fork 位置与父结果的一致率都只有 0.49–0.56。
- 原因：`SnapshotStore` 用 `sort_keys` 的 canonical JSON 落盘，数据库读回来后每条记录的键被按字母重排。tau-bench 的工具把记录原样 `json.dumps` 返回给 agent，所以新进程里 fork 出来的 agent 看到的工具返回和父 episode 不一样（`order_id, user_id, address…` 变成 `address, fulfillments, items…`）。同进程的测试读的是内存副本，reward 的 hash 又与键序无关，所以离线测试没发现。
- 修复：`SnapshotStore.put(..., ordered=True)` 按插入顺序哈希和落盘（`sha256o:` 键），对话 env 的数据库改用它；快照写入改为原子写。新增 `fork_fidelity` 检查和 `scripts/taubench_fork_fidelity.py`：从某一步 fork，回放父 episode 记录的模型回复和用户回复，逐步校验请求 hash，**不调用 API**。修复后 40 个任务 391 个 fork 点全部逐字节一致（修复前第一个 fork 点就不一致）。
- 教训：fork 的正确性不能只看"结果对不对"，要看"agent 看到的输入是否逐字节一致"；以后任何改动快照格式的地方，都先跑 fidelity 再花钱跑 calibration。

**修复后的空补丁 fork 校准**（20 个父 episode，失败/成功各 10 个，4 个位置 × 5 次，共 400 次续跑）

- 起点 fork 通过率 70% vs 同批任务重跑 63.3%：fork 恢复正确（修复前是 20%）。
- 成功的父 episode：各位置续跑通过率 86%–96%，成功基本可复现。
- 失败的父 episode：各位置续跑通过率 44%–52%。按任务看分成两类：任务 15/22/34/91 几乎从不通过（稳定失败）；任务 28/44/71/90/107 大多直接通过（偶发失败，父 episode 失败是噪声）。任务 102 起点 4/5、3/4 处 0/5，说明决定性错误发生在后半段。
- 123 次失败续跑里 27 次是 `r_outputs` 失败（数据库对了但没说出要求的内容）。
- 结论：只有稳定失败值得写编辑；空补丁 fork 通过率骤降的位置就是失败被锁定的位置。P1 据此加入"稳定性过滤"，并把初筛对照改为同任务同位置的空补丁基线。

### Evolve P1：自迭代闭环（本轮新增）

一轮的流程：现有版本在 train 上的结果 → 失败签名聚类（免费）→ 只取"普遍"的簇 → 稳定性过滤（空补丁 fork）→ 稳定失败分成 A/B 两半 → proposer 只看 A 提出候选编辑（1 次 LLM 调用）→ 离线检查 → 在 B 上 fork 初筛 → 在从未给 proposer 看过的任务上过统计门禁 → 提交或拒绝。

**模块**（`src/rewind/evolve/`）

| 文件 | 作用 |
| --- | --- |
| `surface.py` | `ContextVersion`：学到的规则（附加在 wiki 末尾的独立小节，原 wiki 不改）和工具说明补充；编辑是数据，函数式应用；泄漏检查（订单号、商品号、用户名、邮箱、任务编号）|
| `prevalence.py` | 确定性失败签名 `类别 | 工具 | 细节`（rewind trace 里有真实参数，所以能看出是哪个参数错了）；按**不同任务数**计数，用 Wilson 下界判定是否普遍 |
| `stability.py` | 每个失败从起点空补丁 fork，最多 1 次通过才算稳定失败；顺序执行，一旦超过就提前停止 |
| `proposer.py` | 一次调用返回多个候选（JSON 模式）；只给看 A 半的失败、正确动作和对话摘录，以及之前被拒绝的编辑和原因 |
| `screen.py` | 候选上下文下从起点 fork B 半任务，和**同一任务**的空补丁通过率比较；任务内、任务间都会提前止损 |
| `gate.py` | 默认**定向门禁**：① 有效性——在初筛用过、proposer 没见过的稳定失败上重新 fork 5 次，与它们的空补丁通过率做单侧 Fisher 检验（p < 0.05 且提升 ≥ 0.3），先跑、不过就不跑②；② 安全性——在门禁集上与现有版本配对比较，不能显著变差（CI 上界 ≥ 0）且下降 ≤ 2 个点。回归数只报告不否决（基线约 90% 时，什么都不改也会有约 5% 的成功任务重跑失败）。另保留整体显著提升的 superiority 模式。现有版本结果复用，每任务 1 个 trial，先跑一半，明显有害就停 |
| `ledger.py` | SQLite：所有版本、每个候选在哪一步被拒及原因、哪些任务被 proposer 或初筛看过（永久排除出门禁集）|
| `loop.py` | 一轮的编排；每一步先按最坏情况检查预算，`plan()` 不调 API 打印本轮最坏用量 |
| `runner.py` | 按版本批量跑真实 episode，可断点续跑（`run_taubench_rewind.py` 也改用它）|

**为什么从起点 fork 而不是从中间**：上下文编辑从第一轮就会改变行为，把新 system prompt 拼到旧 prompt 下生成的历史里，测的是"规则在第 k 步突然出现"，是有偏的反事实。P0 校准也显示起点 fork 和重跑一致，所以初筛统一从起点 fork。

**范围**：只做工程上的自迭代——模型权重不动，演化对象是上下文（学到的规则、工具说明）和 harness；不做 SFT/DPO 等任何训练。

**省 API 的设计**（DeepSeek 用量有限）：签名、聚类、计划、状态全部免费；稳定性和初筛顺序执行并提前停止；门禁复用现有版本结果、默认 1 trial、半程止损；每轮最多一个候选进门禁；`--budget` 是每轮硬上限；`round` 默认先打印计划并要求确认。

**离线端到端测试**（`tests/test_evolve_loop.py`）：假 agent 在真实 retail 任务、真实工具和真实 grader 上运行，除"第一个写操作是换货"的 17 个任务外都按正确动作完成，这 17 个任务只有 system prompt 里出现特定规则才会做。一轮下来：泄漏 id 的规则被离线拒绝；"禁止调用工具"的有毒规则在初筛 4 次 fork 后被淘汰；正确规则初筛 3/3、门禁 54 个任务 +20.4%（CI [+11.1%, +31.5%]，修好 11 个、弄坏 0 个）被提交；共用 79 次 episode。另外两个测试覆盖"修好簇但弄坏其他任务的规则被门禁以回归拒绝"和"预算不够时在门禁前停下"。

**怎么跑**

```bash
# 1. train 基线（失败样本来源，同时是 v0 的门禁对照）
python scripts/run_taubench_rewind.py --split train --task-ids 0-199 \
    --out runs/p1/train_base.json --store-dir runs/p1/snapshots
# 2. 看本轮最坏用量（不调 API）
python scripts/run_evolve.py plan --base runs/p1/train_base.json --store-dir runs/p1/snapshots
# 3. 跑一轮（会先打印计划并要求确认）
python scripts/run_evolve.py round --base runs/p1/train_base.json --store-dir runs/p1/snapshots --budget 250
# 4. 查看结果（不调 API）
python scripts/run_evolve.py status
```

**为什么改成定向门禁**：train 基线 89.5%（21/200 失败），两个可用簇各 9 个任务。簇里的稳定失败全部被 proposer 和初筛用掉后被排除出门禁集，整体比较几乎看不到修复；而原来的回归数上限 2 会连空操作都拒绝。

**第一轮实跑（train v0，21 个失败，两个 9 任务的簇）**：第一个簇的候选没过初筛；第二个簇由 DeepSeek 写出的一条规则初筛通过（3 个 proposer 没见过的稳定失败中 2 个明显被修好，它们不改时成功率为 0）。但门禁最坏需要 203 次、剩余预算 167 次，系统按设计停下，本轮共用 93 次 episode、2 次 proposer 调用。预算估错的原因是只按一个簇估算，实际处理了两个簇。改进：`plan` 现在直接给出"让第一个簇能走到门禁所需的预算"；通过初筛但没钱过门禁的候选会结束本轮，并可以用 `run_evolve.py gate-pending` 单独补跑门禁，前面的稳定性、初筛和 proposer 都不重做。

**已知局限**：安全性检查每任务 1 个 trial、约 190 个任务，只能发现约 4–5 个点以上的伤害，更小的副作用会漏过；有效性检查只有 2–3 个目标任务，只能证明较强的效果；签名是规则化的，语义相近但签名不同的失败不会合并；校验模板（规则下沉为代码检查）留到 P3。

### Evolve 第一个真实结果：一条有效的规则，先被拒、重审后接受（本轮新增）

**过程**（train v0：200 个任务，pass^1 89.5%，21 个失败）

1. 聚类：两个可用簇，"换货出错"和"改订单商品出错"，各 9 个任务（21 个失败里的 18 个）。
2. 第一个簇的候选没过初筛。第二个簇（`WRONG_ACTION|modify_pending_order_items`）由 DeepSeek 写出的规则 `empty_variant_means_self_swap` 通过初筛：

   > When a customer asks to modify an item's options to empty ({}), perform a same-item modification by passing the current item_id as both item_ids and new_item_ids; do not refuse or offer alternative variants.

3. 本轮预算不够过门禁（需要 203，剩 167），用 `gate-pending` 单独补跑，前面的步骤没有重做。
4. **门禁一（有效性）通过**：proposer 没见过的 3 个稳定失败（任务 8、55、149），不改时 15 次全部失败，加规则后 15 次成功 6 次，提升 +0.40，单侧 Fisher p = 0.008。
5. **门禁二（安全性）拒绝**：188 个没见过的任务各跑 1 次，修好 5 个、弄坏 12 个，pass^1 下降 3.7 个点，95% CI [−8.0%, +0.5%]，超过"下降不超过 2 点"的上限。门禁二共 188 次 episode，加门禁一 15 次，合计 203 次。

**拒绝是不是噪声？事后归因（`scripts/evolve_gate_report.py`，不调 API）**

- 被弄坏的 12 个任务**全部**涉及 `modify_pending_order_items` 或 `exchange_delivered_order_items`，也就是这条规则改变的"商品怎么选"的行为。
- 前 200 个 train 任务里涉及这两个工具的只占 64.5%。如果 12 个失败只是重跑噪声，全部落在这类任务上的概率是 0.645¹² ≈ 0.5%。**（更正，见下一节：这组工具是看到结果之后才挑的，这个 p 值偏乐观，不能当作副作用的证明。）**
- 修好的 5 个任务也全部是这类任务（随机约 11%）：规则确实作用在目标行为上，只是范围写宽了——后半句"不要拒绝、不要推荐其他规格"把正常的改商品流程也带偏了。

**当时的结论（已更正，见下一节）**：曾认为"门禁拦下了一个真实的副作用"。下一节的校准表明：拒绝依据的"下降不超过 2 点"规则本身会误杀约五分之一的无害规则，而事后的工具集中度检验是在看到结果后选的分组。正确的说法是：这条规则**可能**有副作用，现有证据不足以下结论；最终重审结果见下文"重审结果"。

**花费**：第一轮 93 次 episode + 2 次 proposer 调用；`gate-pending` 203 次。train 基线 200 次另计。

**下一步（未做）**：把门禁拒绝的具体信息（弄坏了哪类任务、一个被弄坏任务的对话片段）反馈给 proposer，并允许复用同一簇的 A/B 分组重写更窄的规则；每次重试约 220 次 episode。

### 失败分类修正：只有"没恢复的报错"才算工具报错（本轮新增）

**问题**：`classify_taubench` 原来只要对话里出现过一次 `Error:` 就归为 `TOOL_RUNTIME_ERROR`，而且优先级最高。agent 中途报错、重试成功、最后因为别的原因失败，也会被算成工具报错，签名里还只记最后一次报错。

**修复**：一个报错只有在之后**同一工具或功能等价的工具**再也没成功过时，才算"没恢复"。等价关系用一张显式的表定义，目前只有两个查用户的工具互为等价（邮箱查不到就改用姓名 + 邮编，这是 τ-bench 里很常见的恢复路径）。恢复了的报错记录在 `recovered_tool_errors`，不再决定类别；签名改用最后一个没恢复的报错及其工具。工具名优先读消息里的 `name`（run.py 格式），其次按 `tool_call_id` 对回调用，最后用前一条调用（兼容 rewind 自己导出的格式）。

另外新增 `EPISODE_NOT_FINISHED`：run.py 在对话跑满步数时根本不打分（`reward_info` 为空），rewind 则会照常打分但记录下停止原因，两种情况都归到这一类。

**影响**（test 3-trial，96 个失败，不调 API）：

| 类别 | 修复前 | 修复后 |
| --- | --- | --- |
| TOOL_RUNTIME_ERROR | 26 | 10 |
| WRONG_ACTION | 51 | 55 |
| ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT | 13 | 20 |
| WRONG_OUTPUT | 6 | 10 |
| EPISODE_NOT_FINISHED | — | 1 |

原来的 26 个"工具报错"里有 16 个其实已经恢复，它们原本应该进入操作错误或输出错误的簇。第一轮 Evolve 用的两个簇都是 `WRONG_ACTION`，没有受影响；但以后的聚类会更准确，部分簇会变大。

### 门禁校准：旧的安全规则会误杀约 20% 的无害规则（本轮新增）

**发现**：做分层安全检查前，先模拟了"规则完全无害、只有重跑噪声"时门禁二的表现（基线 89.5%，188 个任务各 1 次，任务级的重跑一致率取自 P0 校准）。"平均下降不超过 2 点"这条规则会拒绝 **18–24%** 的无害规则——它测到的主要是噪声，而这个阈值当初没有先用噪声校准。

**对第一轮结论的更正**：
- 在整个门禁集上，弄坏 12 个对修好 5 个，单侧符号检验 p = 0.072；配对 CI [−8.0%, +0.5%] 包含 0。按校准过的标准，这条规则**不会**被判为显著有害。
- "12 个全落在改商品/换货任务上，随机概率 0.5%"：这组工具是看到哪些任务坏了之后才挑的。如果事先按"会用到 item_ids 参数的工具"分组，这类任务占门禁集的 82%（155/188），12 个全在里面的概率约 9%，不稀奇。
- 所以：**这条规则可能有副作用，但证据不足。** README、面试素材里"门禁拦对了"的说法已改。之后用 `run_evolve.py regate` 按新标准重审：12 个回归 0 个复现，规则被接受为 v1（见下文"重审结果"）。

**新的安全检查：确认翻转**（`gate.py`）。门禁二跑完后，把翻转的任务（被弄坏的、被修好的）在新旧两个版本下各重跑 k 次。只有"旧版本每次都过、新版本每次都挂"才算确认的回归；超过 1 个确认回归就拒绝。整体 CI 显著变差仍然直接拒绝。不同规则的模拟结果（任务级的不稳定性按 P0 数据拟合：80% 的任务通过率 0.97，14% 约 0.7，6% 约 0.1）：

| 规则 | 无害规则被误拒 | 5 个真坏任务被抓到 |
| --- | --- | --- |
| 旧：平均下降 > 2 点就拒绝 | 18–24% | — |
| 确认：旧版本至少过 1 次，k = 2 | 13% | 100% |
| 确认：旧版本至少过 1 次，k = 3 | 5% | 100% |
| **确认：旧版本每次都过，k = 2（采用）** | **约 3%** | **约 99%** |
| 确认：旧版本每次都过，k = 3 | 0.2% | 约 99% |

第一次实现用的是"旧版本至少过 1 次"，模拟后发现硬币型任务经常连挂两次，误拒率 13%，所以改成"每次都过"。新增测试覆盖了三种情况：偶然翻转的任务不会被确认（规则被接受）；小而真实的破坏会被确认，即使整体 CI 是正的；按旧规则被拒的无害规则，用 `regate` 重审后被接受。

**为什么没有做"按工具分层先跑"**：原计划先跑最可能受影响的任务、明显变差就提前停。但规则涉及的工具往往覆盖了门禁集的大部分（第一轮的规则对应 82%），先跑这一层省不下什么。确认翻转更便宜（约 4 × 翻转数次 episode），判断也更准。

### 重审结果：第一条通过门禁的规则（v1）

用 `run_evolve.py regate` 按校准过的门禁重审第一轮被拒的规则 `empty_variant_means_self_swap`。门禁一结果和门禁集 188 次结果都复用，只新跑确认：17 个翻转任务（12 个被弄坏、5 个被修好）在新旧两个版本下各重跑 2 次。

| 项目 | 结果 |
| --- | --- |
| 门禁一（有效性） | 目标 6/15 vs 不改 0/15，p = 0.008（复用） |
| 门禁集整体 | −3.7 点，95% CI [−8.5%, +0.5%]，包含 0（不判为显著有害） |
| 确认翻转 | **12 个回归 0 个复现，5 个修复 0 个复现** |
| 结论 | **接受，提交为 v1** |

**解读**：门禁集上的翻转（无论"弄坏"还是"修好"）在重跑时都不复现，说明当初 −3.7 点主要是重跑噪声，不是规则造成的稳定破坏。第一轮的"拒绝"是旧安全规则校准不当造成的误拒，属于模拟里那 18–24% 的情况。

**仍然没证明的**：
- 确认翻转只能抓"可复现的破坏"；分散在很多任务上、每个都很小的伤害抓不到。整体 CI 下界 −8.5% 也排除不了小幅伤害。
- 这条规则是在"必须写触发条件"之前写的（触发条件写在正文里，`trigger` 字段为空）；规则本身仍有"不要推荐其他规格"这种在条件句里的限制。
- v1 只在 train 上验证过，还没在 test split 上评估。

### 拒绝反馈、范围约束、重试和查看器（本轮新增）

- **具体的拒绝反馈**（`evolve/feedback.py`）：门禁拒绝时自动生成反馈：门禁一结果、门禁集修好/弄坏多少、被弄坏任务在哪个工具上集中（附二项分布尾概率，标出是否可能只是巧合）、一个被弄坏任务前后两个版本的工具调用对比。同簇的被拒规则排在 proposer 提示的最前面，附带完整反馈。旧的拒绝记录会自动补上反馈（读磁盘上的结果，不调 API）。
- **范围约束**（`surface.py`、`proposer.py`）：每条规则要写明触发条件（`trigger`），写入上下文时是"When <触发条件>: <规则>"。全局规则必须有几个词以上的触发条件；只涉及一个工具的规则会被要求改成那个工具的说明（tool_note），只在 agent 考虑这个工具时出现。proposer 的提示要求触发条件尽量窄，禁止在触发条件之外使用无条件的"never / always / do not"。
- **同簇重试**（`run_evolve.py retry`）：复用已有的 A/B 分组，不重新做稳定性过滤，不消耗新任务；每个簇最多 3 次。注意：B 组在多次初筛中被重复使用，初筛结果会略偏乐观；门禁一用新的 fork，门禁集里永远没有 A/B 任务。
- **重审**（`run_evolve.py regate`）：按当前门禁标准重新决定早先被拒的规则；复用门禁一结果和已跑过的门禁集，只新跑确认重跑。
- **查看器**（`viewer/index.html` + `scripts/export_viewer.py`）：单个 HTML 文件，不联网。导出脚本把账本和相关 trace 打包成一个 JSON；页面展示每轮的决策、每个版本的规则、门禁数据和反馈，点击任意一个翻转的任务，可以并排对比它在新旧版本下的完整对话，并标出第一处不同的工具调用；也可以直接拖入任何 `.jsonl` trace（包括 fork 出来的，继承的前缀会变暗）。

### 规则下沉为代码检查（P3 第一步，本轮新增，0 次 API）

**动机**：写在提示里的规则是软约束。第一条通过门禁的规则（v1）在目标任务上只修好了 15 次中的 6 次：agent 读到了规则，但不一定照做。τ-bench 零售政策里有一部分是可以机械判断的，而 τ-bench 自己的工具**不检查**它们。逐个读了 16 个工具的源码，订单状态、支付方式、商品是否同款这类条件，工具自己都会报错；但下面三条只写在 wiki 里，工具完全不管：

1. 修改类操作前，必须列出细节并得到用户明确的 yes；
2. 先认证用户身份，一次对话只帮一个用户，只操作这个用户自己的订单；
3. 改订单前先查订单详情（wiki 对换货有明确要求）。

**设计**（`src/rewind/policy.py`）：

- **检查是数据，不是代码。** 一个 `Check` = 守护哪些工具 + 从固定条件库里选一个条件 + 参数 + 拦截时告诉 agent 的话。条件库目前有 6 种：参数关系（`args_same_length`、`arg_in`）、环境状态（`state_field_in`）、对话历史（`called_before`、`user_confirmed`）、身份（`authenticated_owner`）。LLM 只能组合、填参数，不能写新逻辑；未知条件、缺参数、多参数、带订单号/邮箱等具体值都会在离线校验时被拒。
- **每个条件都是纯函数**，输入只有三样：这次调用、数据库只读视图、到目前为止的对话。所以同一个检查既能在线执行，也能在录下来的 trace 上离线重算，结果一致（有测试对比两者）。
- **"用户确认"的判定**：最近一条用户消息是明确的 yes（且不以 no / wait 开头），它回答的是 agent 的上一句话（开场白里说的"yes"不算，因为 agent 还没列出细节），并且这个 yes 还没有被一次成功的写操作用掉。只伴随工具调用的 agent 文本不算"说过"——在 τ-bench 的循环里，这种文本根本不会发给用户。
- **接入运行时**（`loop.py` 的 `_tool_step`）：在死循环检测之后、执行工具之前。不通过时工具**不执行**，数据库不变，agent 收到 `Error: [policy:<id>] <说明>（<原因>）`，可以在同一个对话里改正（去问用户、去认证）。每次触发都写一个 `guard` 事件，trace 和查看器里能看到。
- **防止卡死**：同一个检查在一个对话里最多拦 2 次，之后放行并记录 `allow_over_budget`。另有 `observe` 模式：只记录不拦截，用来在真实运行里"影子观察"一个检查。
- **检查挂在 ToolRegistry 上**，所以 run、fork、replay、Evolve 的初筛和门禁全部自动带上，不需要改任何调用方。用同一个策略 replay 一个带拦截的 trace 是逐字节一致的；换成不带策略的 registry 则在拦截那一步分叉（有测试）。

**进入 Evolve**：`ContextVersion` 多了第三种可改的东西 `checks`，版本、回滚、账本、指纹全部复用（没有检查的旧版本指纹不变）。proposer 的提示里说明了何时该用检查（"做错的动作"）、何时不该（"该做没做"），并列出条件库。新增 **backtest 阶段**，排在初筛之前：

- 在当前版本的全部历史对话上回放这个检查：在**成功**的对话里，它在多少次被守护的调用上会触发（误拦率，上限 2%）；在**本簇失败**的对话里，有没有触发在一个标准答案里没有的调用上（命中，至少 1 个）。
- 两个条件都满足才花初筛的 fork；不满足就记为 `backtest` 拒绝，原因进入下一次给 proposer 的反馈。整个阶段 0 次 API。
- 测试覆盖了两种典型的 0 成本拒绝：一个检查在成功对话里乱拦（误拦率太高），一个检查对"agent 直接拒绝"类的簇无从下手（没有调用可拦）。

**离线回测脚本**（`scripts/backtest_checks.py`）：对任意 rewind 结果文件 + 快照目录，输出每个检查在成功对话里的触发次数/被守护调用数、在失败对话里命中标准答案调用的次数（会拦错）、命中非标准答案调用的次数（抓到），以及每类的例子。三个手写检查放在 `data/policy_checks/taubench_retail.json`。

**还没做 / 没证明的**：

- 回测只回答"会不会触发"，不回答"拦下之后 agent 能不能改对"——那是反事实，必须真跑（走正常的初筛和门禁）。
- 三个手写检查还没在真实的 DeepSeek trace 上回测（沙箱里没有 runs/ 数据）。下一步是用 `train_base.json` 跑一次，看真实误拦率。
- 一个回合里有多个写操作时（τ-bench 是单工具调用，不受影响），"一个 yes 只用一次"按回合开始时的历史计算。
- 用户确认靠正则判断 yes / no，用户说法很绕时可能误判；误判的代价是多问一次，不会执行错误操作。

## 已知局限（诚实版）

- Fork 前缀是**复制**而非引用共享；blob 内联在事件里，未走对象存储。
- 每个 tool 后都存一次 env snapshot（玩具环境够用，真实环境应只在 hash 变化时存或改增量）。
- 回放/分叉/checkpoint 目前绑定 toy env 的 CLI；`env/taubench_adapter.py` 已经实现了 `snapshot/restore/applied/db_hash`，但**没有**接 replay/fork/checkpoint 的 CLI 命令（`rewind demo`/`ckpt-demo`/`attribute-demo` 这些仍然只认 toy env），也**没有记录 user simulator 的 LLM 调用**——因为这个 adapter 完全没有驱动真实 user simulator（见下一条）。
- **（已由 Evolve P0 解决，见上文 `taubench_dialogue.py`；以下为当时的记录）tau-bench adapter 没有、也不能模拟真实用户对话**：tau-bench 的任务是多轮对话，需要一个真实 LLM 驱动的用户模拟器（`tau_bench.envs.user`，基于 `litellm`），这个沙箱没有任何 LLM API key，agent 侧和用户模拟器侧都调不了。`env/taubench_adapter.py` 能验证的只是"黄金轨迹回放+状态哈希比对"这一半（`r_actions` reward 组件），test split 里 37/115 个需要"agent 说过的话"（`r_outputs`）的任务被显式识别并跳过，不是悄悄算进同一个通过率——100% 这个数字衡量的是"adapter 接得对不对"，不是"模型解决任务的能力"，这两者不能混为一谈（和 W7/W8 合成 ground truth 的局限是同一类问题：排除了"adapter 本身有 bug"的风险，不能预测真实模型表现）。
- tau-bench adapter 跑一个任务（含每步的 env snapshot 深拷贝 ~1550 条记录的 DB）大约 0.9-1s，500 个 train 任务跑全量要 8-9 分钟——这个开销目前没有优化过（深拷贝整个 DB 字典，不是增量/引用共享，和上面"snapshot 目前是复制不是引用共享"是同一个局限的另一个实例），对"黄金轨迹回放校验"这个用途够用，接真实模型跑大规模 eval 时需要重新考虑。
- 这个沙箱装 `tau_bench`（走 `pip install git+...`）会撞上系统 setuptools 的一个 legacy-`setup.py`-安装坑（`AttributeError: install_layout`），和 tau-bench 自己的代码无关；这里用 `REWIND_TAUBENCH_SRC` 环境变量绕过（直接把 clone 的目录加进 `sys.path`，不走 pip 的 build 步骤），正常机器上大概率不会碰到这个问题。
- 幂等台账 `env.applied` 目前是 env 状态的一部分，靠"和 env 一起被 checkpoint、一起被恢复"来保证跨崩溃安全；如果换成真正的外部系统（订单 DB、支付 API），台账必须是外部系统自己承诺的幂等键（如 Idempotency-Key），而不能只依赖我们这边 checkpoint 的时机——这是分布式幂等的标准做法，本仓库只是在单一玩具 env 里把同样的原则演示了一遍。
- `AnthropicProvider` 与 `run`/`fork` 子命令没有真实调用测试；`temperature` 若被所用模型拒绝，把 `AgentConfig.temperature` 设为 `None`。
- 续跑用的 provider（`provider_for`/`_provider_for`）目前都要调用方手工提供，还没有"续跑也从成功轨迹自动配对"这一层——只有候选步骤本身的修正动作（`patch_for_step`）自动化了（`auto_patch_for_step`，本轮 W7 新增）。
- `attribute-demo`/`attribute-eval` 用的是确定性的 `ScriptedProvider`，所以每个候选步骤的翻转率只会是 0 或 1；真实场景接 live API 后 `n_samples` 的方差才有意义，`flip_rate_via_fork` 已经按"每个 sample 独立 fork+跑"实现，接上真模型不用改这层。
- W7 的归因准确率数字（100% exact-match）来自**合成** ground truth（根因步号是构造时就已知的），不是方案要求的人工标注 30-50 条真实失败——还没有真实失败语料可标注（τ-bench adapter 还没写），如实标注而不是拿合成数字冒充真实评测结果；真实场景下受非单调、环境随机性、live API 方差影响，准确率不会是 100%。
- 性能消融清单里的并行 tool call 本轮补上了（见"并行 tool call"一节）；prompt caching、模型路由、tool 结果截断、context compaction 阈值**仍然**完全没做——这些机制本身在 harness 里都还不存在，消融的前提是"开关已经存在"，属于比 W7 已完成部分更大的一块独立工作。
- 失败聚类目前只按 (类别, 根因工具) 两个维度分组，没有错误信息的 embedding 子聚类（如实标注在上面 W7 一节，原因是这个 toy env 的失败模式还不够多样，值不回加一层 embedding 的成本）。
- `PgCoordinator` 的并发测试证明了"8 个并发 claim 不重复"，但没有跨**多台机器**验证——本轮 W6 补上了真实 `kill -9` 一个操作系统进程的验证（同一台机器上的两个独立进程），跨机器（不同主机、真实网络分区而不是同机器上用 Toxiproxy 代理模拟）仍然没做。
- 熔断器的"半开"是用开闸 key 的 TTL 到期简化实现的，不是标准的"放一个试探请求过去、成功才关闸、失败重新计时"那种单次试探型半开——`test_circuit_breaker_recovers_after_cooldown_and_a_success_clears_it` 验证的是"TTL 到期后新请求能过"，不是"半开期间只放一个试探请求"。
- `DualTokenBucket` 的 `token_cost` 在请求真正发出、模型还没返回之前只能是估算值（`tokens_per_request_estimate`），跟实际消耗的 token 数会有误差；真实场景应该用上一轮的真实用量去动态调整这个估算，本仓库还没做。
- 优先级/公平调度（同一批任务里不同 run 按优先级或按租户公平分享 worker）、Trace Writer 落盘到持久化 sink（现在 `TraceStreamConsumer.drain()` 吐出的事件只是被测试断言，没有真的写回一个长期存储）、Mock LLM 服务（带真实延迟分布用于压测）、MinIO 对象存储（blob 目前内联在 SnapshotStore/checkpoint bundle 里）——这四项在方案里都提到过，本轮 W5 明确没做，留给需要时再补。
- 扩展性实验的 mock LLM 是零延迟的——真实场景下 LLM 调用本身的延迟通常远大于 Postgres 往返，所以这条曲线只隔离出了"协调层自己的开销上限"，不是端到端吞吐的预测；要预测真实吞吐还得把 `dist/gateway.py` 的限流/熔断也计入负载模型。
- 恢复时间实验目前只测了"一次 kill、干净的第二个进程接管"这一种场景；没有测"接连 kill 好几次"（worker B 自己也被杀，worker C 才最终完成）或者"kill 发生在网络故障期间"（Toxiproxy toxic 和真实进程 kill 组合），这两个是更贴近生产事故复盘的场景，留给以后按需补。
- **W8 飞轮里的"模型"是确定性启发式函数，不是真实 LLM**：`HeuristicRetailProvider` 用两个可开关的布尔标志位模拟两个真实 prompt/工具改进分别会修好什么，因为这个沙箱没有可用的 LLM API key。这一节验证的是飞轮机制本身（切分/分类/聚类/门禁/合并/汇报）是真实可跑、可复现的，不是"真实 LLM 经过这套流程后效果提升了多少"——这两件事不能混为一谈，接上真实 API 后需要重新measure。
- **W8 的"docker compose up"式一键复现并不存在**：这个沙箱没有 docker daemon 可用的镜像仓库（见上面 W6 一节的坑），所以"复现结果"实际走的是"启动系统服务 + 跑脚本"（`service postgresql start` / `pytest` / `python scripts/xxx.py`），不是方案原文字面写的 `docker compose up`；两条路径对结果本身没有影响，但如实标注这里的落地方式和方案原始设想不完全一样。
- `.github/workflows/ci.yml` 是一份真实、语法有效的 GitHub Actions 配置，但没有被真正的 GitHub Actions 跑过——这个沙箱不是 git 仓库，没有 GitHub 连接；它调用的两个脚本（`ci_pr_gate.sh`/`ci_nightly.sh`）本身是在这个沙箱里真实运行并通过的。
- 可选的 viewer（读取 JSONL、展示步骤时间线与 fork 树的小前端）**没有做**——方案本身把它标注为"（可选）"，本轮判断优先级更高的是把飞轮机制和 CI 门禁做实，所以明确跳过，留给以后需要可视化调试时再补。
- W8 的数据飞轮只跑了 2 轮、候选修复来自一张写死的查找表（`CANDIDATE_FIXES`）——这是刻意的简化（方案要求的是"有候选修复方案"而不是"自动发现修复方案"），不是自动生成修复的系统；一个真正的自动化版本需要接上能读 top cluster 里的失败样本、自己提出 prompt/工具描述改动的机制，这本身是更大的一块工作。

## 下一步

0. **Evolve（自迭代，当前主线）**：P0、P1 已完成，第一个真实结果见上文。下一步是拒绝原因的具体反馈和同簇重试，然后用最终版本在 test 上跑 3 trial 做最终评估（E1）。W8 一条里"候选修复从写死的查找表换成自动生成"已由 Evolve 的 proposer 实现。
1. W1 收尾（**这条线已完整收尾**）：真实 DeepSeek baseline（pass^1=70.4%，81/115）→ 真实失败归因（`flywheel/taubench_failures.py`）→ 挖出一个具体可复现 bug（改商品锁单导致改地址失败）→ round 1 真实 patch + 重跑（单 trial +2，看起来有效）→ 噪声基线对照（什么都不改也有 +9，比 round 1 还大，起疑）→ 认真跑 `--num-trials 3` 的多 trial 对照 + 配对 bootstrap 统计检验（最终结果：pooled pass^1 72.17%→75.36%，95% CI=[-2.0%, +8.7%] 包含 0；`pass^k`(k=3) 反而从 57.39% 降到 55.65%）。**最终诚实结论：排除噪声之后，仍不能证明这个具体修复有统计意义上的系统性效果**——这不是失败，是一次完整、方法论上站得住的"评估一个 LLM agent 改动有没有效"的真实案例，大多数只跑一次就下结论的"改进"在这个标准下都站不住。还没做的（可选，按需再做）：① 如果还想继续，可以换一个更大的候选修复（比如针对 WRONG_ACTION 里"工具名对、参数错"那 12 个案例，或者把 `action_shape()` 细化到真正逐参数 diff）重新走一遍这整套流程；② 把 `rewind demo`/`ckpt-demo`/`attribute-demo` 这些 CLI 命令接到 tau-bench env（而不是只认 toy env）仍然没做。
2. W5 收尾（可选，按需再做）：优先级/公平调度；Trace Writer 接一个真正的持久化 sink；Mock LLM 服务；MinIO 对象存储。另外本轮代码审查还发现了 4 个没修的分布式漏洞（按价值排序）：① Gateway 的缓存 key（`request_hash`）不含模型/provider 身份，同一个 Redis 被不同模型共享时会撞 key，可能把 A 模型的回复当成 B 模型的回复返回；② `TraceStreamConsumer.seen_seqs` 只是进程内存里的 set，consumer 进程本身重启后这份去重状态就丢了，不是真正跨重启持久的去重；③ `drain()`/`reclaim_stale()` 遇到解析失败的消息（毒丸）会在 `xack` 之前抛异常，这条消息永远不会被确认，会无限重投递、无限再抛同一个异常；④ `jobs.attempt` 只递增，没有上限或死信机制，一个总是让 worker 崩溃的任务会在 claim→崩溃→lease 过期→再 claim 之间无限循环。
3. W6 收尾（可选，按需再做）：跨机器的 chaos 验证（不只是同机两个进程）；接连多次 kill 的级联恢复场景；Toxiproxy 故障和真实进程 kill 组合的场景；扩展性实验换成有真实延迟分布的 mock LLM。
4. W7 收尾（可选，按需再做）：剩余性能消融（prompt caching、模型路由、tool 结果截断、context compaction——并行 tool call 本轮已做，见"并行 tool call"一节；每项都要先实现对应机制）；接 τ-bench 后在真实失败样本上人工标注 30-50 条根因、重跑 `attribute-eval` 拿真实（而非合成）准确率数字；续跑 provider 的自动配对；把并行 tool call 接到 `env/taubench_adapter.py` 上实跑验证（理论上直接适用，但没有实跑测过）。
5. W8 收尾（可选，按需再做）：接真实 LLM API 重跑飞轮，拿真实的"改进后效果提升多少"数字（而不是这里的机制验证）；候选修复从写死的查找表换成真正自动生成；真的推到 GitHub 上验证 `.github/workflows/ci.yml`；可选 viewer（JSONL 时间线 + fork 树可视化）。
