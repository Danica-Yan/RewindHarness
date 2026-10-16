"""Rewind: a replayable, forkable, checkpointed agent runtime (skeleton for weeks 1-4)."""
from .checkpoint import Checkpoint, Coordinator, JobSpec, StaleFencingToken, StepCheckpointer
from .events import Event
from .flywheel import AttributionResult, bisect_attribute, flip_rate_via_fork
from .flywheel.failures import (
    FailureRecord, auto_patch_for_step, category_distribution, classify, classify_worker_crash,
    cluster, top_clusters)
from .fork import ForkPatch, fork_run
from .loop import AgentConfig, AgentRunner, Checkpointer, LoopState, RunAbort, RunResult
from .providers import (
    FaultInjectingProvider,
    LLMResponse,
    MalformedOutput,
    Provider,
    ProviderError,
    ScriptedProvider,
    ToolCall,
)
from .replay import ReplayReport, replay_run
from .snapshot import SnapshotStore
from .tools import Tool, ToolError, ToolOutcome, ToolRegistry, TransientToolError
from .trace import TraceWriter, load_trace
from .worker import SimulatedCrash, run_worker_step, submit_job, submit_job_async

__version__ = "0.2.0"
