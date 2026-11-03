"""Fine-tune a small multi-label intent router and evaluate it with the shared protocol.

Default: Qwen3-0.6B + LoRA, sequence-classification head (sigmoid per intent, no text generation ->
no parsing failures, direct probabilities, one forward pass). Encoders work too (--model
answerdotai/ModernBERT-base or BAAI/bge-base-en-v1.5, use --no-lora for full fine-tuning).

Kaggle (T4 x1, fp16):
  !pip -q install peft deepseek-tokenizer
  !python scripts/finetune.py --model Qwen/Qwen3-0.6B --seeds 0 1 2 \
      --extra data/processed/tau_traj_train.jsonl --out out/qwen3_0.6b_lora
CPU smoke test (tiny random model, no download):  python scripts/finetune.py --smoke

Never trains on any row with split == "test".
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intent_funnel.budget import PromptBudget  # noqa: E402
from intent_funnel.data import eval_sets, load, load_replays, make_train  # noqa: E402
from intent_funnel.evaluate import evaluate, headline, latency, to_markdown  # noqa: E402
from intent_funnel.router import SentenceUnion  # noqa: E402
from intent_funnel.taxonomy import INTENTS  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def smoke_model_and_tokenizer(n_labels):
    """Tiny random Qwen3 + word-level tokenizer built locally: exercises the full code path on CPU."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForSequenceClassification
    rows = load()
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator((r["text"] for r in rows[:20000]),
                            trainers.WordLevelTrainer(vocab_size=8000, special_tokens=["[PAD]", "[UNK]"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]")
    cfg = Qwen3Config(vocab_size=hf_tok.vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      num_labels=n_labels, problem_type="multi_label_classification",
                      pad_token_id=hf_tok.pad_token_id)
    return Qwen3ForSequenceClassification(cfg), hf_tok


class HFRouter:
    """predict_proba wrapper so the fine-tuned model plugs into the shared evaluation."""
    name = "hf"

    def __init__(self, model, tok, device, max_len, batch=32):
        self.model, self.tok, self.device, self.max_len, self.batch = model.eval(), tok, device, max_len, batch

    @property
    def classes(self):
        return list(INTENTS)

    def predict_proba(self, texts):
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(texts[i:i + self.batch], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits.float()
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, len(INTENTS)))


def _disable_peft_torchao():
    """Kaggle preinstalls torchao 0.10; recent peft raises ImportError on any torchao < 0.16 while
    dispatching LoRA layers, even though we never use torchao. Tell peft it isn't there."""
    try:
        import peft.import_utils as iu
        iu.is_torchao_available = lambda: False
        import peft.tuners.lora.torchao as lt
        lt.is_torchao_available = lambda: False
    except Exception as e:  # older/newer peft layouts: harmless if the hook doesn't exist
        print("note: torchao patch skipped:", e)


def compat_kwargs(cls, kw):
    """Drop kwargs the installed transformers version doesn't accept (4.x on Kaggle vs 5.x)."""
    import inspect
    ok = set(inspect.signature(cls.__init__).parameters)
    dropped = sorted(set(kw) - ok)
    if dropped:
        print("note: TrainingArguments ignores", dropped)
    return {k: v for k, v in kw.items() if k in ok}


def train_one(args, seed, rows):
    import torch
    import transformers
    from torch.utils.data import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
                              set_seed)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        model, tok = smoke_model_and_tokenizer(len(INTENTS))
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype_kw = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=len(INTENTS), problem_type="multi_label_classification",
            **{dtype_kw: torch.float32})
        model.config.pad_token_id = tok.pad_token_id
    if args.eval_adapter:  # evaluate a saved LoRA adapter, no training
        from peft import PeftModel
        _disable_peft_torchao()
        model = PeftModel.from_pretrained(model, args.eval_adapter)
        model = model.to(device)
        if device == "cuda":
            model = model.half()
        return HFRouter(model, tok, device, args.max_len), {"n_train": 0, "train_seconds": 0, "device": device,
                                                            "adapter": args.eval_adapter}, model
    if args.grad_ckpt:  # trade ~30% speed for a large cut in activation memory
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora:
        _disable_peft_torchao()
        from peft import LoraConfig, TaskType, get_peft_model
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r,
                                                 lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=LORA_TARGETS))
        model.print_trainable_parameters()

    keep = (lambda r: r["source"] == "tau_traj") if args.traj_only else (lambda r: True)
    tr = make_train(rows, keep=keep, seed=seed)
    if args.smoke:
        tr = tr[:400]
    idx = {c: i for i, c in enumerate(INTENTS)}

    class DS(Dataset):
        def __init__(self, rs):
            self.rs = rs

        def __len__(self):
            return len(self.rs)

        def __getitem__(self, i):
            r = self.rs[i]
            enc = tok(r["text"], truncation=True, max_length=args.max_len)
            y = np.zeros(len(INTENTS), dtype=np.float32)
            for lab in r["labels"]:
                y[idx[lab]] = 1.0
            enc["labels"] = y
            return enc

    # positive weights against class imbalance (clipped so rare classes don't explode)
    Y = np.zeros(len(INTENTS))
    for r in tr:
        for lab in r["labels"]:
            Y[idx[lab]] += 1
    pos_w = torch.tensor(np.clip((len(tr) - Y) / np.maximum(Y, 1), 1.0, args.max_pos_weight), dtype=torch.float32)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out.logits.float(), labels.float(), pos_weight=pos_w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    from transformers import DataCollatorWithPadding
    steps_per_epoch = max(1, len(tr) // (args.bs * args.grad_accum))
    total_steps = 20 if args.smoke else int(steps_per_epoch * args.epochs)
    targs = TrainingArguments(**compat_kwargs(TrainingArguments, dict(
        output_dir=str(Path(args.out) / f"seed{seed}_ckpt"), per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, num_train_epochs=args.epochs,
        warmup_steps=max(1, int(0.05 * total_steps)), lr_scheduler_type="cosine", logging_steps=50,
        save_strategy="no", report_to=[], fp16=(device == "cuda" and not args.smoke), seed=seed,
        dataloader_num_workers=2, use_cpu=(device == "cpu"), max_steps=(20 if args.smoke else -1))))
    t0 = time.time()
    WTrainer(model=model, args=targs, train_dataset=DS(tr),
             data_collator=DataCollatorWithPadding(tok)).train()
    train_s = time.time() - t0
    if device == "cuda":
        model = model.half()  # fp16 inference: roughly halves latency on T4
    return HFRouter(model.to(device), tok, device, args.max_len), {"n_train": len(tr), "train_seconds": round(train_s),
                                                                    "device": device}, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)       # effective batch 32
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-pos-weight", type=float, default=10.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--extra", nargs="*", default=[], help="extra training jsonl (trajectory turns, paraphrases)")
    ap.add_argument("--out", default="out/finetune")
    ap.add_argument("--save-adapter", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--traj-only", action="store_true", help="train on trajectory turns only (no external/templates)")
    ap.add_argument("--eval-adapter", default=None, help="path to a saved adapter: evaluate only, no training")
    ap.add_argument("--no-groups", dest="groups", action="store_false", help="disable sibling route groups")
    ap.add_argument("--replay", nargs="*", default=None,
                    help="with --replay-only: replay.jsonl files to score (e.g. data/heldout/gpt-4o_retail.replay.jsonl)")
    ap.add_argument("--replay-only", action="store_true",
                    help="only compute per-turn probabilities for --replay files and save them (no metrics, no tuning)")
    ap.add_argument("--tag", default="seed0", help="file prefix for --replay-only outputs")
    args = ap.parse_args()
    # one GPU: Kaggle's "T4 x2" otherwise triggers DataParallel, which piles memory onto GPU 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.smoke:
        args.lora, args.seeds = True, [0]
        args.out = "out/smoke" if not args.eval_adapter else "out/smoke_eval"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    import intent_funnel.router as router_mod
    router_mod.GROUP_EXPANSION = args.groups
    if args.replay_only:  # score held-out replays with a saved adapter; thresholds are applied offline
        assert args.eval_adapter and args.replay, "--replay-only needs --eval-adapter and --replay"
        from intent_funnel.evaluate import replay_probs
        Path(args.out).mkdir(parents=True, exist_ok=True)
        router, info, _ = train_one(args, 0, load())
        for rpath in args.replay:
            reps = load_replays(rpath)
            t0 = time.time()
            rp = replay_probs(router, reps)
            name = Path(rpath).name.replace(".replay.jsonl", "")
            out = Path(args.out) / f"{args.tag}_probs_{name}.npz"
            np.savez_compressed(out, classes=np.array(router.classes), replay=np.vstack(rp),
                                replay_lens=np.array([len(x) for x in rp]))
            print(f"{name}: {sum(len(x) for x in rp)} turns scored in {time.time() - t0:.0f}s -> {out}")
        return
    budget = PromptBudget(ROOT / "data/taubench")
    rows = load(extra=args.extra)
    test = eval_sets(rows, "test")
    replays = load_replays()
    if args.smoke:
        test = {k: v[:60] for k, v in test.items()}
        replays = replays[:30]
    all_res, md = {}, [f"# Fine-tuned router: {args.model}{' + LoRA' if args.lora else ''}\n"]
    for seed in args.seeds:
        router, info, model = train_one(args, seed, rows)
        for variant, m in [("whole", router), ("sent", SentenceUnion(router))]:
            res = evaluate(m, test, budget, replays=replays)
            res.update(info)
            res["latency"] = latency(m, [r["text"] for r in test.get("tau_traj", next(iter(test.values())))], n=100)
            all_res[f"seed{seed}_{variant}"] = res
            md.append(to_markdown(f"seed {seed} — {variant}", res))
            print(f"seed{seed} {variant} |", headline(res))
        if args.save_adapter and not args.eval_adapter:
            model.save_pretrained(Path(args.out) / f"seed{seed}_adapter")
        # raw probabilities for offline re-analysis (thresholds, groups) without the GPU
        from intent_funnel.evaluate import replay_probs
        rp = replay_probs(router, replays)
        np.savez_compressed(Path(args.out) / f"seed{seed}_probs.npz", classes=np.array(router.classes),
                            replay=np.array(np.vstack(rp) if rp else np.zeros((0, len(router.classes)))),
                            replay_lens=np.array([len(x) for x in rp]),
                            **{f"set_{k}": router.predict_proba([r["text"] for r in v]) for k, v in test.items()})
    # mean ± std across seeds of the headline numbers
    summ = {}
    for variant in ["whole", "sent"]:
        key = "replay" if replays else "crossfit"
        vals = [all_res[f"seed{s}_{variant}"][key] for s in args.seeds]
        fields = ["write_coverage", "task_coverage", "step_token", "traj_fell_back"] if replays else \
            ["coverage", "risk", "safe_rate", "token_ratio"]
        summ[variant] = {k: {"mean": float(np.mean([v[k] for v in vals])), "std": float(np.std([v[k] for v in vals]))}
                         for k in fields}
    all_res["summary_over_seeds"] = summ
    md.insert(1, "## Summary over seeds (primary metric, cross-fit)\n\n```\n" + json.dumps(summ, indent=1) + "\n```\n")
    (Path(args.out) / "results.json").write_text(json.dumps(all_res, indent=1))
    (Path(args.out) / "results.md").write_text("\n".join(md))
    print("wrote", Path(args.out) / "results.md")


if __name__ == "__main__":
    main()
