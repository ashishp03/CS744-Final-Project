import os
import subprocess
import sys
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_scheduler
)
from torch.optim import AdamW
from datasets import load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
import nltk

# ────────────────────────────────────────────────────────────────────────────────
# Auto‑install metric dependencies if needed
# ────────────────────────────────────────────────────────────────────────────────
def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

try:
    import evaluate
    import rouge_score
    import sacrebleu
    import nltk
except ImportError:
    _pip_install([
        "evaluate",
        "rouge-score",
        "absl-py",
        "nltk",
        "sacrebleu",
        "sentencepiece"
    ])
    import evaluate

nltk.download("punkt", quiet=True)
os.environ["WANDB_DISABLED"] = "true"

def main(eval_only: bool, checkpoint_dir: str):
    # 0) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load & split dataset
    dataset = load_dataset("ServiceNow-AI/M2Lingual", "seed_data")
    split = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    # 2) Tokenizer & preprocessing
    model_name = checkpoint_dir if eval_only else "google/mt5-small"
    tokenizer  = AutoTokenizer.from_pretrained("google/mt5-small")

    def preprocess(ex):
        turns = ex["conversation"]
        if turns and turns[-1].get("role") == "assistant":
            target_text = turns[-1]["content"]
            input_turns = turns[:-1]
        else:
            target_text = ""
            input_turns = turns

        input_text = " ".join(t.get("content", "") for t in input_turns)
        inputs  = tokenizer(input_text,  padding="max_length", truncation=True, max_length=512)
        targets = tokenizer(target_text, padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = targets.input_ids
        return inputs

    train_tok = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_tok  = eval_ds.map(preprocess,  remove_columns=eval_ds.column_names)
    cols = ["input_ids", "attention_mask", "labels"]
    train_tok.set_format(type="torch", columns=cols)
    eval_tok .set_format(type="torch", columns=cols)

    # 3) DataLoaders
    train_loader = DataLoader(train_tok, shuffle=True,  batch_size=6) if not eval_only else None
    eval_loader  = DataLoader(eval_tok,               batch_size=6)

    # 4) Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    # 5) Training loop
    if not eval_only:
        optimizer    = AdamW(model.parameters(), lr=5e-5)
        num_steps    = len(train_loader) * 3
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_steps
        )
        model.train()
        for epoch in range(3):
            print(f"\n▶️  Starting epoch {epoch}")
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % 200 == 0:
                    print(f"  [epoch {epoch} | step {step}/{len(train_loader)}] loss = {loss.item():.4f}")

            print(f"✅  Finished epoch {epoch}; last loss = {loss.item():.4f}")

        # save checkpoint
        model.save_pretrained(checkpoint_dir)

    # 6) Generation & evaluation
    model.eval()
    rouge = load_metric("rouge")
    bleu  = load_metric("bleu")
    all_preds, all_refs = [], []

    for batch in tqdm(eval_loader, desc="Generating"):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        gen_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        refs  = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        for p, r in zip(preds, refs):
            if r.strip():
                all_preds.append(p)
                all_refs.append(r)

    if all_refs:
        bleu_scores  = bleu.compute(predictions=all_preds, references=all_refs)
        rouge_scores = rouge.compute(predictions=all_preds, references=all_refs, use_stemmer=True)

        print("\n=== Generation Metrics ===")
        print(f"BLEU: {bleu_scores['bleu']:.4f}")
        for name, score in rouge_scores.items():
            print(f"ROUGE-{name.upper()}: {score:.4f}")
    else:
        print("No non-empty references; skipping evaluation.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only",      action="store_true",
                        help="Skip training and only run generation/eval")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs_gen",
                        help="Where to load/save model")
    args = parser.parse_args()
    main(args.eval_only, args.checkpoint_dir)
