#!/usr/bin/env python3
"""
LOCAL GPU runner - evaluate open-weights instruct models as triage judges.

Writes to the SAME results_raw.csv schema as run_and_score.py, and reuses its
prompt/perturbation logic, so scoring (run_and_score.py --metrics-only) and
plotting work unchanged. Free: only uses your GPU.

Models that fit a modest GPU (7B-14B, use --load-4bit for the bigger ones):
  python run_local.py --model Qwen/Qwen2.5-7B-Instruct --repeats 5
  python run_local.py --model meta-llama/Llama-3.1-8B-Instruct --repeats 5
  python run_local.py --model mistralai/Mistral-7B-Instruct-v0.3 --repeats 5
  python run_local.py --model google/gemma-2-9b-it --load-4bit --repeats 5
  python run_local.py --model microsoft/phi-4 --load-4bit --limit 150 --repeats 3

Then score everything (API + local together):
  python run_and_score.py --metrics-only
"""
import argparse, csv, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from run_and_score import build_prompt, parse_verdict, CONDITIONS, SYSTEM, load_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HuggingFace model id")
    ap.add_argument("--dataset", default="dataset.csv")
    ap.add_argument("--out", default="results_raw.csv")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true", help="allow slow CPU inference if CUDA is unavailable")
    ap.add_argument("--deepseek-r1", action="store_true",
                    help="prefill a closed <think> block so DeepSeek-R1 distills answer after </think>")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()
    use_deepseek_r1 = args.deepseek_r1 or "deepseek-r1" in args.model.lower()
    system_in_user = "gemma" in args.model.lower()

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}", flush=True)
    elif not args.allow_cpu:
        raise SystemExit("CUDA is not available to this Python process. Re-run from a GPU-enabled shell, or pass --allow-cpu for a very slow CPU test.")
    else:
        print("WARNING: CUDA unavailable; running on CPU.", flush=True)

    rows = load_rows(args.dataset)
    if args.limit:
        rows = rows[:args.limit]
    paraphrases = json.load(open("paraphrases.json")) if os.path.exists("paraphrases.json") else {}
    conditions = CONDITIONS + (["paraphrase"] if paraphrases else [])

    print(f"Loading {args.model} ...", flush=True)
    if use_deepseek_r1:
        print("DeepSeek-R1 mode: prefill <think></think> and parse only the final answer.", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    done = set()
    if os.path.exists(args.out):
        for r in load_rows(args.out):
            done.add((r["cve_id"], r["model"], r["condition"], r["repeat"]))
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="", encoding="utf-8")
    fields = ["cve_id", "stratum", "asset", "gold_label", "model", "condition",
              "repeat", "parsed_verdict", "parse_ok", "raw_output"]
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    @torch.no_grad()
    def gen(prompt):
        if system_in_user:
            msgs = [{"role": "user", "content": f"{SYSTEM}\n\n{prompt}"}]
        else:
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        if use_deepseek_r1:
            rendered = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            rendered += "<think>\n</think>\n\n"
            ids = tok(rendered, return_tensors="pt")
        else:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(ids, "input_ids"):
            inputs = {k: v.to(model.device) for k, v in ids.items()}
            prompt_len = inputs["input_ids"].shape[1]
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-5),
                                 top_p=0.9, pad_token_id=tok.pad_token_id)
        else:
            ids = ids.to(model.device)
            prompt_len = ids.shape[1]
            out = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-5),
                                 top_p=0.9, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True)

    total = len(rows) * len(conditions) * args.repeats
    n = 0
    for row in rows:
        prompts = {c: build_prompt(row, c, paraphrases) for c in conditions}
        for cond in conditions:
            for k in range(args.repeats):
                n += 1
                if (row["cve_id"], args.model, cond, str(k)) in done:
                    continue
                text = gen(prompts[cond])
                v, ok = parse_verdict(text)
                w.writerow({"cve_id": row["cve_id"], "stratum": row["stratum"], "asset": row["asset"],
                            "gold_label": row["gold_label"], "model": args.model, "condition": cond,
                            "repeat": k, "parsed_verdict": v, "parse_ok": ok,
                            "raw_output": (text or "").replace("\n", " ")[:8000]})
                if n % 50 == 0:
                    f.flush(); print(f"  {n}/{total}", flush=True)
    f.close()
    print(f"Done {args.model} -> {args.out}\nNow run: python run_and_score.py --metrics-only")

if __name__ == "__main__":
    main()
