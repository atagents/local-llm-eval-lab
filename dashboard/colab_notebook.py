"""Generate a ready-to-run Unsloth fine-tune notebook (.ipynb) for the Fine-tune Loop step 2.

The user downloads this, opens it in Google Colab (free T4 GPU), uploads the `train.jsonl`
exported from the dashboard, runs all cells, and gets a GGUF to load into LM Studio. Every
step has a markdown explanation so a beginner can follow it. Pure stdlib (builds JSON).
"""
import json


def _md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def _code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines]}


def build(base_model="unsloth/Qwen2.5-7B-Instruct", out_name="my-finetune",
          epochs=2, lr=2e-4, max_seq=2048, quant="q4_k_m", train_filename="train.jsonl"):
    """Return the notebook as pretty JSON bytes (a valid .ipynb)."""
    cells = [
        _md(f"# Fine-tune `{base_model}` with Unsloth",
            "",
            "Free Google **Colab T4** notebook. You don't need a GPU at home — Colab lends you one.",
            "",
            "**What happens here (plain English):**",
            "1. Install Unsloth (a fast, low-memory fine-tuning library).",
            "2. Load the base model in 4-bit (so it fits a free GPU).",
            "3. Add a **LoRA** adapter — a small trainable add-on, so we teach new behaviour cheaply.",
            "4. Upload your `train.jsonl` (exported from the eval dashboard) and format it.",
            "5. Train for a few minutes.",
            "6. Export a **GGUF** file and download it — that's the file LM Studio loads.",
            "",
            "**Before you start:** menu **Runtime → Change runtime type → T4 GPU**. Then **Runtime → Run all**."),

        _md("### 1. Install Unsloth"),
        _code("%%capture",
              "!pip install unsloth",
              "!pip install --no-deps --upgrade peft accelerate bitsandbytes",
              '!pip install --force-reinstall --no-deps "trl==0.24.0"   # pin the API this notebook uses'),

        _md("### 2. Load the base model (4-bit, fits a free T4)",
            "If the base model is **gated** on Hugging Face (e.g. Llama), uncomment the login line",
            "and paste your free HF token."),
        _code("from unsloth import FastLanguageModel",
              "import torch",
              "# from huggingface_hub import login; login()   # uncomment for gated base models",
              "",
              f"max_seq_length = {max_seq}",
              "model, tokenizer = FastLanguageModel.from_pretrained(",
              f'    model_name = "{base_model}",',
              "    max_seq_length = max_seq_length,",
              "    dtype = None,",
              "    load_in_4bit = True,",
              ")"),

        _md("### 3. Add the LoRA adapter (the small trainable part)"),
        _code("model = FastLanguageModel.get_peft_model(",
              "    model,",
              "    r = 16,",
              '    target_modules = ["q_proj","k_proj","v_proj","o_proj",',
              '                       "gate_proj","up_proj","down_proj"],',
              "    lora_alpha = 16, lora_dropout = 0, bias = \"none\",",
              '    use_gradient_checkpointing = "unsloth",',
              "    random_state = 3407,",
              ")"),

        _md(f"### 4. Upload your data and format it",
            f"Run the cell, then pick the **`{train_filename}`** you exported from the dashboard",
            "(Fine-tune tab → *Export train as JSONL*). Each line is a chat with user + assistant turns."),
        _code("from google.colab import files",
              "uploaded = files.upload()   # choose your train.jsonl",
              "",
              "from datasets import load_dataset",
              f'dataset = load_dataset("json", data_files="{train_filename}", split="train")',
              "",
              "def fmt(ex):",
              "    msgs = ex[\"messages\"]",
              "    # Gemma has no 'system' role -> fold a leading system turn into the first user turn.",
              "    if msgs and msgs[0][\"role\"] == \"system\":",
              "        sys_txt = msgs[0][\"content\"]; rest = msgs[1:]",
              "        if rest and rest[0][\"role\"] == \"user\":",
              "            rest = [{\"role\": \"user\", \"content\": sys_txt + \"\\n\\n\" + rest[0][\"content\"]}] + rest[1:]",
              "        msgs = rest",
              "    return {\"text\": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}",
              "dataset = dataset.map(fmt)",
              "print(dataset[0][\"text\"][:600])"),

        _md(f"### 5. Train (~a few minutes for a small set, {epochs} epoch(s))"),
        _code("from trl import SFTTrainer, SFTConfig",
              "import torch",
              "",
              "trainer = SFTTrainer(",
              "    model = model,",
              "    train_dataset = dataset,",
              "    processing_class = tokenizer,   # trl >=0.13 renamed `tokenizer`",
              "    args = SFTConfig(",
              '        dataset_text_field = "text",',
              "        max_seq_length = max_seq_length,",
              "        per_device_train_batch_size = 2, gradient_accumulation_steps = 4,",
              f"        warmup_steps = 5, num_train_epochs = {epochs}, learning_rate = {lr},",
              "        logging_steps = 1, optim = \"adamw_8bit\", weight_decay = 0.01,",
              "        lr_scheduler_type = \"linear\", seed = 3407, output_dir = \"outputs\",",
              "        fp16 = not torch.cuda.is_bf16_supported(), bf16 = torch.cuda.is_bf16_supported(),",
              "    ),",
              ")",
              "trainer.train()"),

        _md(f"### 6. Export to GGUF (`{quant}`) and download → load into LM Studio",
            "The download is the model file. In LM Studio: **My Models → Import**, then it appears",
            "in the dashboard's model list as your fine-tuned model for the *after* eval."),
        _code(f'model.save_pretrained_gguf("{out_name}", tokenizer, quantization_method="{quant}")',
              "import os",
              f'for f in os.listdir("{out_name}"):',
              "    if f.endswith(\".gguf\"):",
              f'        print("downloading", f); files.download(f"{out_name}/" + f)'),

        _md("---",
            "Back in the dashboard **🎓 Fine-tune** tab: pick this model under *Eval (after)*, run it,",
            "and step 4 tells you — with statistics — whether the fine-tune **really** helped."),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }
    return json.dumps(nb, indent=1).encode("utf-8")


if __name__ == "__main__":
    data = build()
    nb = json.loads(data)
    assert nb["nbformat"] == 4 and nb["cells"], "bad notebook"
    assert any(c["cell_type"] == "code" and any("FastLanguageModel" in s for s in c["source"]) for c in nb["cells"])
    print(f"colab_notebook OK — {len(nb['cells'])} cells, {len(data)} bytes")
