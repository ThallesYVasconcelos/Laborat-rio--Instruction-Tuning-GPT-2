"""
Script de Fine-tuning e Avaliação com LLM-as-a-Judge
====================================================
Pipeline: Carregar dataset → Fine-tune GPT-2-medium → Gerar respostas (base vs fine-tuned) → Avaliar com LLM juíza
Requer: data/instruction_dataset.json (ou rodar build_instruction_dataset.py antes)
Opcional: Ollama rodando (ollama serve + ollama run llama3) para LLM-as-a-Judge
"""

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import partial

import torch
import requests
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# ============== Carregamento do Dataset ==============

def load_instruction_dataset(data_dir: str = "data") -> tuple:
    """Carrega train, val e test do dataset de instruções."""
    path = Path(data_dir) / "instruction_dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {path}. Execute build_instruction_dataset.py primeiro."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["train"], data["val"], data["test"]


# ============== Formato e Dataset para Treino ==============

def format_input(entry: Dict) -> str:
    """Formato Alpaca: instrução + input (sem resposta). Usado para inferência."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    """Dataset que pré-tokeniza textos no formato Alpaca."""

    def __init__(self, data: List[Dict], tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            ids = tokenizer.encode(full_text, add_special_tokens=False)
            self.encoded_texts.append(ids)

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """Agrupa batch com padding, targets shift+1, e ignore_index para padding."""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = list(item) + [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


# ============== Fine-tuning ==============

def calc_loss_loader(loader, model, device, num_batches=None):
    """Calcula loss média no loader."""
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            if num_batches and i >= num_batches:
                break
            logits = model(inputs).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            n += 1
    return total_loss / n if n else 0


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    tokenizer,
    device,
    batch_size: int = 8,
    lr: float = 5e-5,
    weight_decay: float = 0.1,
    num_epochs: int = 2,
    model_save_path: str = "gpt2-medium-sft.pth",
) -> torch.nn.Module:
    """Fine-tune GPT-2-medium no dataset de instruções."""
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.to(device)

    customized_collate = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate,
        shuffle=True,
        drop_last=True
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate,
        shuffle=False,
        drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    eval_freq = 5

    print(f"Treinando: {len(train_data)} exemplos, {num_epochs} épocas")
    torch.manual_seed(123)
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        for step, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(inputs).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100
            )
            loss.backward()
            optimizer.step()

            if step % eval_freq == 0:
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
                print(f"Ep {epoch+1} (Step {step:05d}): Train loss {loss.item():.3f}, Val loss {val_loss:.3f}")

    print(f"Treino concluído em {(time.time() - start) / 60:.2f} min")
    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo salvo: {model_save_path}")
    return model


# ============== Geração de Respostas ==============

def generate_response(
    model,
    entry: Dict,
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> str:
    """Gera resposta dado instruction+input."""
    prompt = format_input(entry)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=temperature,
            do_sample=True
        )
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    response = full[len(prompt):].replace("### Response:", "").strip()
    if "<|endoftext|>" in response:
        response = response.split("<|endoftext|>")[0].strip()
    return response


# ============== LLM-as-a-Judge ==============

def query_ollama(
    prompt: str,
    model: str = "llama3",
    url: str = "http://localhost:11434/api/chat"
) -> str:
    """Chama Ollama (Llama 3) como juiz. Requer ollama serve rodando."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"seed": 123, "temperature": 0}
    }
    try:
        with requests.post(url, json=data, stream=True, timeout=60) as r:
            r.raise_for_status()
            out = ""
            for line in r.iter_lines(decode_unicode=True):
                if line:
                    j = json.loads(line)
                    if "message" in j:
                        out += j["message"]["content"]
            return out
    except Exception as e:
        return f"[ERRO Ollama: {e}. Instale ollama e rode 'ollama serve' + 'ollama run llama3']"


JUDGE_PROMPT = """Você é um juiz imparcial. Avalie as duas respostas (A e B) para a mesma instrução e entrada.

Instrução: {instruction}
Entrada: {input_text}

Resposta A (modelo base): {response_a}
Resposta B (modelo fine-tuned): {response_b}

Resposta esperada (referência): {reference}

Forneça:
1. Correção factual (0-5): A e B
2. Aderência à instrução (0-5): A e B
3. Clareza/utilidade (0-5): A e B
4. Vencedor: A, B ou EMPATE
5. Justificativa breve

Formato:
Correção factual - A: X, B: Y
Aderência - A: X, B: Y
Clareza - A: X, B: Y
Vencedor: A/B/EMPATE
Justificativa: ..."""


def parse_winner(judge_output: str) -> str:
    """Extrai vencedor do output do juiz."""
    if "Vencedor: A" in judge_output or "vencedor: A" in judge_output.lower():
        return "A"
    if "Vencedor: B" in judge_output or "vencedor: B" in judge_output.lower():
        return "B"
    return "EMPATE"


def parse_scores(judge_output: str) -> Dict[str, Tuple[float, float]]:
    """
    Extrai scores (A, B) do output do juiz.
    Retorna dict com keys: correcao_factual, aderencia, clareza.
    Cada valor é (score_A, score_B). Retorna None para scores não parseados.
    """
    result = {}
    # Padrões: "Correção factual - A: X, B: Y" ou "Correção factual - A:X, B:Y"
    patterns = {
        "correcao_factual": r"[Cc]orre[cç][ãa]o\s*factual\s*[-–:]\s*A\s*:?\s*(\d+(?:\.\d+)?)\s*,\s*B\s*:?\s*(\d+(?:\.\d+)?)",
        "aderencia": r"[Aa]der[eê]ncia\s*[-–:]\s*A\s*:?\s*(\d+(?:\.\d+)?)\s*,\s*B\s*:?\s*(\d+(?:\.\d+)?)",
        "clareza": r"[Cc]lareza\s*[-–:]\s*A\s*:?\s*(\d+(?:\.\d+)?)\s*,\s*B\s*:?\s*(\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, judge_output, re.IGNORECASE)
        if match:
            try:
                a_val = float(match.group(1))
                b_val = float(match.group(2))
                result[key] = (a_val, b_val)
            except (ValueError, IndexError):
                pass
    return result


# ============== Pipeline de Avaliação ==============

def run_evaluation(
    test_data: List[Dict],
    model_base: torch.nn.Module,
    model_finetuned: torch.nn.Module,
    tokenizer,
    device,
    n_test: Optional[int] = None,
    use_ollama: bool = True,
) -> List[Dict]:
    """
    Gera respostas para base e fine-tuned, avalia com LLM-as-a-Judge.
    Retorna lista de resultados com winner, judge_output, etc.
    """
    n_test = n_test or min(50, len(test_data))
    torch.manual_seed(42)

    test_with_responses = []
    for i in tqdm(range(n_test), desc="Gerando respostas"):
        entry = test_data[i].copy()
        entry["response_base"] = generate_response(model_base, entry, tokenizer)
        entry["response_finetuned"] = generate_response(model_finetuned, entry, tokenizer)
        test_with_responses.append(entry)

    results = []
    for entry in tqdm(test_with_responses, desc="LLM-as-a-Judge"):
        prompt = JUDGE_PROMPT.format(
            instruction=entry["instruction"],
            input_text=entry.get("input", "(vazio)"),
            response_a=entry["response_base"],
            response_b=entry["response_finetuned"],
            reference=entry["output"]
        )
        judge_out = query_ollama(prompt) if use_ollama else "[Ollama não disponível]"
        winner = parse_winner(judge_out)
        scores = parse_scores(judge_out)
        results.append({
            "entry": entry,
            "judge_output": judge_out,
            "winner": winner,
            "scores": scores
        })

    return results


def compute_score_stats(results: List[Dict]) -> Dict:
    """Calcula diferença média de scores (B - A) por critério."""
    stats = {"correcao_factual": [], "aderencia": [], "clareza": []}
    for r in results:
        scores = r.get("scores", {})
        for key in stats:
            if key in scores:
                a_val, b_val = scores[key]
                stats[key].append(b_val - a_val)  # positivo = fine-tuned melhor
    return {
        k: (sum(v) / len(v), len(v)) if v else (None, 0)
        for k, v in stats.items()
    }


def print_results(results: List[Dict]):
    """Imprime resumo dos resultados LLM-as-a-Judge."""
    wins_finetuned = sum(1 for r in results if r["winner"] == "B")
    wins_base = sum(1 for r in results if r["winner"] == "A")
    ties = sum(1 for r in results if r["winner"] == "EMPATE")
    n = len(results)

    print(f"\n=== Resultados LLM-as-a-Judge (n={n}) ===")
    print(f"Vitórias fine-tuned (B): {wins_finetuned} ({100 * wins_finetuned / n:.1f}%)")
    print(f"Vitórias base (A): {wins_base} ({100 * wins_base / n:.1f}%)")
    print(f"Empates: {ties} ({100 * ties / n:.1f}%)")

    # Diferença média de scores (B - A; positivo = fine-tuned melhor)
    score_stats = compute_score_stats(results)
    print("\n--- Diferença média de scores (B - A, positivo = fine-tuned melhor) ---")
    for key, (diff, count) in score_stats.items():
        label = {"correcao_factual": "Correção factual", "aderencia": "Aderência", "clareza": "Clareza"}[key]
        if diff is not None:
            print(f"  {label}: {diff:+.2f} (n={count})")
        else:
            print(f"  {label}: (não parseado)")

    print("\n--- Exemplo onde fine-tuned venceu ---")
    for r in results:
        if r["winner"] == "B":
            e = r["entry"]
            print("Instrução:", e["instruction"][:80], "...")
            print("Base:", e["response_base"][:100])
            print("Fine-tuned:", e["response_finetuned"][:100])
            print("Esperado:", e["output"])
            break

    print("\n--- Exemplo onde base venceu ---")
    for r in results:
        if r["winner"] == "A":
            e = r["entry"]
            print("Instrução:", e["instruction"][:80], "...")
            print("Base:", e["response_base"][:100])
            print("Fine-tuned:", e["response_finetuned"][:100])
            break


# ============== Main ==============

def main(
    data_dir: str = "data",
    model_path: str = "gpt2-medium-sft.pth",
    skip_training: bool = False,
    n_test: int = 20,
):
    """
    Pipeline completo: carrega dados, treina (ou carrega modelo), avalia com LLM-as-a-Judge.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    train_data, val_data, test_data = load_instruction_dataset(data_dir)
    print(f"Dataset: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    if skip_training and Path(model_path).exists():
        print("Carregando modelo fine-tuned existente...")
        model_finetuned = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        model_finetuned.load_state_dict(torch.load(model_path, map_location=device))
        model_finetuned.to(device)
        model_finetuned.eval()
    else:
        print("Iniciando fine-tuning...")
        model_finetuned = train_model(
            train_data, val_data, tokenizer, device,
            batch_size=8, lr=5e-5, weight_decay=0.1, num_epochs=2,
            model_save_path=model_path
        )
        model_finetuned.eval()

    model_base = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model_base.to(device)
    model_base.eval()

    results = run_evaluation(
        test_data, model_base, model_finetuned, tokenizer, device,
        n_test=n_test, use_ollama=True
    )

    print_results(results)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 e avaliar com LLM-as-a-Judge")
    parser.add_argument("--skip-train", action="store_true", help="Pular treino (usar modelo existente, só avaliação)")
    parser.add_argument("--n-test", type=int, default=20, help="Número de exemplos para avaliação (default: 20)")
    parser.add_argument("--data-dir", type=str, default="data", help="Diretório do dataset (default: data)")
    parser.add_argument("--model", type=str, default="gpt2-medium-sft.pth", help="Caminho do modelo fine-tuned")
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        model_path=args.model,
        skip_training=args.skip_train,
        n_test=args.n_test,
    )
