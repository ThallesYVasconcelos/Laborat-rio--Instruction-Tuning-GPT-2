"""
Script de Construção do Dataset de Instruções
============================================
Pipeline: Carregar dados → Converter para instruções → Variações → Curadoria → Normalização → Split 80/10/10
Saída: data/instruction_train.jsonl, data/instruction_val.jsonl, data/instruction_test.jsonl, data/instruction_dataset.json
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional

from datasets import load_dataset
from tqdm import tqdm


# ============== Carregamento de Dados ==============

def load_ag_news(max_samples: Optional[int] = None) -> List[Dict]:
    """Carrega AG News (classificação de notícias em 4 categorias)."""
    ds = load_dataset("ag_news", split="train")
    labels = ["World", "Sports", "Business", "Sci/Tech"]

    data = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        data.append({
            "text": ex["text"],
            "label": labels[ex["label"]],
            "label_id": ex["label"]
        })
    return data


def load_sst2(max_samples: Optional[int] = None) -> List[Dict]:
    """Carrega SST-2 (classificação de sentimento)."""
    ds = load_dataset("glue", "sst2", split="train")
    labels = ["negative", "positive"]

    data = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        data.append({
            "text": ex["sentence"],
            "label": labels[ex["label"]],
            "label_id": ex["label"]
        })
    return data


def convert_classification_to_instruction_format(
    data: List[Dict],
    task: str = "ag_news"
) -> List[Dict]:
    """Converte dataset de classificação para formato instrução/entrada/resposta."""
    if task == "ag_news":
        labels = "World, Sports, Business, Sci/Tech"
        instruction = f"Classifique o texto nas categorias: {labels}."
    elif task == "sst2":
        labels = "negative, positive"
        instruction = f"Classifique o sentimento do texto nas categorias: {labels}."
    else:
        labels = "classe1, classe2"
        instruction = f"Classifique o texto nas categorias: {labels}."

    result = []
    for item in data:
        result.append({
            "instruction": instruction,
            "input": item["text"],
            "output": item["label"],
            "source": task
        })
    return result


def load_and_combine_ag_news_sst2(
    max_per_dataset: Optional[int] = None
) -> List[Dict]:
    """Carrega AG News e SST-2, converte para instruções e combina."""
    ag_raw = load_ag_news(max_samples=max_per_dataset)
    sst_raw = load_sst2(max_samples=max_per_dataset)

    ag_data = convert_classification_to_instruction_format(ag_raw, task="ag_news")
    sst_data = convert_classification_to_instruction_format(sst_raw, task="sst2")

    combined = ag_data + sst_data
    random.shuffle(combined)
    return combined


# ============== Variações de Instrução ==============

INSTRUCTION_GENERATION_PROMPT = """
Dado o seguinte exemplo de tarefa de NLP, gere 3 variações diferentes e realistas
da instrução que um usuário poderia dar. Mantenha o mesmo objetivo da tarefa.
Retorne APENAS as 3 instruções, uma por linha, sem numeração.

Exemplo original: {instruction}
Contexto: classificação de texto
"""


def generate_instruction_variations_with_llm(
    data: List[Dict],
    api_func,
    sample_ratio: float = 0.3,
    seed: int = 42
) -> List[Dict]:
    """Usa LLM para gerar variações de instruções em uma amostra."""
    random.seed(seed)
    n_sample = max(1, int(len(data) * sample_ratio))
    indices = random.sample(range(len(data)), min(n_sample, len(data)))

    instruction_pool = {}
    for idx in tqdm(indices, desc="Gerando variações de instrução"):
        entry = data[idx]
        inst = entry["instruction"]
        if inst not in instruction_pool:
            prompt = INSTRUCTION_GENERATION_PROMPT.format(instruction=inst)
            try:
                variations = api_func(prompt).strip().split("\n")
                variations = [v.strip().lstrip("0123456789.-) ") for v in variations if v.strip()]
                instruction_pool[inst] = variations[:3] if variations else [inst]
            except Exception:
                instruction_pool[inst] = [inst]

    result = []
    for i, entry in enumerate(data):
        new_entry = entry.copy()
        if i in indices and entry["instruction"] in instruction_pool:
            pool = instruction_pool[entry["instruction"]]
            new_entry["instruction"] = random.choice(pool) if pool else entry["instruction"]
        result.append(new_entry)

    return result


def generate_instruction_variations_static(data: List[Dict]) -> List[Dict]:
    """Alternativa SEM API: usa variações pré-definidas (não precisa de LLM externo)."""
    variations = {
        "Classifique o texto nas categorias: World, Sports, Business, Sci/Tech.": [
            "Qual a categoria desta notícia? Opções: World, Sports, Business, Sci/Tech.",
            "Atribua uma das categorias ao texto: World, Sports, Business ou Sci/Tech.",
            "Identifique se o texto é sobre World, Sports, Business ou Sci/Tech.",
        ],
        "Classifique o sentimento do texto nas categorias: negative, positive.": [
            "O texto é positivo ou negativo?",
            "Qual o sentimento expresso: negative ou positive?",
            "Identifique a polaridade do texto: negative ou positive.",
        ]
    }

    result = []
    for entry in data:
        inst = entry["instruction"]
        if inst in variations:
            new_entry = entry.copy()
            new_entry["instruction"] = random.choice(variations[inst])
            result.append(new_entry)
        else:
            result.append(entry.copy())

    random.shuffle(result)
    return result


# ============== Filtragem e Curadoria ==============

def filter_vague_instructions(data: List[Dict]) -> List[Dict]:
    """Remove instruções vagas (muito curtas ou genéricas)."""
    vague_patterns = [
        r"^[Ff]aça\s+algo\.?$",
        r"^[Rr]esponda\.?$",
        r"^[Aa]nalise\.?$",
        r"^.{1,15}$",
    ]
    compiled = [re.compile(p) for p in vague_patterns]

    filtered = []
    for entry in data:
        inst = entry.get("instruction", "")
        if any(c.search(inst) for c in compiled):
            continue
        filtered.append(entry)
    return filtered


def filter_label_leakage(data: List[Dict], tasks: List[str] = None) -> List[Dict]:
    """Remove exemplos onde o rótulo aparece no input (vazamento)."""
    tasks = tasks or ["ag_news", "sst2"]
    filtered = []
    for entry in data:
        output = entry.get("output", "").strip()
        input_text = entry.get("input", "").lower()
        if output.lower() in input_text and len(output) > 3:
            continue
        filtered.append(entry)
    return filtered


def find_duplicate_indices(data: List[Dict], key: str = "input", threshold: float = 0.95) -> set:
    """Encontra índices de duplicatas/near-duplicatas por similaridade Jaccard."""
    def tokenize(s: str) -> set:
        return set(re.findall(r"\w+", s.lower()))

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    to_remove = set()
    texts = [(i, tokenize(entry.get(key, ""))) for i, entry in enumerate(data)]

    for i in range(len(texts)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(texts)):
            if j in to_remove:
                continue
            sim = jaccard(texts[i][1], texts[j][1])
            if sim >= threshold:
                to_remove.add(j)

    return to_remove


def filter_duplicates(data: List[Dict], keys: List[str] = None, threshold: float = 0.9) -> List[Dict]:
    """Remove duplicatas e near-duplicatas."""
    keys = keys or ["instruction", "input"]
    all_to_remove = set()

    for key in keys:
        to_remove = find_duplicate_indices(data, key=key, threshold=threshold)
        all_to_remove.update(to_remove)

    return [entry for i, entry in enumerate(data) if i not in all_to_remove]


VALID_OUTPUTS_AG_NEWS = {"World", "Sports", "Business", "Sci/Tech"}
VALID_OUTPUTS_SST2 = {"negative", "positive"}
VALID_OUTPUTS_COMBINED = VALID_OUTPUTS_AG_NEWS | VALID_OUTPUTS_SST2


def filter_inconsistent(data: List[Dict], tasks: List[str] = None) -> List[Dict]:
    """Remove exemplos inconsistentes (output fora das opções válidas)."""
    tasks = tasks or ["ag_news", "sst2"]
    valid = VALID_OUTPUTS_COMBINED
    return [e for e in data if e.get("output", "").strip() in valid]


def apply_curation_pipeline(data: List[Dict], tasks: List[str] = None) -> List[Dict]:
    """Aplica toda a curadoria e filtragem."""
    tasks = tasks or ["ag_news", "sst2"]
    data = filter_vague_instructions(data)
    data = filter_label_leakage(data, tasks)
    data = filter_inconsistent(data, tasks)
    data = filter_duplicates(data, threshold=0.92)
    return data


# ============== Normalização ==============

def normalize_entry(entry: Dict) -> Dict:
    """Padroniza formato: instruction, input, output (strings limpas)."""
    out = {
        "instruction": str(entry.get("instruction", "")).strip(),
        "input": str(entry.get("input", "")).strip(),
        "output": str(entry.get("output", "")).strip()
    }
    if "source" in entry:
        out["source"] = entry["source"]
    return out


def normalize_dataset(data: List[Dict]) -> List[Dict]:
    """Normaliza todo o dataset para formato padrão."""
    normalized = []
    for entry in data:
        n = normalize_entry(entry)
        if n["instruction"] and n["output"]:
            normalized.append(n)
    return normalized


def to_alpaca_format(entry: Dict) -> str:
    """Converte para formato Alpaca (texto linear para treino)."""
    parts = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "",
        "### Instruction:",
        entry["instruction"],
        "",
        "### Input:",
        entry["input"] if entry["input"] else "(empty)",
        "",
        "### Response:",
        entry["output"]
    ]
    return "\n".join(parts)


# ============== Split ==============

def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> tuple:
    """Split: 80% treino, 10% validação, 10% teste."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.seed(seed)
    data = data.copy()
    random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    return train_data, val_data, test_data


# ============== Pipeline Principal ==============

def main(max_per_dataset: int = 1000, output_dir: str = "data", seed: int = 42):
    """
    Executa o pipeline completo de construção do dataset.
    Retorna (train_data, val_data, test_data).
    """
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SEÇÃO 1: Carregando datasets (AG News + SST-2)")
    print("=" * 60)
    data = load_and_combine_ag_news_sst2(max_per_dataset=max_per_dataset)
    n_ag = sum(1 for e in data if e.get("source") == "ag_news")
    n_sst = sum(1 for e in data if e.get("source") == "sst2")
    print(f"AG News: {n_ag} | SST-2: {n_sst} | Total: {len(data)}")

    print("\n" + "=" * 60)
    print("SEÇÃO 2: Variações de instrução (estático, sem API)")
    print("=" * 60)
    data = generate_instruction_variations_static(data)
    print("Exemplos após variação:")
    for e in data[:2]:
        print(json.dumps(e, indent=2, ensure_ascii=False))
        print("---")

    print("\n" + "=" * 60)
    print("SEÇÃO 3: Curadoria e filtragem")
    print("=" * 60)
    before = len(data)
    data = apply_curation_pipeline(data, tasks=["ag_news", "sst2"])
    print(f"Antes: {before} | Depois: {len(data)}")

    print("\n" + "=" * 60)
    print("SEÇÃO 4: Normalização")
    print("=" * 60)
    data = normalize_dataset(data)
    print(f"Exemplos normalizados: {len(data)}")

    print("\n" + "=" * 60)
    print("SEÇÃO 5: Split (80/10/10)")
    print("=" * 60)
    train, val, test = split_dataset(data, seed=seed)
    print(f"Treino: {len(train)} | Val: {len(val)} | Teste: {len(test)}")

    # Salvar
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        path = OUTPUT_DIR / f"instruction_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for entry in subset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Salvo: {path}")

    full_path = OUTPUT_DIR / "instruction_dataset.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2, ensure_ascii=False)
    print(f"Salvo: {full_path}")

    print("\nPipeline concluído.")
    return train, val, test


if __name__ == "__main__":
    main(max_per_dataset=1000)
