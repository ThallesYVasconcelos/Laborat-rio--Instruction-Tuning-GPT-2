"""
Experimento: Dataset Pequeno vs Médio
=====================================
Executa o pipeline completo com dataset PEQUENO (200 por dataset = ~400 total)
para comparar com o experimento MÉDIO já realizado (1000 por dataset = ~2000 total).

Uso:
    python run_experiment_small.py              # pipeline completo
    python run_experiment_small.py --skip-train # pular treino (usar modelo existente)

Saída:
    - data_small/instruction_*.jsonl  (dataset pequeno)
    - gpt2-medium-sft-small.pth       (modelo treinado no dataset pequeno)
    - logs/experiment_small.txt       (resultados e métricas)

Comparação: Ver RELATORIO.md seção 5.6
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório ao path para imports
sys.path.insert(0, str(Path(__file__).parent))

from build_instruction_dataset import main as build_main
from evaluate_llm_judge import main as evaluate_main, compute_score_stats, print_results


def run_small_experiment(n_test: int = 20, skip_training: bool = False):
    """Executa pipeline completo com dataset pequeno."""
    print("=" * 70)
    print("EXPERIMENTO: Dataset PEQUENO (200 AG News + 200 SST-2)")
    print("=" * 70)

    # 1. Construir dataset pequeno (a menos que já exista e skip_training)
    if not skip_training or not Path("data_small/instruction_dataset.json").exists():
        print("\n[1/2] Construindo dataset pequeno (data_small/)...")
        build_main(max_per_dataset=200, output_dir="data_small", seed=42)
    else:
        print("\n[1/2] Dataset data_small/ já existe, pulando construção.")

    # 2. Fine-tune e avaliar
    print("\n[2/2] Fine-tuning e avaliação LLM-as-a-Judge...")
    results = evaluate_main(
        data_dir="data_small",
        model_path="gpt2-medium-sft-small.pth",
        skip_training=skip_training,
        n_test=n_test,
    )

    # 3. Salvar resultados
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    out_path = logs_dir / "experiment_small.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== EXPERIMENTO DATASET PEQUENO ===\n\n")
        f.write("Config: max_per_dataset=200, output_dir=data_small\n\n")
        f.write("--- Resultados LLM-as-a-Judge ---\n")
        wins_b = sum(1 for r in results if r["winner"] == "B")
        wins_a = sum(1 for r in results if r["winner"] == "A")
        ties = sum(1 for r in results if r["winner"] == "EMPATE")
        n = len(results)
        f.write(f"Vitórias fine-tuned (B): {wins_b} ({100*wins_b/n:.1f}%)\n")
        f.write(f"Vitórias base (A): {wins_a} ({100*wins_a/n:.1f}%)\n")
        f.write(f"Empates: {ties} ({100*ties/n:.1f}%)\n\n")
        score_stats = compute_score_stats(results)
        f.write("Diferença média de scores (B - A):\n")
        for key, (diff, count) in score_stats.items():
            label = {"correcao_factual": "Correção factual", "aderencia": "Aderência", "clareza": "Clareza"}[key]
            if diff is not None:
                f.write(f"  {label}: {diff:+.2f} (n={count})\n")
            else:
                f.write(f"  {label}: (não parseado)\n")

    print(f"\nResultados salvos em: {out_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true", help="Pular treino (usar modelo existente)")
    parser.add_argument("--n-test", type=int, default=20, help="Número de exemplos para avaliação")
    args = parser.parse_args()
    run_small_experiment(n_test=args.n_test, skip_training=args.skip_train)
