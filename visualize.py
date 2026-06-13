from __future__ import annotations

import argparse

import torch

from moodtox.visualization import visualize_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained molecular model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--output-dir", default="visualization")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    summary = visualize_checkpoint(
        args.checkpoint, args.smiles, args.output_dir, args.device
    )
    print(f"probability={summary['probability']:.6f}")


if __name__ == "__main__":
    main()
