"""Compute per-output CDTC scores; no inferential statistics or plots."""
import argparse
from pathlib import Path
from src.cdtc import compute_cdtc_for_all
from src.io_utils import load_config, load_expert_scores, load_trajectories

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/cdtc'))
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config.yaml')
    args=parser.parse_args()
    config=load_config(args.config)
    expert=load_trajectories(args.input_dir/'expert_trajectories.jsonl',is_model=False)
    model=load_trajectories(args.input_dir/'model_trajectories.jsonl',is_model=True)
    scores=load_expert_scores(args.input_dir/'expert_scores.csv')
    result=compute_cdtc_for_all(model,expert,scores,config=config)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output_dir/'cdtc_results.csv',index=False)
    print('Per-output scores saved. No inferential analysis was run.')

if __name__=='__main__':main()
