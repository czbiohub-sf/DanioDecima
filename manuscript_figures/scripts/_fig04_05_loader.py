"""Shared data loader for Fig 4 and Fig 5.

Both figures consume the same `aggregated_data` dict produced by
`analyze_timepoint_enrichment_all_models()` in
`decima-applications-main/notebooks/4_evaluation/01_evaluate_celltypes.ipynb`
(L3334). Each h5ad is ~240 MB but only `.obs` is read.

Run under the gReLu env with `LD_LIBRARY_PATH` prefixed:
    LD_LIBRARY_PATH=/hpc/mydata/yang-joon.kim/.conda/envs/gReLu/lib:$LD_LIBRARY_PATH \\
        /hpc/mydata/yang-joon.kim/.conda/envs/gReLu/bin/python ...
"""

import scanpy as sc
import pandas as pd

BASE = "/hpc/scratch/group.data.science/mathias.voges/zebrahub-decima/experiments"
EXP1 = f"{BASE}/decima_experiments_20250617_225114"
EXP2 = f"{BASE}/decima_experiments_20250618_111138"

MODEL_PATHS = [
    f"{EXP1}/random_lr3e-06_seed42/version_0/data_out_decima_random_lr3e-06_seed42_20250619_Random_0.h5ad",
    f"{EXP1}/random_lr3e-06_seed43/version_0/data_out_decima_random_lr3e-06_seed43_20250619_Random_1.h5ad",
    f"{EXP1}/random_lr3e-06_seed44/version_0/data_out_decima_random_lr3e-06_seed44_20250619_Random_2.h5ad",
    f"{EXP1}/random_lr3e-06_seed45/version_0/data_out_decima_random_lr3e-06_seed45_20250619_Random_3.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep0_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep0_lr3e-05_seed42_20250619_Human_Borzoi_0.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep1_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep1_lr3e-05_seed42_20250619_Human_Borzoi_1.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep2_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep2_lr3e-05_seed42_20250619_Human_Borzoi_2.h5ad",
    f"{EXP1}/pretrained_wandb-human_rep3_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-human_rep3_lr3e-05_seed42_20250619_Human_Borzoi_3.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep0_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep0_lr3e-05_seed42_20250619_Mouse_Borzoi_0.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep1_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep1_lr3e-05_seed42_20250619_Mouse_Borzoi_1.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep2_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep2_lr3e-05_seed42_20250619_Mouse_Borzoi_2.h5ad",
    f"{EXP1}/pretrained_wandb-mouse_rep3_lr3e-05_seed42/version_0/data_out_decima_pretrained_wandb-mouse_rep3_lr3e-05_seed42_20250619_Mouse_Borzoi_3.h5ad",
    f"{EXP2}/pretrained_decima-human_rep0_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep0_lr3e-05_seed42_20250619_Human_Decima_0.h5ad",
    f"{EXP2}/pretrained_decima-human_rep1_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep1_lr3e-05_seed42_20250619_Human_Decima_1.h5ad",
    f"{EXP2}/pretrained_decima-human_rep2_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep2_lr3e-05_seed42_20250619_Human_Decima_2.h5ad",
    f"{EXP2}/pretrained_decima-human_rep3_lr3e-05_seed42/version_0/data_out_decima_pretrained_decima-human_rep3_lr3e-05_seed42_20250619_Human_Decima_3.h5ad",
]

MODEL_NAMES = [
    "Random_0", "Random_1", "Random_2", "Random_3",
    "Human_Borzoi_0", "Human_Borzoi_1", "Human_Borzoi_2", "Human_Borzoi_3",
    "Mouse_Borzoi_0", "Mouse_Borzoi_1", "Mouse_Borzoi_2", "Mouse_Borzoi_3",
    "Human_Decima_0", "Human_Decima_1", "Human_Decima_2", "Human_Decima_3",
]

MODEL_TYPE_MAPPING = {
    "Random": "Random Init",
    "Human_Borzoi": "Human-Borzoi",
    "Mouse_Borzoi": "Mouse-Borzoi",
    "Human_Decima": "Human-Decima",
}

MODEL_ORDER = ["Random Init", "Mouse-Borzoi", "Human-Borzoi", "Human-Decima"]

MODEL_COLORS = {
    "Random Init":  "#d62728",
    "Mouse-Borzoi": "#ff7f0e",
    "Human-Borzoi": "#2ca02c",
    "Human-Decima": "#1f77b4",
}

TIMEPOINT_ORDER = [
    "10hpf", "12hpf", "14hpf", "16hpf", "19hpf",
    "24hpf", "2dpf", "3dpf", "5dpf", "10dpf",
]

DEV_STAGES = {
    "Early (10-16hpf)": ["10hpf", "12hpf", "14hpf", "16hpf"],
    "Mid (19hpf-24hpf)": ["19hpf", "24hpf"],
    "Late (2dpf+)": ["2dpf", "3dpf", "5dpf", "10dpf"],
}


def aggregate_by_model_type(n_performers=50):
    """Load 16 h5ad files, group by model type, return dict ready for Fig 4/5."""
    model_groups = {model_type: [] for model_type in MODEL_ORDER}
    for path, name in zip(MODEL_PATHS, MODEL_NAMES):
        model_type = None
        for key, value in MODEL_TYPE_MAPPING.items():
            if key in name:
                model_type = value
                break
        if model_type is None:
            continue
        ad = sc.read_h5ad(path)
        model_groups[model_type].append((ad, name))
        print(f"Loaded {name} -> {model_type}: {ad.n_obs} pseudobulks")

    aggregated_data = {}
    for model_type, replicate_list in model_groups.items():
        if not replicate_list:
            continue
        obs_list = []
        for ad, name in replicate_list:
            df_rep = ad.obs.copy().dropna(subset=["test_pearson"])
            df_rep["replicate"] = name
            df_rep["model_type"] = model_type
            obs_list.append(df_rep)
        combined_df = pd.concat(obs_list, ignore_index=True)
        aggregated_data[model_type] = {
            "combined_df": combined_df,
            "poorest": combined_df.nsmallest(n_performers, "test_pearson").copy(),
            "best": combined_df.nlargest(n_performers, "test_pearson").copy(),
            "n_replicates": len(replicate_list),
        }
        print(
            f"  {model_type}: n_pb={len(combined_df):,}, "
            f"mean_pearson={combined_df['test_pearson'].mean():.3f}"
        )
    return aggregated_data
