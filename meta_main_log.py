
import torch, math
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from meta_model_grad_b import MetaLearner
from meta_dataset import SolarPredictionDataset, MetaSolarPredictionDataset, _extract_time_from_path
from meta_engine_log import meta_train_one_epoch, meta_evaluate
from utils import set_seed, get_device, start_run_log, RunLogger

CONFIG = {
    "DATA_DIR": "/home/user/hanwool/new_npy",
    "MODEL_SAVE_PATH": "./best_bayesian_meta_model_with_logging.pth",
    "SEED": 42,
    "EPOCHS": 10,
    "META_LR": 1e-5,
    "INNER_LR": 1e-4,
    "INNER_STEPS": 5,
    "KL_WEIGHT_INIT": 1e-8,
    "KL_WEIGHT_MAX": 1e-6,
    "GRAD_CLIP_NORM": 5.0,
    "TASKS_PER_EPOCH": 10,
    "K_SHOT": 2,
    "K_QUERY": 4,
    "NUM_ADAPTATION_STEPS": 5,
    "NUM_EVAL_SAMPLES": 3,
    "INPUT_LEN": 4,
    "TARGET_LEN": 4,
    "DATA_MIN": 0.0,
    "DATA_MAX": 26.41,

    "MC_INNER_SAMPLES": 2,
    "MC_OUTER_SAMPLES": 4,
    "NLL_TAU2": 1e-3,
    "W_VEL": 0.50,
    "W_ACC": 0.25,
    "TIME_WEIGHTS": [1.0, 0.9, 0.8, 0.7],
    "MC_DIVERSITY_THR": 1e-4,
    "MC_INPUT_NOISE_STD": 5e-3,
    "MC_DIVERSITY_MAX_TRIES": 3,
    "VAR_INFLATE_ALPHA": 0.05,
}

def visualize_meta_predictions(mean_pred, std_pred, ground_truth, sample_idx=0):
    print("\\n--- Visualizing Final Test Task Prediction ---")
    def denormalize(tensor):
        val_range = CONFIG['DATA_MAX'] - CONFIG['DATA_MIN']
        return (tensor.cpu().numpy() * (val_range / 2.0)) + ((CONFIG['DATA_MAX'] + CONFIG['DATA_MIN']) / 2.0)

    gt_sequence   = denormalize(ground_truth[sample_idx])
    mean_sequence = denormalize(mean_pred[sample_idx])
    std_sequence  = std_pred[sample_idx].cpu().numpy()

    T = gt_sequence.shape[0]
    fig, axes = plt.subplots(3, T, figsize=(4*T, 10))
    fig.suptitle('Final Test Task Prediction (Sample Index: {})'.format(sample_idx+1))

    for j in range(T):
        time_step = (j + 1) * 30
        ax = axes[0, j]
        im = ax.imshow(gt_sequence[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Target (t+{time_step}m)'); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, j]
        im = ax.imshow(mean_sequence[j], cmap='gray', vmin=CONFIG['DATA_MIN'], vmax=CONFIG['DATA_MAX'])
        ax.set_title(f'Mean Prediction (t+{time_step}m)'); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[2, j]
        im = ax.imshow(std_sequence[j], cmap='viridis')
        ax.set_title('Uncertainty (Std Dev)'); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    save_path = f"meta_prediction_sample_{sample_idx+1}_with_logging.png"
    plt.savefig(save_path)
    print(f"Saved meta-prediction visualization to {save_path}")
    plt.show()

def main():
    set_seed(CONFIG['SEED'])
    device = get_device()

    data_dir = Path(CONFIG['DATA_DIR'])
    all_files = sorted(list(data_dir.glob("*.npy")))

    # Simple seasonal split like the original file
    from meta_dataset import _extract_time_from_path
    timestamps = [_extract_time_from_path(p.name) for p in all_files]
    month_indices = [int(ts[4:6]) for ts in timestamps]
    year_indices = [int(ts[0:4]) for ts in timestamps]

    def get_indices_months(month_list, years=None):
        idx = []
        for i, (m, y) in enumerate(zip(month_indices, year_indices)):
            if m in month_list and (years is None or y in years):
                idx.append(i)
        return idx

    train_months = [3,4,5,6,7,8,9]
    val_months   = [10,11,12]
    test_years   = [2023]

    train_indices = get_indices_months(train_months, years=[2021,2022])
    val_indices   = get_indices_months(val_months,   years=[2021,2022])
    test_indices  = [i for i,y in enumerate(year_indices) if y in test_years]

    base_train_dataset = SolarPredictionDataset([all_files[i] for i in train_indices], CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'])
    base_val_dataset   = SolarPredictionDataset([all_files[i] for i in val_indices],   CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'])
    base_test_dataset  = SolarPredictionDataset([all_files[i] for i in test_indices],  CONFIG['INPUT_LEN'], CONFIG['TARGET_LEN'])

    train_task_generator = MetaSolarPredictionDataset(base_train_dataset, CONFIG['TASKS_PER_EPOCH'], CONFIG['K_SHOT'], CONFIG['K_QUERY'], shuffle=True)
    val_task_generator   = MetaSolarPredictionDataset(base_val_dataset,   1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])
    test_task_generator  = MetaSolarPredictionDataset(base_test_dataset,  1, CONFIG['K_SHOT'], CONFIG['K_QUERY'])

    val_task  = val_task_generator[0]
    test_task = test_task_generator[0]

    meta_learner = MetaLearner(CONFIG).to(device)
    meta_optimizer = optim.AdamW(meta_learner.parameters(), lr=CONFIG['META_LR'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, 'min', patience=5, factor=0.5)

    run_dir = start_run_log(base_dir="log", config_dict=CONFIG)
    logger = RunLogger(run_dir)

    best_val_loss = float('inf')
    print("\\n--- Starting Meta-Training ---")
    for epoch in range(CONFIG['EPOCHS']):
        kl_weight = min(CONFIG['KL_WEIGHT_MAX'], CONFIG['KL_WEIGHT_INIT'] + (CONFIG['KL_WEIGHT_MAX'] - CONFIG['KL_WEIGHT_INIT']) * (2 * epoch / CONFIG['EPOCHS']))
        meta_learner.config['KL_WEIGHT'] = kl_weight

        train_loss, train_metrics = meta_train_one_epoch(meta_learner, train_task_generator, meta_optimizer, device, CONFIG['GRAD_CLIP_NORM'])
        val_loss, _, _, _, val_metrics = meta_evaluate(meta_learner, val_task, device, CONFIG['NUM_ADAPTATION_STEPS'], CONFIG['NUM_EVAL_SAMPLES'])

        print(f"\\nEpoch {epoch+1}/{CONFIG['EPOCHS']} | KL Weight: {kl_weight:.2e} | Meta-Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Train metrics: {train_metrics} | Val metrics: {val_metrics}")
        logger.log_row(epoch+1, "train", train_loss, train_metrics, kl_weight)
        logger.log_row(epoch+1, "val",   val_loss,   val_metrics,   kl_weight)

        if not math.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(meta_learner.state_dict(), CONFIG['MODEL_SAVE_PATH'])
            print(f"Best model saved to {CONFIG['MODEL_SAVE_PATH']} with validation loss: {best_val_loss:.6f}")
        scheduler.step(val_loss)

    if Path(CONFIG['MODEL_SAVE_PATH']).exists():
        meta_learner.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH'], map_location=device))
        test_loss, mean_pred, std_pred, ground_truth, test_metrics = meta_evaluate(
            meta_learner, test_task, device, CONFIG['NUM_ADAPTATION_STEPS'], CONFIG['NUM_EVAL_SAMPLES']
        )
        if not math.isnan(test_loss) and mean_pred is not None:
            print(f"Final Test Task Loss: {test_loss:.6f} | MSE: {test_metrics.get('mse')} | MAE: {test_metrics.get('mae')} | SSIM: {test_metrics.get('ssim')}")
            logger.log_row("final", "test", test_loss, test_metrics, None, notes="final evaluation")
            # optional viz
        else:
            print("Final evaluation failed: NaN was produced during adaptation.")
    else:
        print("Could not find a saved model. Final evaluation skipped.")

if __name__ == '__main__':
    main()
