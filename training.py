from ultralytics import YOLO
import os
import wandb

wandb.login()

HOME = os.path.expanduser('~')
yolov10_models = {
    "yolov10n": YOLO("yolov10n.pt"),
}

DATASET_CONFIG = os.path.join(HOME, 'Desktop', 'trafic_data', 'data_1.yaml')
main_folder = 'new-training-graphs'
if not os.path.exists(main_folder):
    os.makedirs(main_folder)

PATIENCE = 5
metrics = dict()

def train_models():
    for model_name, model_weights in yolov10_models.items():
        print(f"Training {model_name}...")
        wandb.init(project='Eco-Sight yolov10', name=model_name)
        results = model_weights.train(
            data=DATASET_CONFIG,
            epochs=100,
            batch=16,
            imgsz=640,
            name=model_name,
            workers=2,
            plots=True,
            lr0=0.01,
            patience=PATIENCE,
            device=0,
            save=True,
            freeze=11,
            verbose=True,
        )
        precision, recall = results.results_dict['metrics/precision(B)'], results.results_dict['metrics/recall(B)']
        mAP50, mAP50_95 = results.results_dict['metrics/mAP50(B)'], results.results_dict['metrics/mAP50-95(B)']
        epochs = range(1, 101)  # 修改为2个epoch，因为设置了2个epochs进行训练
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores = [f1_score] * len(epochs)

        for epoch, f1_score in zip(epochs, f1_scores):
            wandb.log({
                "mAP50": mAP50,
                "mAP50_95": mAP50_95,
                "F1_Score": f1_score,
                "Precision": precision,
                "Recall": recall,
                "_epoch": epoch
            })
        metrics[model_name] = [mAP50, mAP50_95, f1_scores, epochs]
        print(f"Model {model_name} training complete.")
        wandb.finish()

train_models()