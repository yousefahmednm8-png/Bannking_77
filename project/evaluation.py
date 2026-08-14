import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data import BankingDataModule
from model import BankingClassifier


train_path = r"E:\Dl(learn)\Bannking_77\train.csv"
test_path = r"E:\Dl(learn)\Bannking_77\test.csv"

model_path = r"E:\Dl(learn)\hugging face (parctise)\distilbert-base"

checkpoint_path = (
    r"E:\Dl(learn)\Bannking_77\checkpoints\banking 77-v1.ckpt"
)


def main():

    data = BankingDataModule(
        train_path=train_path,
        test_path=test_path,
        batch_size=32,
    )

    model = BankingClassifier.load_from_checkpoint(
        checkpoint_path,
        model_name=model_path,
        num_classes=77,
    )

    data.setup(stage="test")

    test_loader = data.test_dataloader()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(
                logits,
                dim=1,
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Macro Precision : {precision:.4f}")
    print(f"Macro Recall    : {recall:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")

    class_names = data.label_encoder.classes_

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    plt.figure(figsize=(20, 18))

    plt.imshow(cm)

    plt.title(
        "Banking77 - Confusion Matrix"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.colorbar()

    plt.tight_layout()

    plt.show()

    cm_without_diagonal = cm.copy()

    np.fill_diagonal(
        cm_without_diagonal,
        0,
    )

    top_confusions = []

    for i in range(len(class_names)):

        for j in range(len(class_names)):

            if cm_without_diagonal[i, j] > 0:

                top_confusions.append(
                    (
                        cm_without_diagonal[i, j],
                        class_names[i],
                        class_names[j],
                    )
                )

    top_confusions.sort(
        reverse=True
    )

    print("\n" + "=" * 60)
    print("TOP CONFUSIONS")
    print("=" * 60)

    for count, true_class, predicted_class in top_confusions[:20]:

        print(
            f"{count:3d} | "
            f"True: {true_class:<45} "
            f"Predicted: {predicted_class}"
        )


if __name__ == "__main__":
    main()