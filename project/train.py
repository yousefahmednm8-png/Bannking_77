
import lightning as l 
import torch 
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)
from data import BankingDataModule
from model import BankingClassifier



train_path=r"E:\Dl(learn)\Bannking_77\train.csv"
test_path=r"E:\Dl(learn)\Bannking_77\test.csv"
model_path=r"E:\Dl(learn)\hugging face (parctise)\distilbert-base"

batch_size=32
learning_rate=1e-4
weight_decay=0.1

num_classes=77
max_epochs=8


# data module
data=BankingDataModule(
    train_path=train_path,
    test_path=test_path,
    batch_size=batch_size
)

# model
model=BankingClassifier(
    model_name=model_path,
    num_classes=num_classes,
    learning_rate=learning_rate,
    weight_decay=weight_decay
)

# checkpoint 

checkpoint_callback=ModelCheckpoint(
    dirpath="checkpoints/",
    filename="banking 77",
    monitor="val_f1",
    mode="max",
    save_top_k=1,
    save_last=True
)

# early stopping
early_stopping=EarlyStopping(
    monitor="val_f1",
    mode="max",
    patience=2,
    verbose=True
)

# trainer

trainer=l.Trainer(
    max_epochs=max_epochs,
    accelerator="auto",
    devices="auto",
    precision="16-mixed",
    gradient_clip_val=1.0,
    callbacks=[
        checkpoint_callback,
        early_stopping,
    ],
    
    log_every_n_steps=10,
    deterministic=False,
)


# training


if __name__=="__main__":
    trainer.fit(
        model,
        datamodule=data,
    )
    