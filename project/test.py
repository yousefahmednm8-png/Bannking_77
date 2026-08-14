import lightning as l 
from data import BankingDataModule
from model import BankingClassifier


# paths

train_path=r"E:\Dl(learn)\Bannking_77\train.csv"
test_path=r"E:\Dl(learn)\Bannking_77\test.csv"
model_path=r"E:\Dl(learn)\hugging face (parctise)\distilbert-base"
checkpoint_path=r"E:\Dl(learn)\Bannking_77\checkpoints\banking 77.ckpt"


# data module
data =BankingDataModule(
    train_path=train_path,
    test_path=test_path,
    batch_size=32
)


# load model
model=BankingClassifier.load_from_checkpoint(
    checkpoint_path,
    model_name=model_path,
    num_classes=77
)


# trainer 

trainer=l.Trainer(
    accelerator="auto",
    devices="auto"
)

# test 

if __name__=="__main__":
    results=trainer.test(
        model=model,
        datamodule=data,
    )
    print("\n Results :")
    print(results)