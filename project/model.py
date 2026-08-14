import lightning as l
import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification,get_cosine_schedule_with_warmup
from torchmetrics.classification import MulticlassAccuracy,MulticlassF1Score

class BankingClassifier(l.LightningModule):
    def __init__(self,
                model_name,
                num_classes,
                learning_rate=1e-4,
                weight_decay=0.1):
        super().__init__()
        self.save_hyperparameters()
        # distil bert
        self.model=AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            local_files_only=True
        )
        # partial freezing
        for layer in self.model.distilbert.transformer.layer[:3]:
             for param in layer.parameters():
                 param.requires_grad = False
        # loss
        self.loss_fn=nn.CrossEntropyLoss()
        
        # metrics
        self.train_accuracy=MulticlassAccuracy(
            num_classes=num_classes
        )
        
        self.val_accuracy=MulticlassAccuracy(
            num_classes=num_classes
        )
        
        self.test_accuracy=MulticlassAccuracy(
            num_classes=num_classes
        )
        
        self.val_f1=MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )
        
        self.test_f1=MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )
        
    def forward(self,input_ids,attention_mask):
        outputs=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return outputs.logits
    def training_step(self,batch,batch_idx):
        logits=self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        labels=batch["labels"].long()
        loss=self.loss_fn(
            logits,
            labels
        )
        preds=torch.argmax(logits,dim=1)
        self.train_accuracy(
            preds,
            labels
        )
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        
        self.log(
            "train_acc",
            self.train_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return loss
    def validation_step(self,batch,batch_idx):
        logits=self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        labels=batch["labels"].long()
        
        loss=self.loss_fn(
            logits,
            labels
        )
        self.log(
            "val_loss",
            loss,
             prog_bar=True,
             on_step=False,
             on_epoch=True
        )
        preds=torch.argmax(logits,dim=1)
        
        self.val_accuracy(
            preds,
            labels
        )
        
        self.val_f1(
            preds,
            labels
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True
                )
        
        self.log(
            "val_f1",
            self.val_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True
                )
        
    
    def test_step(self, batch, batch_idx):

        logits = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels=batch["labels"].long()
        loss = self.loss_fn(
            logits,
            labels,
        )
        preds=torch.argmax(logits,dim=1)
        
        
        self.test_accuracy(
            preds,
            labels,
        )
        self.test_f1(
            preds,
            labels,
        )
        self.log(
            "test_acc",
            self.test_accuracy,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_f1",
            self.test_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
    self.parameters(),
    lr=self.hparams.learning_rate,
    weight_decay=self.hparams.weight_decay,
)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
model_name=r"E:\Dl(learn)\hugging face (parctise)\distilbert-base"
if __name__=="__main__":
    model=BankingClassifier(
        model_name="distilbert-base-uncased",
        num_classes=77
    )
    print(model)