from operator import index
import re

from idna import encode
import lightning as l
from matplotlib.pylab import random_sample
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,DataCollatorWithPadding
from sklearn.model_selection import train_test_split


class BankingDataset(Dataset):

    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        text = self.dataframe.iloc[idx]["text"]
        label = self.dataframe.iloc[idx]["label"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_len=60
        )

        encoding["labels"] = label

        return encoding






Model_path=r"E:\Dl(learn)\hugging face (parctise)\distilbert-base"
class BankingDataModule(l.LightningDataModule):
    def __init__(
        self,
        train_path,
        test_path,
        batch_size=32,
    ):
        super().__init__()
        self.train_path=train_path
        self.test_path=test_path
        self.batch_size=batch_size
        self.tokenizer=AutoTokenizer.from_pretrained(
            Model_path,
            local_files_only=True
        )
        self.data_collator=DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )      
        self.label_encoder=LabelEncoder()
        
        self.train_df=None
        
        self.test_df=None
        self.num_classes=None
    def setup(self,stage=None):
         #load data
         self.train_df=pd.read_csv(self.train_path)
         self.test_df=pd.read_csv(self.test_path)
        
        # fit labelecoder on train Categories
         self.label_encoder.fit(self.train_df["category"])
         
         self.train_df["label"]=self.label_encoder.transform(
             self.train_df["category"]
         )     
         self.test_df["label"]=self.label_encoder.transform(
             self.test_df["category"]
         )     
         # num of classes 
         self.num_classes=len(self.label_encoder.classes_)
         
         self.train_dataset=BankingDataset(
             self.train_df,
             self.tokenizer
         )
         self.test_dataset=BankingDataset(
             self.test_df,
             self.tokenizer
         )
         train_df,val_df=train_test_split(
             self.train_df,
             test_size=0.1,
             random_state=42,
             stratify=self.train_df["label"]
         )
         self.train_df=train_df.reset_index(drop=True)
         self.val_df=val_df.reset_index(drop=True)
         
         
         self.train_dataset=BankingDataset(
             self.train_df,
             self.tokenizer
         )

         self.val_dataset=BankingDataset(
             self.val_df,
             self.tokenizer
         )
         self.test_dataset=BankingDataset(
             self.test_df,
             self.tokenizer
         )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            pin_memory=True,
            persistent_workers=False,
            num_workers=0,

            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.data_collator
        )
         
         
         


if __name__ == "__main__":

    dm = BankingDataModule(
        train_path=r"E:\Dl(learn)\Bannking_77\train.csv",
        test_path=r"E:\Dl(learn)\Bannking_77\test.csv",
        batch_size=32,
    )

    dm.setup()
    print("Train size:", len(dm.train_dataset))
    print("Validation size:", len(dm.val_dataset))
    print("Test size:", len(dm.test_dataset))