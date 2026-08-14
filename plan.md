                BANKING77
                    │
                    ▼
                   EDA
                    │
                    ▼
              DataModule
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   AutoTokenizer        Dynamic Padding
        │                       │
        └───────────┬───────────┘
                    ▼
                DataLoader
                    │
                    ▼
                DistilBERT
                    │
                    ▼
             LightningModule
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        AdamW    Scheduler  Metrics
          │
          ▼
        Trainer
          │
     ┌────┴────┐
     ▼         ▼
Checkpoint  EarlyStopping
     │
     ▼
 Best Model
     │
     ▼
    Test
     │
     ▼
 Inference