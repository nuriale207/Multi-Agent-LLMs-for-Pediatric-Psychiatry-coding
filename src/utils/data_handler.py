import numpy as np
import pandas as pd

from datasets import Dataset


class DataHandler:
    def __init__(self, data_path, batch_size, labels_ids,tokenizer,max_length, labels_default_start_index=None,data_sep="\t", text_column="text"):
        print("DataHandler init")
        self.data_path = data_path
        self.batch_size = batch_size
        self.labels_ids = str(labels_ids)
        self.data_sep=data_sep
        if not labels_ids is None:
            self.labels_ids = labels_ids.replace("\n", "")
            self.labels_ids = labels_ids.rstrip("\n")
            self.labels_ids=list(labels_ids.split(',')) #convert to list
            self.labels_ids= [label.lower() for label in self.labels_ids]
            print("LABELS IDS: "+str(self.labels_ids))
        elif not labels_default_start_index is None:
            df_train = pd.read_csv(self.data_path, index_col = False, sep=self.data_sep)
            self.labels_ids = list(df_train.columns[labels_default_start_index:])
            self.labels_ids= [label.lower() for label in self.labels_ids]
        self.text_column = text_column.lower()
        self.tokenizer= tokenizer
        self.max_length = max_length
        self.get_loader()

    def get_loader(self):
        df_train = pd.read_csv(self.data_path, sep=self.data_sep)
        #Get the data and labels columns
        df_train.columns = [x.lower() for x in df_train.columns]

        df_train = df_train.rename(columns={self.text_column: "text"})
        df_train = df_train.filter(["text"]+self.labels_ids, axis=1)
        df_train[self.labels_ids] = df_train[self.labels_ids].fillna(0).astype(int)

        #Create a dataset from the pandas dataframe
        train_data = Dataset.from_pandas(df_train)
        #Save data in local variable
        self.data = train_data
        #Get the labels
        labels = [label for label in train_data.features.keys() if label not in ['text']]
        self.labels=labels

    def preprocess_data(self,examples):
        # take a batch of texts
        text = examples["text"]
        # encode them
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length)
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

    def get_tokenized_data(self):
        tokenized_train = self.data.map(self.preprocess_data, batched=True)
        tokenized_train.set_format("torch")
        return tokenized_train

    def get_labels_dict(self):
        id2label = {idx:label for idx, label in enumerate(self.labels)}
        label2id = {label:idx for idx, label in enumerate(self.labels)}
        return label2id,id2label
