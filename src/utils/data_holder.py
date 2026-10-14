from transformers import AutoTokenizer

from utils.data_handler import DataHandler


class DataHolder:
    def __init__(self, args):
        self.args = args
        #load arguments in self.args

        self.data_sep=self.args.csv_sep
        self.text_column=self.args.text_column
        self.model_id = self.args.model_id
        self.model_type = self.args.model_type
        self.model_out = self.args.model_out
        self.output_dir = self.args.output_dir
        self.output_attentions = self.args.output_attentions
        self.no_cache = self.args.no_cache
        self.max_len= self.args.max_len
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.lr= self.args.lr
        self.warmup_proportion = self.args.warmup_proportion
        self.adam_epsilon = self.args.adam_epsilon
        self.weight_decay = self.args.weight_decay
        self.max_grad_norm = self.args.max_grad_norm
        self.metric_for_best_model = self.args.metric_for_best_model
        self.seed = self.args.seed

        self.training_data_dir = self.args.training_data_dir
        self.test_data_dir = self.args.test_data_dir
        self.eval_data_dir = self.args.eval_data_dir
        self.labels_ids = self.args.labels_ids
        self.labels_default_start_index = self.args.labels_default_start_index

    def load_data(self):
        #instantiate data handler
        tokenizer=AutoTokenizer.from_pretrained(self.model_id)

        print("Loading training data...")
        training_data_handler=DataHandler(self.training_data_dir,self.batch_size,
                                               self.labels_ids,tokenizer,self.max_len, self.labels_default_start_index,
                                          self.data_sep, self.text_column)
        print("Loading test data...")
        test_data_handler=DataHandler(self.test_data_dir,self.batch_size,
                                                  self.labels_ids,tokenizer,self.max_len, self.labels_default_start_index,
                                      self.data_sep, self.text_column)

        print("Loading training data...")

        dev_data_handler=DataHandler(self.eval_data_dir,self.batch_size,
                                                    self.labels_ids,tokenizer,self.max_len, self.labels_default_start_index,
                                      self.data_sep, self.text_column)

        #load data
        self.training_data=training_data_handler.get_tokenized_data()
        self.test_data=test_data_handler.get_tokenized_data()
        self.dev_data=dev_data_handler.get_tokenized_data()
        label2id, id2label = training_data_handler.get_labels_dict()
        return self.training_data,self.dev_data,self.test_data,label2id, id2label
