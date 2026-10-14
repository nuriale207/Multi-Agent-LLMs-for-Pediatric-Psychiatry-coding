import argparse
import os
import torch as torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from utils import metrics
from utils.data_holder import DataHolder

from transformers import AdamW, get_cosine_schedule_with_warmup

#def main method
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Arguments for training and prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Arguments
    parser.add_argument('--training_data_dir', type=str, default="train_data",
                             help="Directory containing the training dataset")
    parser.add_argument('--test_data_dir', type=str, default="test_data",
                             help="Directory containing the test dataset")
    # evaluation data
    parser.add_argument('--eval_data_dir', type=str, default="eval_data",
                             help="Directory containing the evaluation dataset")
    #train or test mode
    parser.add_argument('--mode', type=str, default="train",
                                help="train or test")
    parser.add_argument("--labels_ids", type=str, default=None)
    parser.add_argument("--labels_default_start_index", type=int, default=None,
                        help="Default start index for the labels. If None, the labels will be indexed from 0 to n_labels-1")

    parser.add_argument("--csv_sep", type=str, default="\t")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Name of the column containing the text to classify")
    parser.add_argument('--model_id', type=str, default="allenai/longformer-base-4096",
                             help="Pretrained model name")
    parser.add_argument('--model_type', type=str, default="longformer",
                             help="Pretrained model type")
    parser.add_argument('--model_path', type=str, default="model_id",
                                help="Directory containing the model to load")
    parser.add_argument('--model_out', type=str, default="output",
                             help="Directory containing the model to load")
    parser.add_argument('--output_dir', type=str, default="output",
                             help="Directory to save the predictions")
    parser.add_argument("--experiment_name", type=str, default="longformer",
                                help="Name of the experiment")
    parser.add_argument('--output_attentions', type=bool, default=False,
                             help="Whether to output attentions")
    parser.add_argument('--no_cache', type=bool, default=False,
                             help="Whether to cache the features")
    parser.add_argument('--max_len', type=int, default=512,
                             help="Maximum length of the sequence")
    parser.add_argument('--batch_size', type=int, default=4,
                             help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50,
                             help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=1e-5,
                             help="Learning rate")
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                             help="Proportion of training to perform linear learning rate warmup for")
    parser.add_argument('--warmup_steps', type=int, default=125,
                        help="Set the number of warmup steps")
    parser.add_argument('--grad_accum', type=int, default=1,
                        help="Set the number of gradient accumulation steps")
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                             help="Epsilon for Adam optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.07,
                             help="Weight decay if we apply some")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                             help="Max gradient norm")
    parser.add_argument('--metric_for_best_model', type=str, default="f1",
                             help="Metric for best model")
    parser.add_argument('--seed', type=int, default=42,
                             help="Random seed for initialization")

    #Instantiate DataHolder
    args = parser.parse_args()
    print(args)
    data_holder=DataHolder(args)
    train_data,dev_data,test_data,labels2id,id2label=data_holder.load_data()

    print(id2label)
    print(labels2id)
    print(len(labels2id))
    if args.model_path=="model_id":
        model_path=args.model_id
    else:
        model_path=args.model_path

        #If we are in train mode
    if args.mode=="train":
        # Instantiate Model
        print("Loading model")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(labels2id),
                                                                   problem_type="multi_label_classification",
                                                                   id2label=id2label,
                                                                   label2id=labels2id,)

        output_dir = os.path.join(args.output_dir, args.experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        log_dir = os.path.join(output_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        model_save = os.path.join(output_dir, "model")
        if not os.path.exists(model_save):
            os.makedirs(model_save)

        #Get metrics
        #Instantiate training arguments
        training_args = TrainingArguments(
            output_dir=model_save,          # output directory
            num_train_epochs=args.epochs,              # total number of training epochs
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
            learning_rate=args.lr,   # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,               # strength of weight decay
            logging_dir=log_dir,            # directory for storing logs
            logging_steps=1000,
            save_steps=1000,
            save_total_limit=1,
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            seed=args.seed,
            fp16=True,
            metric_for_best_model="f1_micro",
            adam_epsilon=args.adam_epsilon
        )
        optimizer=AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
        scheduler=get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=training_args.num_train_epochs*len(train_data)/args.batch_size)
        print("Training model")
        trainer= Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_data,         # training dataset
            eval_dataset=dev_data,             # evaluation dataset
            compute_metrics=metrics.compute_metrics,         # define metrics function
            optimizers=(optimizer,scheduler)

        )


        #Train model
        trainer.train()
    #Test mode
    elif args.mode=="test":
        # Instantiate  from path
        print(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(labels2id),
                                                                   problem_type="multi_label_classification",
                                                                   id2label=id2label,
                                                                   label2id=labels2id,
                                                                   local_files_only=True
                                                                   )

        #Create output directory inside the experiment directory
        output_dir = os.path.join(args.output_dir, args.experiment_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        #Create test directory inside the output directory
        test_dir=os.path.join(output_dir,"test")
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # Instantiate training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,  # output directory
            num_train_epochs=args.epochs,  # total number of training epochs
            per_device_train_batch_size=args.batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
            learning_rate=args.lr,  # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,  # strength of weight decay
            logging_dir=test_dir,  # directory for storing logs
            logging_steps=300,
            save_steps=300,
            save_total_limit=1,
            evaluation_strategy="steps",
            eval_steps=300,
            load_best_model_at_end=True,
            seed=args.seed,
            fp16=True,
            metric_for_best_model="f1_micro",
            adam_epsilon=args.adam_epsilon
        )
        optimizer = AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=training_args.num_train_epochs * len(
                                                        train_data) / args.batch_size)
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_data,  # training dataset
            eval_dataset=test_data,  # evaluation dataset
            compute_metrics=metrics.compute_metrics,  # define metrics function

        )
        #Test model
        evaluation=trainer.evaluate(test_data)
        predictions=trainer.predict(test_data)

        #Save predictions
        predictions_file = os.path.join(test_dir, "predictions.csv")
        model_config=model.config
        labels_names=list(model_config.id2label.values())
        pred_rows = []
        pred_sigmoid_rows = []
        for prediction in predictions.predictions:
            sigmoid = torch.nn.Sigmoid()
            prediction_bin = sigmoid(torch.Tensor(prediction))
            prediction_bin[prediction_bin > 0.5] = 1
            prediction_bin[prediction_bin <= 0.5] = 0
            prediction_bin=prediction_bin.tolist()
            pred_dict=dict(zip(labels_names,prediction_bin))
            pred_rows.append(pred_dict)
            #append the sigmoid predictions
            prediction_sig = sigmoid(torch.Tensor(prediction))
            # turn to list
            prediction_sig = prediction_sig.tolist()
            pred_sig_dict=dict(zip(labels_names,prediction_sig))
            pred_sigmoid_rows.append(pred_sig_dict)

        pred_df = pd.DataFrame(pred_rows, columns=labels_names)
        pred_sigmoid_df = pd.DataFrame(pred_sigmoid_rows, columns=labels_names)

        #Save predictions
        pred_df.to_csv(predictions_file,index=False)

        predictions_sigmoid_file = os.path.join(test_dir, "predictions_raw.csv")
        pred_sigmoid_df.to_csv(predictions_sigmoid_file,index=False)

        #Save evaluation
        evaluation_file=os.path.join(test_dir,"evaluation.txt")
        with open(evaluation_file,"w") as f:
            for key,value in evaluation.items():
                f.write(str(key)+" "+str(value)+"\n")
