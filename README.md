# README

本程序为[bert模型](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py)代码改动，程序旨在解决NLI问题

训练方式(--do_train) 和测试 (--do_eval)

```
(nlp) tfhou@DB18:/DATA4_DB3/data/tfhou/pytorch-pretrained-BERT-master/examples$ 
CUDA_VISIBLE_DEVICES=4 python run_classifier.py   \
--task_name $TASK_NAME  \
--do_train   --do_eval  \
--do_lower_case   --data_dir $GLUE_DIR/$TASK_NAME \
--bert_model bert-base-uncased   --max_seq_length 128 \
--train_batch_size 4   --learning_rate 2e-5  \
--num_train_epochs 3.0  \
--output_dir /tmp/$TASK_NAME/

```

测试方式(--do_test,--eval_batch_size 1)

```
(nlp) tfhou@DB18:/DATA4_DB3/data/tfhou/pytorch-pretrained-BERT-master/examples$ 
CUDA_VISIBLE_DEVICES=0 python run_classifier.py   \
--task_name $TASK_NAME     \
--do_test \
--do_lower_case   --data_dir $GLUE_DIR/$TASK_NAME  \
--bert_model bert-base-uncased   --max_seq_length 128 \
--train_batch_size 1   --learning_rate 2e-5  \
--num_train_epochs 3.0 \
--eval_batch_size 1  --output_dir /tmp/$TASK_NAME/

```

pytorch=1.0.0 DB18 python 3.7.3

~~训练模型\temp\QNLI 和 DATA4\pytorch-pretrained-BERT\model 下~~

训练模型来找我要吧，也可以烧实验室GPU. :)
