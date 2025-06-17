import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# 加载预训练的T5模型和分词器
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载数据集，这里我们使用一个示例数据集
# 可以根据需要替换为你自己的数据集
dataset = load_dataset('cnn_dailymail', '3.0.0', split='train[:1%]')
dataset = dataset.train_test_split(test_size=0.1)

# 数据预处理函数
def preprocess_function(examples):

    inputs = [ex['article'] for ex in examples['input']]
    targets = [ex['highlights'] for ex in examples['target']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)

    # 将目标文本作为标签进行处理
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=150, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 应用预处理函数
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=3,
)

# 定义评估指标
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 计算BLEU、ROUGE等指标
    bleu = load_metric('bleu')
    rouge = load_metric('rouge')

    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        'bleu': bleu_result['bleu'],
        'rouge1': rouge_result['rouge1'],
        'rouge2': rouge_result['rouge2'],
        'rougeL': rouge_result['rougeL']
    }

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()
