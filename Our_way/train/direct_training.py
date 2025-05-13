# -*- coding: utf-8 -*-
import os
import torch
import json
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback


# 1. 定义动态路径生成函数
def generate_paths(data_name, suffix=""):
    base_dir = "./our_model_train_results/new_2.20_template_right/original_way"  # 基础目录
    if suffix:
        data_name_with_suffix = f"{data_name}_{suffix}"
    else:
        data_name_with_suffix = data_name
    data_dir = os.path.join(base_dir, data_name_with_suffix)  # 数据集名称目录
    output_dir = os.path.join(data_dir, "results")  # 输出目录
    logging_dir = os.path.join(data_dir, "logs")  # 日志目录
    model_dir = os.path.join(data_dir, "models")  # 模型保存目录
    dataset_path = f"./{data_name}.json"  # 数据集路径

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    return {
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "model_dir": model_dir,
        "dataset_path": dataset_path,
    }


data_name = "filtered_no_repeat_output_36788_0.10_0.70"  # 可以动态修改为其他名称
suffix = "epoch16_no_eval_original_way"  # 自定义后缀，可以根据需要修改，若不需要后缀则留空, no_eval _again
paths = generate_paths(data_name, suffix)
epochs = 16
# 增加一个参数来控制数据集使用情况
# 0: 只用训练集进行训练，不使用验证集
# 1: 使用验证集，但是验证集是在提供的数据集中拆分的
# 2: 使用外部的两个指定的数据集分别作为训练集和验证集
dataset_mode = 2

# ================== 需要修改的路径配置 ==================
# 模型ID
model_id = "./llama3-8b-instruction-hf"
max_seq_length = 2048
# ======================================================

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def process_data(samples):
    processed_data = []
    for sample in samples:
        task_type = sample.get("task type")
        if task_type is None:
            print("Missing 'task type' key in sample.")
            continue
        if task_type == 'BQA':
            instruction = "Please answer the following question based on the context, and **only output 'yes' or 'no' without any explanation**."
            user_content = f"Context: {sample['context']}\nQuestion: {sample['question']}"
        elif task_type == 'MCQA':
            instruction = "Please answer the question based on the context, select the correct option from the options, and **only output the number (0,1,2,3) of the correct answer without any explanation**."
            options_str = "\n".join([f"{i}. {option}" for i, option in enumerate(sample['options'])])
            user_content = f"Context: {sample['context']}\nQuestion: {sample['question']}\nOptions:\n{options_str}"
        else:
            print(f"Unknown task type: {task_type}")
            continue

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample["answer"]}
        ]
        processed_data.append({"messages": messages})
    return processed_data


def formatting_prompts_func(samples):
    output_texts = []
    for i in range(len(samples['messages'])):
        text = tokenizer.apply_chat_template(samples['messages'][i], tokenize=False)
        output_texts.append(text)
    return output_texts

if dataset_mode == 0:
    # 只用训练集进行训练，不使用验证集
    train_data_path = paths["dataset_path"]
    train_dataset = load_dataset("json", data_files=train_data_path)['train']
    train_dataset = process_data(train_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = None
elif dataset_mode == 1:
    # 使用验证集，但是验证集是在提供的数据集中拆分的
    train_data_path = paths["dataset_path"]
    dataset = load_dataset("json", data_files=train_data_path)['train']
    split_dataset = dataset.train_test_split(test_size=0.1)  # 10% 作为验证集
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset = process_data(train_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = process_data(eval_dataset)
    eval_dataset = Dataset.from_list(eval_dataset)
elif dataset_mode == 2:
    # 使用外部的两个指定的数据集分别作为训练集和验证集
    train_data_path = paths["dataset_path"]
    test_data_path = "./all_test_data_no_multi.json"

    train_dataset = load_dataset("json", data_files=train_data_path)['train']
    eval_dataset = load_dataset("json", data_files=test_data_path)['train']

    train_dataset = process_data(train_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = process_data(eval_dataset)
    eval_dataset = Dataset.from_list(eval_dataset)
else:
    raise ValueError("Invalid dataset_mode. Valid values are 0, 1, or 2.")

# 打印一些样本进行检查
print("Training data sample:")
print(train_dataset[0])
if eval_dataset is not None:
    print("Evaluation data sample:")
    print(eval_dataset[0])
print(f"train size: {len(train_dataset)}, eval size: {len(eval_dataset) if eval_dataset is not None else 0}")

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=64,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir=paths["output_dir"],
    num_train_epochs=epochs,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8 if eval_dataset is not None else 1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    eval_strategy="epoch" if eval_dataset is not None else "no",
    logging_strategy="steps",
    logging_steps=100,
    bf16=True,
    learning_rate=2e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="tensorboard",
    max_seq_length=max_seq_length,
    packing=False,
    logging_dir=paths["logging_dir"]
)

# 9. 定义回调函数以收集 loss 数据
loss_data = {"train_loss": [], "eval_loss": [], "steps": []}

def save_loss_data(loss_data, model_dir):
    # 保存为 JSONL 文件
    file_path = os.path.join(model_dir, "loss_data.jsonl")
    try:
        with open(file_path, "w") as f:
            if not loss_data["steps"]:
                print("No steps data available. Skipping JSONL writing.")
                return
            for step, train_loss, eval_loss in zip(loss_data["steps"], loss_data["train_loss"], loss_data["eval_loss"]):
                data = {"step": step, "train_loss": float(train_loss), "eval_loss": float(eval_loss) if eval_loss is not None else None}
                f.write(json.dumps(data) + "\n")
            print(f"Successfully wrote {len(loss_data['steps'])} records to {file_path}.")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

    # 生成折线图
    plt.figure(figsize=(10, 6))
    valid_steps = []
    valid_train_loss = []
    valid_eval_loss = []
    for step, train_loss, eval_loss in zip(loss_data["steps"], loss_data["train_loss"], loss_data["eval_loss"]):
        if eval_loss is not None:
            valid_steps.append(step)
            valid_train_loss.append(train_loss)
            valid_eval_loss.append(eval_loss)

    if valid_steps:
        plt.plot(valid_steps, valid_train_loss, label="Train Loss")
        plt.plot(valid_steps, valid_eval_loss, label="Eval Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Train and Eval Loss Over Steps")
        plt.legend()
        plot_path = os.path.join(model_dir, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Successfully saved loss plot to {plot_path}.")
    else:
        print("No valid steps data for plotting. Skipping plot generation.")

# 10. 自定义训练循环以收集 loss 数据
class LossCallback(TrainerCallback):
    def __init__(self, loss_data, model_dir):
        self.loss_data = loss_data
        self.model_dir = model_dir

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.loss_data["train_loss"].append(logs["loss"])
                self.loss_data["steps"].append(state.global_step)
                if "eval_loss" not in logs:
                    self.loss_data["eval_loss"].append(None)
            if "eval_loss" in logs:
                if not self.loss_data["eval_loss"]:
                    self.loss_data["eval_loss"].append(logs["eval_loss"])
                else:
                    self.loss_data["eval_loss"][-1] = logs["eval_loss"]
                print(f"Current loss data: {self.loss_data}")  # 新增：打印当前损失数据
                # 保存 loss 数据和生成折线图
                save_loss_data(self.loss_data, self.model_dir)

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束时再次保存 loss 数据和生成折线图
        print(f"Final loss data: {self.loss_data}")  # 新增：打印最终损失数据
        save_loss_data(self.loss_data, self.model_dir)

# 添加回调函数
loss_callback = LossCallback(loss_data, paths["model_dir"])

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func
)

trainer.add_callback(loss_callback)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model
trainer.save_model(paths["model_dir"])
