# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import Sampler, SequentialSampler, WeightedRandomSampler
import json
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_callback import TrainerCallback

# 相对于epoch=3来说，增加了同步训练的参数等一些相应的调节
# 1. 定义动态路径生成函数
def generate_paths(data_name, suffix=""):
    base_dir = "./our_model_train_results/new_2.20_template_right/our_idea_course_learning/epoch6"  # 基础目录
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


data_name = "final_all_MCQA_23880_data_new_score"  # 可以动态修改为其他名称
suffix = "_all_test_deepseek_change_epoch16_again1"  # 自定义后缀，可以根据需要修改，若不需要后缀则留空, no_eval _again
paths = generate_paths(data_name, suffix)
epoch_number = 16 # epoch数量
# ================== 新增课程学习配置 ==================
curriculum_config = {
    'initial_max': 0.3,
    'max_epochs': epoch_number,
    'min_coverage': 0.3,
    'warmup_epochs': 1,          # 缩短暖身阶段到1个epoch
    'decay_factor': 0.7,         # 降低衰减系数以保留更多历史数据
    'progressive_factor': 0.85   # 新增：渐进扩展系数
}
# ==============================================
# 增加一个参数来控制数据集使用情况
# 0: 只用训练集进行训练，不使用验证集 no_eval
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
    complexity_scores = []
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
        complexity_scores.append(sample["ComplexityScore_norm"])
        # 新增排序逻辑
    combined = zip(processed_data, complexity_scores)
    processed_data, complexity_scores = zip(*combined)
    return list(processed_data), list(complexity_scores)

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
    train_dataset, train_complexity_scores = process_data(train_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = None
elif dataset_mode == 1:
    # 使用验证集，但是验证集是在提供的数据集中拆分的
    train_data_path = paths["dataset_path"]
    dataset = load_dataset("json", data_files=train_data_path)['train']
    split_dataset = dataset.train_test_split(test_size=0.1)  # 10% 作为验证集
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset, train_complexity_scores = process_data(train_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = process_data(eval_dataset)[0]
    eval_dataset = Dataset.from_list(eval_dataset)
elif dataset_mode == 2:
    # 使用外部的两个指定的数据集分别作为训练集和验证集
    train_data_path = paths["dataset_path"]
    test_data_path = "./all_test_data_no_multi.json"

    train_dataset = load_dataset("json", data_files=train_data_path)['train']
    eval_dataset = load_dataset("json", data_files=test_data_path)['train']

    train_dataset, train_complexity_scores = process_data(train_dataset)
    train_dataset = Dataset.from_list(train_dataset)
    eval_dataset = process_data(eval_dataset)[0]
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
    num_train_epochs=epoch_number,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8 if eval_dataset is not None else 1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    eval_strategy="epoch" if eval_dataset is not None else "no",
    logging_strategy="steps",
    logging_steps=100,
    bf16=True,
    learning_rate=1.8e-4,
    max_grad_norm=0.3,
    warmup_ratio=0.15,
    lr_scheduler_type="cosine",
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

# 修改3：增强版课程学习采样器
class CurriculumSampler(Sampler):
    def __init__(self, data_source, complexity_scores, config):
        super().__init__(data_source)
        self.complexity_scores = np.array(complexity_scores)
        self.config = config
        self.current_max = config['initial_max']
        self._precompute_masks()

    def _precompute_masks(self):
        """预计算各难度区间的掩码"""
        self.masks = {
            'easy': (self.complexity_scores <= 0.3),
            'medium': (self.complexity_scores <= 0.6),
            'hard': (self.complexity_scores <= 1.0)
        }

    def update_threshold(self, epoch):
        """改进的渐进式扩展策略"""
        if epoch < self.config['warmup_epochs']:
            return

        # 动态调整扩展速度
        progress = (epoch - self.config['warmup_epochs']) / \
                  (self.config['max_epochs'] - self.config['warmup_epochs'])

        # 前70%阶段慢速扩展，后30%快速扩展
        if progress <= 0.7:
            expansion_rate = 0.15 * (1 - 0.5*progress)
        else:
            expansion_rate = 0.3 * (1 + 2*(progress-0.7))

        self.current_max = min(
            self.current_max + expansion_rate * (1 - self.current_max),
            1.0
        )

    def __iter__(self):
        # 改进的数据混合策略
        valid_mask = self.complexity_scores <= self.current_max
        prev_mask = self.complexity_scores <= (self.current_max * self.config['decay_factor'])

        # 动态调整混合比例
        mix_ratio = 0.3 + 0.5*(self.current_max - self.config['initial_max'])/(1 - self.config['initial_max'])
        combined_mask = valid_mask | (prev_mask & (np.random.rand(len(self.complexity_scores)) < mix_ratio))

        # 确保最少覆盖min_coverage
        coverage = combined_mask.mean()
        if coverage < self.config['min_coverage']:
            required_quantile = np.quantile(self.complexity_scores, self.config['min_coverage'])
            combined_mask = self.complexity_scores <= required_quantile

        indices = np.where(combined_mask)[0]
        np.random.shuffle(indices)

        # 动态过采样策略（核心改进）
        if self.current_max < 0.8:  # 在前80%难度时进行过采样
            hard_samples = np.where(self.complexity_scores > 0.7)[0]
            add_indices = np.random.choice(hard_samples,
                                         size=int(len(indices)*0.1),  # 过采样10%
                                         replace=True)
            indices = np.concatenate([indices, add_indices])

        print(f"Curriculum: Max={self.current_max:.2f}, Samples={len(indices)}")
        return iter(indices.tolist())


# 修改CustomSFTTrainer类
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, complexity_scores=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.complexity_scores = complexity_scores
        self.curriculum_sampler = None

    def create_train_dataloader(self):
        # 初始化课程采样器
        self.curriculum_sampler = CurriculumSampler(
            self.train_dataset,
            self.complexity_scores,
            curriculum_config
        )

        return DataLoader(
            self.train_dataset,  # 新增
            batch_size=self.args.per_device_train_batch_size,
            sampler=self.curriculum_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# 新增课程回调函数
# 2. 增加数据增强回调


# 修改训练器初始化
trainer = CustomSFTTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    complexity_scores=train_complexity_scores  # 传入复杂度分数
)


# 添加回调函数
loss_callback = LossCallback(loss_data, paths["model_dir"])
trainer.add_callback(loss_callback)


# 打印完整数据分布（验证参数合理性）
total_samples = len(train_complexity_scores)
print(f"\n总样本数: {total_samples}")
print("各区间样本分布:")
bins = [0.0, 0.2, 0.5, 1.0]
for i in range(len(bins)-1):
    count = sum(bins[i] <= score < bins[i+1] for score in train_complexity_scores)
    print(f"[{bins[i]:.1f}-{bins[i+1]:.1f}): {count} ({count/total_samples*100:.1f}%)")


# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

class CurriculumMonitor(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        sampler = trainer.curriculum_sampler
        current_indices = sampler.curriculum_sampler.indices  # 获取当前采样索引

        # 计算难度分布
        scores = sampler.complexity_scores[current_indices]
        print(f"\n[Curriculum Monitor] Epoch {state.epoch}/{args.num_train_epochs}")
        print(f"Current Max Difficulty: {sampler.current_max:.2f}")
        print(f"Sampled Data Stats:")
        print(f"- Average Difficulty: {np.mean(scores):.2f}")
        print(f"- Hard Samples (>0.7): {np.sum(scores > 0.7)} ({(np.sum(scores > 0.7)/len(scores))*100:.1f}%)")
        print(f"- Coverage: {len(scores)}/{len(sampler.complexity_scores)} ({len(scores)/len(sampler.complexity_scores)*100:.1f}%)")

    def on_step_end(self, args, state, control, **kwargs):
        # 实时监控batch难度分布
        if state.global_step % 50 == 0:
            batch_scores = kwargs['logs'].get('batch_scores', [])
            if batch_scores:
                print(f"Step {state.global_step} - Batch Difficulty: "
                      f"Avg={np.mean(batch_scores):.2f}, "
                      f"Max={np.max(batch_scores):.2f}")


trainer.add_callback(CurriculumMonitor())

# save model
trainer.save_model(paths["model_dir"])

