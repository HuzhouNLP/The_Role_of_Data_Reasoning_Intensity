from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report
import torch
import json
import os

# 加载微调后的模型
peft_model_id = "{{ADAPTER_DIR}}/{{ADAPTER_SUFFIX}}/models"
# peft_model_id = "{{ADAPTER_DIR}}/{{ADAPTER_SUFFIX}}"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="cuda", quantization_config=bnb_config)

# 加载 Llama 3 的 tokenizer
model_id = "./llama3-8b-instruction-hf"  # 这里使用你训练时的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载测试数据集
test_data_path = "{{TEST_DATA_DIR}}/{{DATASET_NAME}}"
test_dataset = load_dataset("json", data_files=test_data_path)['train']

# 创建保存结果的文件夹
output_dir = "{{OUTPUT_DIR}}/{{CHECK_POINTS}}{{ADAPTER_SUFFIX}}_{{WAYS}}"
os.makedirs(output_dir, exist_ok=True)

# 保存真实标签和预测标签到 JSONL 文件
results_file = f"{output_dir}/generated_predictions.jsonl"
# 保存分类报告到文本文件
report_file = f"{output_dir}/classification_report.txt"

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
        ]
        processed_data.append({"messages": messages, "answer": sample["answer"], "task_type": task_type})
    return processed_data

def predict_batch(texts, task_types, batch_size=16):
    all_pred_labels = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_task_types = task_types[i:i+batch_size]
        encoding = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        outputs = model.generate(**encoding, max_new_tokens=20, do_sample=True, temperature=0.0000001,
                                 eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        generated_ids = outputs[:, encoding.input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for j, generated_text in enumerate(generated_texts):
            # 去除 "assistant\n\n"
            generated_text = generated_text.replace("assistant\n\n", "")
            task_type = batch_task_types[j]
            if task_type == 'BQA':
                if "yes" in generated_text.lower():
                    answer = "yes"
                elif "no" in generated_text.lower():
                    answer = "no"
                else:
                    answer = None
            elif task_type == 'MCQA':
                for k in range(4):
                    if str(k) in generated_text:
                        answer = str(k)
                        break
                else:
                    answer = None
            else:
                answer = None
            all_pred_labels.append(answer)
    return all_pred_labels

# 处理测试数据
test_dataset = process_data(test_dataset)

true_labels, pred_labels = [], []
texts = []
task_types = []
for sample in test_dataset:
    message = sample["messages"]
    true_label = sample["answer"]
    # 将真实标签转换为小写
    true_label = true_label.lower() if isinstance(true_label, str) else true_label
    true_labels.append(true_label)
    text = tokenizer.apply_chat_template(message, tokenize=False)
    texts.append(text)
    task_type = sample["task_type"]
    task_types.append(task_type)

pred_labels = predict_batch(texts, task_types)
# 将预测标签转换为小写
pred_labels = [label.lower() if isinstance(label, str) else label for label in pred_labels]

for true_label, pred_label in zip(true_labels, pred_labels):
    print(f"True Label: {true_label}, Pred Label: {pred_label}")

with open(results_file, "w") as f:
    for true_label, pred_label in zip(true_labels, pred_labels):
        result = {"label": true_label, "predict": pred_label}
        f.write(json.dumps(result) + "\n")

# 计算评估指标
report = classification_report(y_true=true_labels, y_pred=pred_labels, digits=4)
print(report)

with open(report_file, "w") as f:
    f.write(report)
