from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import classification_report
import torch
import json

# 保存真实标签和预测标签到 JSONL 文件
results_file = "./our_test_results/test_model_demo/prediction_results.jsonl"
# 保存分类报告到文本文件
report_file = "./our_test_results/test_model_demo/classification_report.txt"
# 加载微调后的模型
peft_model_id = "./our_model_train_results/new_2.20_template_right/original_way/score_final_data_36788_gpt_correct_sig/models"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(peft_model_id, device_map="cuda", quantization_config=bnb_config)

# 加载 Llama 3 的 tokenizer
model_id = "./fine_tuning_test/llama3-8b-instruction-hf"  # 这里使用你训练时的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 加载测试数据集
test_data_path = "./our_data/clean_data/2-sample.json"
test_dataset = load_dataset("json", data_files=test_data_path)['train']

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

def predict(text, task_type):
    encoding = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**encoding, max_new_tokens=20, temperature=0.1, do_sample=True,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    generated_ids = outputs[:, encoding.input_ids.shape[1]:]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 去除 "assistant\n\n"
    generated_text = generated_text.replace("assistant\n\n", "")

    if task_type == 'BQA':
        if "yes" in generated_text.lower():
            answer = "yes"
        elif "no" in generated_text.lower():
            answer = "no"
        else:
            answer = None
    elif task_type == 'MCQA':
        for i in range(4):
            if str(i) in generated_text:
                answer = str(i)
                break
        else:
            answer = None
    else:
        answer = None

    return answer

# 处理测试数据
test_dataset = process_data(test_dataset)

true_labels, pred_labels = [], []
for sample in test_dataset:
    message = sample["messages"]
    true_label = sample["answer"]
    true_labels.append(true_label)
    text = tokenizer.apply_chat_template(message, tokenize=False)
    task_type = sample["task_type"]
    pred_label = predict(text=text, task_type=task_type)
    pred_labels.append(pred_label)
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
