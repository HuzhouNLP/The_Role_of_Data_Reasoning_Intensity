import json
import math
import re

# 这个代码是用来计算总的分数的，包括context和options
# 使用LayerNorm_Sigmoid而非Min-Max做最终归一化
# 值分别是1,0,10的-5次方


#input_json_file = "LogicBench_test_2020.json"
#input_json_file = "LogiQA2.0_test_1572.json"
#input_json_file = "LogiQA_test_651.json"
#input_json_file = "reclor_dev_500.json"
input_json_file = "all_test_data.json"

#output_json_file = "./sig_score_test/LogicBench_test_2020_norm_sig.json"
#output_json_file = "./sig_score_test/LogiQA2.0_test_1572_norm_sig.json"
#output_json_file = "./sig_score_test/LogiQA_test_651_norm_sig.json"
#output_json_file = "./sig_score_test/reclor_dev_500_norm_sig.json"
output_json_file = "./sig_score_test/all_test_data.json"

def compute_nesting_depth(expression):
    """
    计算单个逻辑表达式的嵌套深度。
    若无任何括号，则记为 1。
    """
    max_depth = 0
    current_depth = 0
    for char in expression:
        if char in ('(', '[', '{'):
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char in (')', ']', '}'):
            current_depth -= 1
    return max_depth if max_depth > 0 else 1

def compute_average_nesting_depth(expressions):
    """
    计算若干表达式的平均嵌套深度。
    """
    if not expressions:
        return 1  # 没有表达式时，默认深度设为 1
    depths = []
    for expr in expressions:
        if not isinstance(expr, str):
            continue  # 跳过非字符串类型的表达式
        # 去除可能的反引号、前后多余空格
        expr = expr.strip("`'\" ").strip()
        d = compute_nesting_depth(expr)
        depths.append(d)
    return sum(depths) / len(depths) if depths else 1

def count_logic_operators(expression):
    """
    简单统计逻辑运算符的出现次数：
    包括 ->, ∧, ∨, ¬ (或 ~), 以及其他可选符号。
    """
    # 先去掉反引号
    expr = expression.strip("`'\" ").strip()
    # 为了统计 "->"，将其替换成单个符号
    expr = expr.replace("->", "→")

    count = 0
    for op in ["∧", "∨", "→", "¬", "~", "←"]:
        count += expr.count(op)

    return count

def compute_context_score(item):
    """
    计算单条数据的 Context 分数：
    score_context = E × (D^2) + P + C
    E = Logical Expressions 的数量
    D = Average Nesting Depth
    P = Predicates 数量
    C = Constants 数量
    """
    E = len(item.get('Logical Expressions', []))
    D = compute_average_nesting_depth(item.get('Logical Expressions', []))
    P = len(item.get('Predicates', []))
    C_val = len(item.get('Constants', []))

    score_context = E * (D ** 2) + P + C_val
    return score_context

def compute_preconditions_score(preconditions):
    """
    对一个选项的 preconditions 计算分数：
    score_pre = E_pre * (D_avg_pre^2)
    - E_pre: preconditions 的表达式个数
    - D_avg_pre: 这些表达式的平均嵌套深度
    """
    if not isinstance(preconditions, list):
        preconditions = []

    E_pre = len(preconditions)
    if E_pre == 0:
        return 0

    D_avg_pre = compute_average_nesting_depth(preconditions)
    score_pre = E_pre * (D_avg_pre ** 2)
    return score_pre

def compute_deduction_steps_score(deduction_steps):
    """
    计算一个选项的 deduction steps 总分：
      对每个 step:
        若 expression 为 "N/A" 或 "Derivation cannot proceed" => 0
        否则 => (D^2) * (LogicOps + 1)
          - D = 嵌套深度
          - LogicOps = 逻辑运算符数量
    最后把各 step 分数相加。
    """
    if not isinstance(deduction_steps, list):
        deduction_steps = []

    total_score = 0
    for step in deduction_steps:
        if not isinstance(step, dict):
            continue
        expr = step.get("expression", "")
        if not isinstance(expr, str):
            expr = ""
        expr = expr.strip()

        if expr in ["N/A", "Derivation cannot proceed"]:
            step_score = 0
        else:
            d = compute_nesting_depth(expr)
            logic_ops = count_logic_operators(expr)
            step_score = (d ** 2) * (logic_ops + 1)
        total_score += step_score

    return total_score

def compute_options_score(item):
    """
    对一条数据的所有选项 (option_analysis) 计算分数：
      对每个选项 i:
        score_option_i = score_pre_i + score_deduct_i
      最后所有选项相加。
    """
    option_analysis = item.get("option_analysis", [])
    if not isinstance(option_analysis, list):
        option_analysis = []

    total_score_options = 0
    for opt in option_analysis:
        if not isinstance(opt, dict):
            continue

        # 1) 计算 preconditions 分数
        preconditions = opt.get("preconditions", [])
        score_pre = compute_preconditions_score(preconditions)

        # 2) 计算 deduction steps 分数
        deduction_steps = opt.get("deduction_steps", [])
        score_deduction = compute_deduction_steps_score(deduction_steps)

        # 选项分数
        score_option = score_pre + score_deduction
        total_score_options += score_option

    return total_score_options

def compute_item_score(item):
    """
    计算单条数据的 (Context + Options) 综合分数，然后做 log。
    """
    # 先计算 context 分数
    context_score = compute_context_score(item)
    # 再计算 options 分数
    options_score = compute_options_score(item)
    # 合并
    total_score = context_score + options_score

    # 避免 total_score 为负数或 0，确保 log 计算安全
    if total_score < 0:
        total_score = 0
    final_score = math.log(total_score + 1)

    return final_score

def main():
    # 读取数据
    with open(input_json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON 解码错误: {e}")
            return

    if not isinstance(data, list):
        print("JSON 根对象并非列表，请根据实际数据结构进行适配。")
        return

    # 逐条计算对数分数
    raw_scores = []
    for idx, item in enumerate(data):
        try:
            score = compute_item_score(item)
            raw_scores.append(score)
        except Exception as e:
            print(f"处理第 {idx} 条数据时出错: {e}")
            raw_scores.append(0)  # 出错时赋予最低分

    # ============ 使用 LayerNorm_Sigmoid 而非 Min-Max =============
    # 1) 计算对数分数的均值和方差
    eps = 1e-5
    n = len(raw_scores)
    if n < 1:
        print("数据为空，无法处理。")
        return

    mean_score = sum(raw_scores) / n
    var_score = sum((s - mean_score) ** 2 for s in raw_scores) / n
    std_score = math.sqrt(var_score + eps)

    # 2) gamma=1, beta=0 (可根据需要调整)
    gamma = 1.0
    beta = 0.0

    # 3) 对每个得分应用 LayerNorm_Sigmoid
    def layernorm_sigmoid(x):
        # z = (x - mean) / std
        z = (x - mean_score) / std_score
        # y = sigmoid(gamma * z + beta) = 1 / (1 + e^(-(gamma*z + beta)))
        val = 1.0 / (1.0 + math.exp(-(gamma * z + beta)))
        return val

    normalized_scores = [layernorm_sigmoid(s) for s in raw_scores]
    # ===============================================================

    # 将分数写回数据
    for i, item in enumerate(data):
        # 对数分数
        item["ComplexityScore_log"] = float(raw_scores[i])
        # LayerNorm_Sigmoid归一化分数
        item["ComplexityScore_norm"] = float(normalized_scores[i])

    # 输出到文件
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"处理完成！已输出到 {output_json_file}")

if __name__ == "__main__":
    main()
