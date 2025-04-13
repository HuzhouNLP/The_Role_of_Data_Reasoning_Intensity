# -*- coding: gbk -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  # 新增导入
import seaborn as sns
# ================================
# 全局配置区域：只需修改这里的参数
# ================================
config = {
    # 基础路径配置（仅用于第一步数据合并）
#    "base_path": r"F:\合成数据数据集\pythonProject\all_data_experiment_36788\experiment_processing\test_results_correct_TRL\epoch16",
    "base_path": r"F:\合成数据数据集\pythonProject\all_data_experiment_36788\experiment_processing\filter_data_epoch16_4.3",
    # 数据合并配置
    "sub_folders": ["LogicBench", "LogiQA2.0", "LogiQA", "Reclor"],
    "common_folder_name": "filtered_no_repeat_output_36788_0.50_1.00_epoch16_no_eval_original_way_",
    "jsonl_file_name": "generated_predictions.jsonl",

    # 分数源文件（需放在当前目录）
    "source_score_file": "all_test_data（全测试集分数）.json",

    # 分析配置
    "log_bin_size": 1,        # ComplexityScore_log区间大小
    "norm_bin_size": 0.05,     # ComplexityScore_norm区间大小
    "log_min": 0,             # log分数最小值
    "log_max": 10             # log分数最大值
}
# ================================
# 配置区域结束
# ================================

class DataProcessor:
    def __init__(self, config):
        self.config = config

        # 解决中文字体问题（带调试）
        self._set_chinese_font()

        # 创建结果文件夹（带调试）
        self.result_folder = os.path.join(os.getcwd(), config["common_folder_name"])
        print(f"\n创建结果目录: {self.result_folder}")
        os.makedirs(self.result_folder, exist_ok=True)
        print(f"目录存在: {os.path.exists(self.result_folder)}")

        # 定义文件路径
        self.merged_file = os.path.join(self.result_folder, "merged_predictions.json")
        self.scored_file = os.path.join(self.result_folder, "scores（全测试集分数）.json")


    def _set_chinese_font(self):
        """设置中文字体（带错误处理）"""
        try:
            font_path = "C:/Windows/Fonts/simhei.ttf"
            print(f"\n尝试加载字体: {font_path}")
            print(f"字体文件存在: {os.path.exists(font_path)}")

            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print("字体设置成功")
        except Exception as e:
            print(f"字体设置失败: {str(e)}")
            raise

    def merge_predictions(self):
        """步骤1：合并预测文件"""
        print("\n" + "="*40)
        print("正在合并预测文件...")

        merged_data = []
        for sub_folder in self.config["sub_folders"]:
            # 构建原始数据路径（使用base_path）
            jsonl_path = os.path.join(
                self.config["base_path"],
                sub_folder,
                self.config["common_folder_name"],
                self.config["jsonl_file_name"]
            )

            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        merged_data.append(json.loads(line))
                print(f"已加载 {sub_folder} 的数据")
            except Exception as e:
                print(f"加载 {sub_folder} 失败: {str(e)}")
                continue

        # 保存到当前目录的结果文件夹
        with open(self.merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        print(f"合并完成，文件已保存至: {self.merged_file}")

    def add_scores(self):
        """步骤2：添加复杂度分数"""
        print("\n" + "="*40)
        print("正在添加复杂度分数...")

        try:
            # 加载源数据（当前目录下的文件）
            with open(self.config["source_score_file"], 'r', encoding='utf-8') as f:
                source_data = json.load(f)

            # 加载目标数据（结果文件夹中的文件）
            with open(self.merged_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)

            # 数据校验
            if len(source_data) != len(target_data):
                raise ValueError("源文件和目标文件数据条目不一致")

            # 合并数据
            for src, tgt in zip(source_data, target_data):
                tgt["ComplexityScore_log"] = src.get("ComplexityScore_log")
                tgt["ComplexityScore_norm"] = src.get("ComplexityScore_norm")

            # 保存到结果文件夹
            with open(self.scored_file, 'w', encoding='utf-8') as f:
                json.dump(target_data, f, ensure_ascii=False, indent=4)
            print(f"分数添加完成，文件已保存至: {self.scored_file}")

        except Exception as e:
            print(f"添加分数时出错: {str(e)}")
            raise

    def analyze_data(self):
        """步骤3：数据分析与可视化（带调试）"""
        print("\n" + "="*40)
        print("正在进行数据分析...")

        try:
            # 加载数据（带验证）
            print(f"\n加载数据文件: {self.scored_file}")
            print(f"文件存在: {os.path.exists(self.scored_file)}")
            with open(self.scored_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"\n数据样本:\n{df.head()}")
            print(f"\n数据列名: {df.columns.tolist()}")

            # 标记错误
            df['error'] = df['label'] != df['predict']
            print(f"\n错误数量统计:\n{df['error'].value_counts()}")

            # 处理log分数
            self._process_score(df, 'log',
                           bins=list(range(
                               self.config["log_min"],
                               self.config["log_max"] + self.config["log_bin_size"],
                               self.config["log_bin_size"]
                           )))

            # 处理norm分数
            self._process_score(df, 'norm',
                           bins=[
                               x * self.config["norm_bin_size"]
                               for x in range(0, int(1/self.config["norm_bin_size"]) + 1)
                           ])


        except Exception as e:
            print(f"数据分析时出错: {str(e)}")
            raise

    def _process_score(self, df, score_type, bins):
        """处理单个分数类型的分析（带调试）"""
        print(f"\n{'='*20} 处理 {score_type} 分数 {'='*20}")
        try:
            col_name = f"ComplexityScore_{score_type}"
            print(f"分析列: {col_name}")

            # 验证列存在
            if col_name not in df.columns:
                raise ValueError(f"列 {col_name} 不存在！可用列: {df.columns.tolist()}")

            # 生成区间标签
            labels = []
            for i in range(len(bins)-1):
                if score_type == 'log':
                    labels.append(f"{bins[i]}-{bins[i+1]}")
                else:
                    labels.append(f"{bins[i]:.2f}-{bins[i+1]:.2f}")
            print(f"生成分箱标签: {labels}")

            # 数据分箱
            df['bin'] = pd.cut(df[col_name], bins=bins, labels=labels, include_lowest=True)
            print(f"\n分箱结果统计:\n{df['bin'].value_counts().sort_index()}")

            # 统计信息
            stats = df.groupby('bin', observed=False).agg(
                total=('error', 'size'),
                errors=('error', 'sum')
            ).reindex(labels, fill_value=0)
            stats['correct'] = stats['total'] - stats['errors']
            stats['error_rate'] = stats['errors'] / stats['total'].replace(0, 1)

            print(f"\n统计信息预览:\n{stats}")

            # 生成统计文件
            self._save_stats_files(stats, col_name, labels)

            # 绘制折线图
            self._plot_error_rate(stats, col_name, labels)

        except Exception as e:
            print(f"处理 {score_type} 分数时出错: {str(e)}")
            raise

    def _save_stats_files(self, stats, col_name, labels):
        """保存统计文件（带调试）"""
        print(f"\n{'='*20} 保存 {col_name} 统计文件 {'='*20}")
        try:
            # 区间统计文件（直接保存到结果目录）
            stats_file = os.path.join(self.result_folder, f"{col_name}_stats.txt")
            print(f"写入统计文件: {stats_file}")
            with open(stats_file, 'w', encoding='utf-8') as f:
                for label in labels:
                    total = stats.loc[label, 'total']
                    correct = stats.loc[label, 'correct']
                    errors = stats.loc[label, 'errors']
                    f.write(f"区间 {label}: 总数 = {total}, 正确数 = {correct}, 错误数 = {errors}\n")

            print(f"统计文件已生成: {os.path.exists(stats_file)}")

            # 错误率文件
            rate_file = os.path.join(self.result_folder, f"{col_name}_error_rate.txt")
            print(f"写入错误率文件: {rate_file}")

            with open(rate_file, 'w', encoding='utf-8') as f:
                for label in labels:
                    rate = stats.loc[label, 'error_rate']
                    f.write(f"区间 {label}: 错误率 = {rate:.2%}\n")

            print(f"错误率文件已生成: {os.path.exists(rate_file)}")

        except Exception as e:
            print(f"保存文件失败: {str(e)}")
            raise

    def _plot_error_rate(self, stats, col_name, labels):
        """绘制错误率折线图（带调试）"""
        print(f"\n{'='*20} 绘制 {col_name} 图表 {'='*20}")
        try:
            plt.figure(figsize=(12, 7))
            plt.plot(labels, stats['error_rate'], marker='o', linestyle='-', color='blue')

            plt.title(f"{col_name} 错误率趋势", fontsize=14)
            plt.xlabel('分数区间', fontsize=12)
            plt.ylabel('错误率', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(self.result_folder, f"{col_name}_error_rate.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"图表已保存到: {plot_path}")
            print(f"图片存在: {os.path.exists(plot_path)}")

        except Exception as e:
            print(f"绘图失败: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DataProcessor(config)

    try:
        processor.merge_predictions()
        processor.add_scores()
        processor.analyze_data()
    except Exception as e:
        print(f"\n处理过程中发生错误: {str(e)}")
    finally:
        print("\n处理流程结束")
        print(f"最终结果保存在: {processor.result_folder}")