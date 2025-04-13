# -*- coding: gbk -*-
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager  # ��������
import seaborn as sns
# ================================
# ȫ����������ֻ���޸�����Ĳ���
# ================================
config = {
    # ����·�����ã������ڵ�һ�����ݺϲ���
#    "base_path": r"F:\�ϳ��������ݼ�\pythonProject\all_data_experiment_36788\experiment_processing\test_results_correct_TRL\epoch16",
    "base_path": r"F:\�ϳ��������ݼ�\pythonProject\all_data_experiment_36788\experiment_processing\filter_data_epoch16_4.3",
    # ���ݺϲ�����
    "sub_folders": ["LogicBench", "LogiQA2.0", "LogiQA", "Reclor"],
    "common_folder_name": "filtered_no_repeat_output_36788_0.50_1.00_epoch16_no_eval_original_way_",
    "jsonl_file_name": "generated_predictions.jsonl",

    # ����Դ�ļ�������ڵ�ǰĿ¼��
    "source_score_file": "all_test_data��ȫ���Լ�������.json",

    # ��������
    "log_bin_size": 1,        # ComplexityScore_log�����С
    "norm_bin_size": 0.05,     # ComplexityScore_norm�����С
    "log_min": 0,             # log������Сֵ
    "log_max": 10             # log�������ֵ
}
# ================================
# �����������
# ================================

class DataProcessor:
    def __init__(self, config):
        self.config = config

        # ��������������⣨�����ԣ�
        self._set_chinese_font()

        # ��������ļ��У������ԣ�
        self.result_folder = os.path.join(os.getcwd(), config["common_folder_name"])
        print(f"\n�������Ŀ¼: {self.result_folder}")
        os.makedirs(self.result_folder, exist_ok=True)
        print(f"Ŀ¼����: {os.path.exists(self.result_folder)}")

        # �����ļ�·��
        self.merged_file = os.path.join(self.result_folder, "merged_predictions.json")
        self.scored_file = os.path.join(self.result_folder, "scores��ȫ���Լ�������.json")


    def _set_chinese_font(self):
        """�����������壨��������"""
        try:
            font_path = "C:/Windows/Fonts/simhei.ttf"
            print(f"\n���Լ�������: {font_path}")
            print(f"�����ļ�����: {os.path.exists(font_path)}")

            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print("�������óɹ�")
        except Exception as e:
            print(f"��������ʧ��: {str(e)}")
            raise

    def merge_predictions(self):
        """����1���ϲ�Ԥ���ļ�"""
        print("\n" + "="*40)
        print("���ںϲ�Ԥ���ļ�...")

        merged_data = []
        for sub_folder in self.config["sub_folders"]:
            # ����ԭʼ����·����ʹ��base_path��
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
                print(f"�Ѽ��� {sub_folder} ������")
            except Exception as e:
                print(f"���� {sub_folder} ʧ��: {str(e)}")
                continue

        # ���浽��ǰĿ¼�Ľ���ļ���
        with open(self.merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)
        print(f"�ϲ���ɣ��ļ��ѱ�����: {self.merged_file}")

    def add_scores(self):
        """����2����Ӹ��Ӷȷ���"""
        print("\n" + "="*40)
        print("������Ӹ��Ӷȷ���...")

        try:
            # ����Դ���ݣ���ǰĿ¼�µ��ļ���
            with open(self.config["source_score_file"], 'r', encoding='utf-8') as f:
                source_data = json.load(f)

            # ����Ŀ�����ݣ�����ļ����е��ļ���
            with open(self.merged_file, 'r', encoding='utf-8') as f:
                target_data = json.load(f)

            # ����У��
            if len(source_data) != len(target_data):
                raise ValueError("Դ�ļ���Ŀ���ļ�������Ŀ��һ��")

            # �ϲ�����
            for src, tgt in zip(source_data, target_data):
                tgt["ComplexityScore_log"] = src.get("ComplexityScore_log")
                tgt["ComplexityScore_norm"] = src.get("ComplexityScore_norm")

            # ���浽����ļ���
            with open(self.scored_file, 'w', encoding='utf-8') as f:
                json.dump(target_data, f, ensure_ascii=False, indent=4)
            print(f"���������ɣ��ļ��ѱ�����: {self.scored_file}")

        except Exception as e:
            print(f"��ӷ���ʱ����: {str(e)}")
            raise

    def analyze_data(self):
        """����3�����ݷ�������ӻ��������ԣ�"""
        print("\n" + "="*40)
        print("���ڽ������ݷ���...")

        try:
            # �������ݣ�����֤��
            print(f"\n���������ļ�: {self.scored_file}")
            print(f"�ļ�����: {os.path.exists(self.scored_file)}")
            with open(self.scored_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print(f"\n��������:\n{df.head()}")
            print(f"\n��������: {df.columns.tolist()}")

            # ��Ǵ���
            df['error'] = df['label'] != df['predict']
            print(f"\n��������ͳ��:\n{df['error'].value_counts()}")

            # ����log����
            self._process_score(df, 'log',
                           bins=list(range(
                               self.config["log_min"],
                               self.config["log_max"] + self.config["log_bin_size"],
                               self.config["log_bin_size"]
                           )))

            # ����norm����
            self._process_score(df, 'norm',
                           bins=[
                               x * self.config["norm_bin_size"]
                               for x in range(0, int(1/self.config["norm_bin_size"]) + 1)
                           ])


        except Exception as e:
            print(f"���ݷ���ʱ����: {str(e)}")
            raise

    def _process_score(self, df, score_type, bins):
        """�������������͵ķ����������ԣ�"""
        print(f"\n{'='*20} ���� {score_type} ���� {'='*20}")
        try:
            col_name = f"ComplexityScore_{score_type}"
            print(f"������: {col_name}")

            # ��֤�д���
            if col_name not in df.columns:
                raise ValueError(f"�� {col_name} �����ڣ�������: {df.columns.tolist()}")

            # ���������ǩ
            labels = []
            for i in range(len(bins)-1):
                if score_type == 'log':
                    labels.append(f"{bins[i]}-{bins[i+1]}")
                else:
                    labels.append(f"{bins[i]:.2f}-{bins[i+1]:.2f}")
            print(f"���ɷ����ǩ: {labels}")

            # ���ݷ���
            df['bin'] = pd.cut(df[col_name], bins=bins, labels=labels, include_lowest=True)
            print(f"\n������ͳ��:\n{df['bin'].value_counts().sort_index()}")

            # ͳ����Ϣ
            stats = df.groupby('bin', observed=False).agg(
                total=('error', 'size'),
                errors=('error', 'sum')
            ).reindex(labels, fill_value=0)
            stats['correct'] = stats['total'] - stats['errors']
            stats['error_rate'] = stats['errors'] / stats['total'].replace(0, 1)

            print(f"\nͳ����ϢԤ��:\n{stats}")

            # ����ͳ���ļ�
            self._save_stats_files(stats, col_name, labels)

            # ��������ͼ
            self._plot_error_rate(stats, col_name, labels)

        except Exception as e:
            print(f"���� {score_type} ����ʱ����: {str(e)}")
            raise

    def _save_stats_files(self, stats, col_name, labels):
        """����ͳ���ļ��������ԣ�"""
        print(f"\n{'='*20} ���� {col_name} ͳ���ļ� {'='*20}")
        try:
            # ����ͳ���ļ���ֱ�ӱ��浽���Ŀ¼��
            stats_file = os.path.join(self.result_folder, f"{col_name}_stats.txt")
            print(f"д��ͳ���ļ�: {stats_file}")
            with open(stats_file, 'w', encoding='utf-8') as f:
                for label in labels:
                    total = stats.loc[label, 'total']
                    correct = stats.loc[label, 'correct']
                    errors = stats.loc[label, 'errors']
                    f.write(f"���� {label}: ���� = {total}, ��ȷ�� = {correct}, ������ = {errors}\n")

            print(f"ͳ���ļ�������: {os.path.exists(stats_file)}")

            # �������ļ�
            rate_file = os.path.join(self.result_folder, f"{col_name}_error_rate.txt")
            print(f"д��������ļ�: {rate_file}")

            with open(rate_file, 'w', encoding='utf-8') as f:
                for label in labels:
                    rate = stats.loc[label, 'error_rate']
                    f.write(f"���� {label}: ������ = {rate:.2%}\n")

            print(f"�������ļ�������: {os.path.exists(rate_file)}")

        except Exception as e:
            print(f"�����ļ�ʧ��: {str(e)}")
            raise

    def _plot_error_rate(self, stats, col_name, labels):
        """���ƴ���������ͼ�������ԣ�"""
        print(f"\n{'='*20} ���� {col_name} ͼ�� {'='*20}")
        try:
            plt.figure(figsize=(12, 7))
            plt.plot(labels, stats['error_rate'], marker='o', linestyle='-', color='blue')

            plt.title(f"{col_name} ����������", fontsize=14)
            plt.xlabel('��������', fontsize=12)
            plt.ylabel('������', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            plot_path = os.path.join(self.result_folder, f"{col_name}_error_rate.png")
            plt.savefig(plot_path, dpi=300)
            plt.close()

            print(f"ͼ���ѱ��浽: {plot_path}")
            print(f"ͼƬ����: {os.path.exists(plot_path)}")

        except Exception as e:
            print(f"��ͼʧ��: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DataProcessor(config)

    try:
        processor.merge_predictions()
        processor.add_scores()
        processor.analyze_data()
    except Exception as e:
        print(f"\n��������з�������: {str(e)}")
    finally:
        print("\n�������̽���")
        print(f"���ս��������: {processor.result_folder}")