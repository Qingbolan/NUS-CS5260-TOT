import pandas as pd
import json
from typing import List, Dict

def save_results(results: List[Dict], output_dir: str):
    """Evaluate results and generate submission files."""
    # Generate submission CSV
    submission_df = pd.DataFrame({
        "question_id": [r["question_id"] for r in results],
        "answer": [r["predicted"] for r in results]
    })
    submission_df.to_csv(f'{output_dir}/submission.csv', index=False)
    
    # Save detailed results including CoT history
    with open(f"{output_dir}/detailed_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

import pandas as pd
from typing import List, Dict, Any

def evaluate_accuracy(predictions: List[str],
                      ground_truth: List[str],
                      output_csv: str = "evaluation_report.csv") -> Dict[str, Any]:
    """
    计算预测准确率，并生成详细的错误和整体报告，同时将错误/不匹配详情保存到 CSV 文件中。

    对于每个预测项，将预测值和真实值尝试转换为整数：
    - 如果转换失败，则记录转换错误。
    - 如果转换成功但数值不相等，则记录不匹配信息。
    - 如果转换成功且数值相等，则认为预测正确。

    打印整体报告后，返回一个包含以下信息的字典：
    - accuracy: 总体准确率
    - total: 总样本数
    - correct: 正确预测的数量
    - errors_count: 转换错误的数量
    - mismatches_count: 数值不匹配的数量
    - errors: 详细的转换错误列表
    - mismatches: 详细的不匹配列表

    同时，将所有错误和不匹配信息保存为 CSV 文件。

    Args:
        predictions (List[str]): 预测结果列表，每个元素为字符串形式的数字。
        ground_truth (List[str]): 真实结果列表，每个元素为字符串形式的数字。
        output_csv (str): 保存报告详情的 CSV 文件路径，默认为 "evaluation_report.csv"。

    Returns:
        Dict[str, Any]: 包含整体评估报告的字典。
    """
    correct = 0
    errors = []       # 存储转换失败的详细信息
    mismatches = []   # 存储转换成功但数值不匹配的详细信息

    for idx, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        try:
            pred_int = int(pred["predicted"])
        except Exception as e:
            errors.append({
                'index': truth["question_id"],
                'prediction': pred["predicted"],
                'ground_truth': truth["answer"],
                'issue': 'conversion error',
                'error_detail': f"预测值转换错误: {e}"
            })
            continue  # 跳过当前记录

        try:
            truth_int = int(truth["answer"])
        except Exception as e:
            errors.append({
                'index': truth["question_id"],
                'prediction': pred["predicted"],
                'ground_truth': truth["answer"],
                'issue': 'conversion error',
                'error_detail': f"真实值转换错误: {e}"
            })
            continue  # 跳过当前记录

        if pred_int == truth_int:
            correct += 1
        else:
            mismatches.append({
                'index': truth["question_id"],
                'prediction': pred["predicted"],
                'ground_truth': truth["answer"],
                'question': truth["question"],
                'truth': truth["solution"],
                'issue': pred["solution"],
            })

    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0

    # 构造整体报告字典
    report = {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'errors_count': len(errors),
        'mismatches_count': len(mismatches),
        'errors': errors,
        'mismatches': mismatches
    }

    # 打印整体报告
    print("Evaluation Report:")
    print(f"Total samples       : {total}")
    print(f"Correct predictions : {correct}")
    print(f"Accuracy            : {accuracy:.2%}")
    print(f"Conversion errors   : {len(errors)}")
    print(f"Mismatches          : {len(mismatches)}")

    if errors:
        print("\nConversion Errors Details:")
        for err in errors:
            print(f"Index {err['index']}: Prediction: {err['prediction']} | Ground Truth: {err['ground_truth']} | Error: {err['error_detail']}")

    if mismatches:
        print("\nMismatch Details:")
        for mismatch in mismatches:
            print(f"Index {mismatch['index']}: Prediction: {mismatch['prediction']} | Ground Truth: {mismatch['ground_truth']}")

    # 将错误和不匹配信息合并为一个列表，用于保存 CSV 文件
    issues = errors + mismatches
    if issues:
        df_issues = pd.DataFrame(issues)
        df_issues.to_csv(output_csv, index=False)
        print(f"\nDetailed issues saved to CSV file: {output_csv}")
    else:
        print("\nNo issues found; CSV file not created.")

    return report
