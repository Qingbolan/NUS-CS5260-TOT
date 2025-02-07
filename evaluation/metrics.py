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
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.emoji import Emoji
from datetime import datetime
import os

def evaluate_accuracy(predictions: List[str],
                     ground_truth: List[str],
                     report_dir: str = "report") -> Dict[str, Any]:
    """
    计算预测准确率，并生成彩色详细报告，将评估结果保存到带时间戳的文件中。
    
    Args:
        predictions (List[str]): 预测结果列表
        ground_truth (List[str]): 真实结果列表
        report_dir (str): 报告保存目录，默认为 "report"
    
    Returns:
        Dict[str, Any]: 包含整体评估报告的字典
    """
    # 创建Rich控制台对象
    console = Console()
    
    # 确保报告目录存在
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(report_dir, f"evaluation_report_{timestamp}.csv")
    
    correct = 0
    errors = []
    mismatches = []

    # 评估逻辑
    for idx, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        try:
            pred_int = int(float(pred["predicted"].replace(",", "")))
        except Exception as e:
            errors.append({
                'index': truth["question_id"],
                'prediction': pred["predicted"],
                'ground_truth': truth["answer"],
                'issue': 'conversion error',
                'error_detail': f"预测值转换错误: {e}"
            })
            continue

        try:
            truth_int = int(float(truth["answer"].replace(",", "")))
        except Exception as e:
            errors.append({
                'index': truth["question_id"],
                'prediction': pred["predicted"],
                'ground_truth': truth["answer"],
                'issue': 'conversion error',
                'error_detail': f"真实值转换错误: {e}"
            })
            continue

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

    # 构造报告字典
    report = {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'errors_count': len(errors),
        'mismatches_count': len(mismatches),
        'errors': errors,
        'mismatches': mismatches
    }

    # 使用Rich创建精美的控制台输出
    console.print("\n[bold cyan]📊 评估报告[/bold cyan]", justify="center")
    
    # 创建主要指标表格
    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("指标", style="cyan")
    metrics_table.add_column("数值", justify="right", style="green")
    
    metrics_table.add_row("总样本数", f"{total} 📝")
    metrics_table.add_row("正确预测", f"{correct} ✅")
    metrics_table.add_row("准确率", f"{accuracy:.2%} 🎯")
    metrics_table.add_row("转换错误", f"{len(errors)} ❌")
    metrics_table.add_row("预测不匹配", f"{len(mismatches)} ⚠️")
    
    console.print(Panel(metrics_table, title="主要评估指标", border_style="cyan"))

    # 打印错误详情
    if errors:
        console.print("\n[bold red]❌ 转换错误详情[/bold red]")
        error_table = Table(show_header=True, header_style="bold red")
        error_table.add_column("索引")
        error_table.add_column("预测值")
        error_table.add_column("真实值")
        error_table.add_column("错误详情")
        
        for err in errors:
            error_table.add_row(
                str(err['index']),
                err['prediction'],
                err['ground_truth'],
                err['error_detail']
            )
        console.print(error_table)

    # 打印不匹配详情
    if mismatches:
        console.print("\n[bold yellow]⚠️ 预测不匹配详情[/bold yellow]")
        mismatch_table = Table(show_header=True, header_style="bold yellow")
        mismatch_table.add_column("索引")
        mismatch_table.add_column("预测值")
        mismatch_table.add_column("真实值")
        mismatch_table.add_column("问题")
        
        for mismatch in mismatches:
            mismatch_table.add_row(
                str(mismatch['index']),
                mismatch['prediction'],
                mismatch['ground_truth'],
                mismatch['question']
            )
        console.print(mismatch_table)

    # 保存详细信息到CSV
    issues = errors + mismatches
    if issues:
        df_issues = pd.DataFrame(issues)
        df_issues.to_csv(output_csv, index=False)
        console.print(f"\n[green]📁 详细问题报告已保存至: {output_csv}[/green]")
    else:
        console.print("\n[green]✨ 未发现任何问题，无需创建CSV文件[/green]")

    return report