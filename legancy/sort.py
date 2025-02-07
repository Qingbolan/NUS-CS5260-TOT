import pandas as pd
import re

def natural_sort_key(s):
    """提供自然排序的key函数，处理字符串中的数字"""
    # 将字符串中的数字部分转换为整数进行比较
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', str(s))]

def sort_csv_by_question_id(input_file='sample_submission.csv', output_file='sorted_submission.csv'):
    # 读取CSV文件
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    
    # 显示排序前的前几行
    print("\nBefore sorting:")
    print(df.head())
    
    # 按question_id排序
    df.sort_values('question_id', key=lambda x: pd.Series(x).map(natural_sort_key), inplace=True)
    
    # 显示排序后的前几行
    print("\nAfter sorting:")
    print(df.head())
    
    # 保存排序后的文件
    df.to_csv(output_file, index=False)
    print(f"\nSorted file saved as: {output_file}")
    
    # 显示总行数
    print(f"Total rows: {len(df)}")

if __name__ == "__main__":
    sort_csv_by_question_id()