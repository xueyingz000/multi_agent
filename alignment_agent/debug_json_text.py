#!/usr/bin/env python3
"""Debug script to analyze JSON to text conversion."""

import json
from data_processing.text_processor import TextProcessor

def main():
    # Initialize text processor
    tp = TextProcessor()
    
    # Load JSON data
    with open('output-1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to text
    text = tp.process_json_regulations(data)
    
    print("=== JSON转换后的文本格式 ===")
    print(text[:2000])
    print("\n=== 文本统计信息 ===")
    print(f"总长度: {len(text)} 字符")
    lines = text.split('\n')
    print(f"行数: {len(lines)}")
    
    print("\n=== 前15行内容 ===")
    for i, line in enumerate(lines[:15]):
        print(f"{i+1:2d}: {line}")
    
    print("\n=== 文本结构分析 ===")
    # 分析文本中的关键词
    keywords = ['建筑', '高度', '面积', '规定', '要求', 'Section', 'Requirement']
    for keyword in keywords:
        count = text.count(keyword)
        print(f"'{keyword}' 出现次数: {count}")
    
    # 检查是否有空行或格式问题
    empty_lines = sum(1 for line in lines if not line.strip())
    print(f"空行数量: {empty_lines}")
    
    # 检查平均行长度
    non_empty_lines = [line for line in lines if line.strip()]
    if non_empty_lines:
        avg_length = sum(len(line) for line in non_empty_lines) / len(non_empty_lines)
        print(f"非空行平均长度: {avg_length:.1f} 字符")
    
    print("\n=== 原始JSON结构分析 ===")
    print(f"JSON keys: {list(data.keys())}")
    if 'sections' in data:
        print(f"Sections数量: {len(data['sections'])}")
        if data['sections']:
            first_section = data['sections'][0]
            print(f"第一个section的keys: {list(first_section.keys())}")

if __name__ == "__main__":
    main()