#!/usr/bin/env python3
"""Compare JSON processing methods and analyze LLM extraction issues."""

import json
from data_processing.text_processor import TextProcessor
from llm.ner_relation_extractor import NERRelationExtractor
from utils.logger import get_logger

def main():
    logger = get_logger(__name__)
    
    print("=== JSON处理方法对比分析 ===")
    print("\n1. 加载JSON数据")
    
    # Load JSON data
    with open('output-1.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    print(f"JSON结构: {list(json_data.keys())}")
    print(f"地区数量: {len(json_data.get('per_region', {}))}")
    
    # Initialize processors
    text_processor = TextProcessor()
    ner_extractor = NERRelationExtractor()
    
    print("\n2. 方法一：process_json_regulations (修复后)")
    print("-" * 50)
    
    # Method 1: Convert JSON to text using process_json_regulations
    converted_text = text_processor.process_json_regulations(json_data)
    print(f"转换后文本长度: {len(converted_text)} 字符")
    print(f"包含关键词:")
    keywords = ['建筑', '面积', '高度', '层高', 'Height', 'Evidence']
    for keyword in keywords:
        count = converted_text.count(keyword)
        print(f"  '{keyword}': {count} 次")
    
    print("\n转换后文本示例:")
    print(converted_text[:300] + "...")
    
    print("\n3. 方法二：_process_json_regulatory_text (直接结构化处理)")
    print("-" * 50)
    
    # Method 2: Direct structured processing
    try:
        structured_result = text_processor._process_json_regulatory_text('output-1.json')
        print(f"提取的实体数量: {structured_result.get('entity_count', 0)}")
        print(f"提取的法规数量: {structured_result.get('regulation_count', 0)}")
        print(f"文本块数量: {structured_result.get('chunk_count', 0)}")
        
        # Show some extracted entities
        entities = structured_result.get('entities', [])
        if entities:
            print("\n提取的实体示例:")
            for i, entity in enumerate(entities[:5]):
                print(f"  {i+1}. {entity.get('text', 'N/A')} ({entity.get('label', 'N/A')})")
    except Exception as e:
        print(f"结构化处理出错: {e}")
    
    print("\n4. LLM实体提取测试")
    print("-" * 50)
    
    # Test LLM extraction on converted text
    try:
        print("使用转换后的文本进行LLM实体提取...")
        entities = ner_extractor.extract_entities(converted_text)
        relations = ner_extractor.extract_relations(converted_text, entities)
        
        print(f"LLM提取结果:")
        print(f"  实体数量: {len(entities)}")
        print(f"  关系数量: {len(relations)}")
        
        if entities:
            print("\n提取的实体:")
            for i, entity in enumerate(entities[:5]):
                print(f"  {i+1}. {entity.text} ({entity.label}) - 置信度: {entity.confidence:.3f}")
        else:
            print("  ⚠️ 未提取到任何实体")
            
        if relations:
            print("\n提取的关系:")
            for i, relation in enumerate(relations[:3]):
                print(f"  {i+1}. {relation.subject.text} --{relation.predicate}--> {relation.object.text}")
            
    except Exception as e:
        print(f"LLM提取失败: {e}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")
    
    print("\n5. 问题分析")
    print("-" * 50)
    
    print("\n修复前的问题:")
    print("1. process_json_regulations方法无法处理per_region格式")
    print("2. JSON转换后生成空文本，导致LLM无内容可提取")
    print("3. 缺少结构化信息，LLM难以理解上下文")
    
    print("\n修复后的改进:")
    print("1. 支持per_region格式的JSON数据")
    print("2. 保留中文建筑术语和证据文本")
    print("3. 结构化输出，便于LLM理解和提取")
    print("4. 包含完整的高度规则和面积计算规则")
    
    print("\n6. 建议")
    print("-" * 50)
    print("1. 对于结构化数据，优先使用直接处理方法")
    print("2. JSON转文本时保持语义完整性")
    print("3. 为LLM提供清晰的上下文和格式")
    print("4. 结合规则方法和LLM方法提高准确性")

if __name__ == "__main__":
    main()