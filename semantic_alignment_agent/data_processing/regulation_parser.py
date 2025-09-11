from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils import (
    log, RegulationRule, RegulationCategory, Evidence
)


class RegulationParser:
    """法规数据解析器，解析regulation analysis agent的输出"""
    
    def __init__(self):
        self.regulation_mapping = {
            'full': (RegulationCategory.INCLUDE_FULL, 1.0),
            'half': (RegulationCategory.INCLUDE_PARTIAL, 0.5),
            'excluded': (RegulationCategory.EXCLUDE, 0.0),
            'conditional': (RegulationCategory.CONDITIONAL, 0.5),  # 默认0.5，需要进一步分析
            'unknown': (RegulationCategory.UNKNOWN, 0.0)
        }
    
    def parse_regulation_output(self, regulation_data: Dict[str, Any], target_region: str = None) -> List[RegulationRule]:
        """解析regulation analysis agent的输出"""
        rules = []
        
        try:
            per_region = regulation_data.get('per_region', {})
            
            # 如果指定了目标区域，只解析该区域
            if target_region and target_region in per_region:
                regions_to_process = {target_region: per_region[target_region]}
            else:
                regions_to_process = per_region
            
            for region, region_data in regions_to_process.items():
                log.info(f"Parsing regulation rules for region: {region}")
                
                # 解析高度规则
                height_rules = self._parse_height_rules(region, region_data.get('height_rules', []))
                rules.extend(height_rules)
                
                # 解析覆盖/围护规则
                cover_rules = self._parse_feature_rules(
                    region, 'cover_enclosure', region_data.get('cover_enclosure_rules', [])
                )
                rules.extend(cover_rules)
                
                # 解析特殊用途规则
                special_rules = self._parse_feature_rules(
                    region, 'special_use', region_data.get('special_use_rules', [])
                )
                rules.extend(special_rules)
            
            log.info(f"Parsed {len(rules)} regulation rules")
            return rules
            
        except Exception as e:
            log.error(f"Error parsing regulation output: {e}")
            return []
    
    def _parse_height_rules(self, region: str, height_rules: List[Dict[str, Any]]) -> List[RegulationRule]:
        """解析高度规则"""
        rules = []
        
        for rule_data in height_rules:
            try:
                label = rule_data.get('label', 'unknown')
                category, coefficient = self.regulation_mapping.get(label, 
                    (RegulationCategory.UNKNOWN, 0.0))
                
                # 构建条件
                conditions = {
                    'comparator': rule_data.get('comparator'),
                    'threshold_min_m': rule_data.get('threshold_min_m'),
                    'threshold_max_m': rule_data.get('threshold_max_m')
                }
                
                # 构建证据
                evidence = []
                for ev_data in rule_data.get('evidence', []):
                    if isinstance(ev_data, dict):
                        evidence.append(Evidence(
                            text=ev_data.get('text', ''),
                            source='regulation_analysis_agent',
                            start=ev_data.get('start', -1),
                            end=ev_data.get('end', -1)
                        ))
                    elif isinstance(ev_data, str):
                        evidence.append(Evidence(
                            text=ev_data,
                            source='regulation_analysis_agent'
                        ))
                
                rule = RegulationRule(
                    region=region,
                    rule_type='height',
                    feature_key=f"height_{label}",
                    label=label,
                    coefficient=coefficient,
                    conditions=conditions,
                    evidence=evidence
                )
                
                rules.append(rule)
                
            except Exception as e:
                log.warning(f"Error parsing height rule: {e}")
        
        return rules
    
    def _parse_feature_rules(self, region: str, rule_type: str, feature_rules: List[Dict[str, Any]]) -> List[RegulationRule]:
        """解析特征规则（覆盖/围护规则和特殊用途规则）"""
        rules = []
        
        for rule_data in feature_rules:
            try:
                feature_key = rule_data.get('feature_key') or rule_data.get('feature', 'unknown')
                label = rule_data.get('label', 'unknown')
                notes = rule_data.get('notes', '')
                
                category, coefficient = self.regulation_mapping.get(label, 
                    (RegulationCategory.UNKNOWN, 0.0))
                
                # 构建条件
                conditions = {'notes': notes}
                
                # 构建证据
                evidence = []
                for ev_data in rule_data.get('evidence', []):
                    if isinstance(ev_data, dict):
                        evidence.append(Evidence(
                            text=ev_data.get('text', ''),
                            source='regulation_analysis_agent',
                            start=ev_data.get('start', -1),
                            end=ev_data.get('end', -1)
                        ))
                    elif isinstance(ev_data, str):
                        evidence.append(Evidence(
                            text=ev_data,
                            source='regulation_analysis_agent'
                        ))
                
                rule = RegulationRule(
                    region=region,
                    rule_type=rule_type,
                    feature_key=feature_key,
                    label=label,
                    coefficient=coefficient,
                    conditions=conditions,
                    evidence=evidence
                )
                
                rules.append(rule)
                
            except Exception as e:
                log.warning(f"Error parsing {rule_type} rule: {e}")
        
        return rules
    
    def load_from_file(self, file_path: str, target_region: str = None) -> List[RegulationRule]:
        """从文件加载法规数据"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Regulation file not found: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self.parse_regulation_output(data, target_region)
            
        except Exception as e:
            log.error(f"Error loading regulation file {file_path}: {e}")
            return []
    
    def get_rules_by_type(self, rules: List[RegulationRule], rule_type: str) -> List[RegulationRule]:
        """按类型筛选规则"""
        return [rule for rule in rules if rule.rule_type == rule_type]
    
    def get_rules_by_feature(self, rules: List[RegulationRule], feature_key: str) -> List[RegulationRule]:
        """按特征键筛选规则"""
        return [rule for rule in rules if rule.feature_key == feature_key]
    
    def get_height_thresholds(self, rules: List[RegulationRule]) -> Dict[str, Dict[str, float]]:
        """获取高度阈值信息"""
        height_rules = self.get_rules_by_type(rules, 'height')
        thresholds = {}
        
        for rule in height_rules:
            conditions = rule.conditions
            comparator = conditions.get('comparator')
            min_m = conditions.get('threshold_min_m')
            max_m = conditions.get('threshold_max_m')
            
            if rule.label not in thresholds:
                thresholds[rule.label] = {}
            
            if comparator and (min_m is not None or max_m is not None):
                thresholds[rule.label][comparator] = {
                    'min_m': min_m,
                    'max_m': max_m,
                    'coefficient': rule.coefficient
                }
        
        return thresholds
    
    def create_feature_mapping(self, rules: List[RegulationRule]) -> Dict[str, Dict[str, Any]]:
        """创建特征映射表"""
        mapping = {}
        
        for rule in rules:
            if rule.rule_type in ['cover_enclosure', 'special_use']:
                feature_key = rule.feature_key
                if feature_key not in mapping:
                    mapping[feature_key] = {
                        'category': rule.label,
                        'coefficient': rule.coefficient,
                        'rule_type': rule.rule_type,
                        'conditions': rule.conditions,
                        'evidence': rule.evidence
                    }
                else:
                    # 如果有多个规则，选择优先级更高的
                    priority_order = ['excluded', 'half', 'full', 'conditional', 'unknown']
                    current_priority = priority_order.index(mapping[feature_key]['category']) if mapping[feature_key]['category'] in priority_order else len(priority_order)
                    new_priority = priority_order.index(rule.label) if rule.label in priority_order else len(priority_order)
                    
                    if new_priority < current_priority:
                        mapping[feature_key] = {
                            'category': rule.label,
                            'coefficient': rule.coefficient,
                            'rule_type': rule.rule_type,
                            'conditions': rule.conditions,
                            'evidence': rule.evidence
                        }
        
        return mapping
    
    def validate_rules(self, rules: List[RegulationRule]) -> Dict[str, List[str]]:
        """验证规则的完整性和一致性"""
        issues = {
            'warnings': [],
            'errors': []
        }
        
        # 检查高度规则的完整性
        height_rules = self.get_rules_by_type(rules, 'height')
        height_labels = {rule.label for rule in height_rules}
        expected_labels = {'full', 'half', 'excluded'}
        
        missing_labels = expected_labels - height_labels
        if missing_labels:
            issues['warnings'].append(f"Missing height rule labels: {missing_labels}")
        
        # 检查阈值的逻辑性
        for rule in height_rules:
            conditions = rule.conditions
            min_m = conditions.get('threshold_min_m')
            max_m = conditions.get('threshold_max_m')
            
            if min_m is not None and max_m is not None and min_m >= max_m:
                issues['errors'].append(f"Invalid threshold range for {rule.label}: min={min_m}, max={max_m}")
        
        # 检查特征规则的重复
        feature_counts = {}
        for rule in rules:
            if rule.rule_type in ['cover_enclosure', 'special_use']:
                key = f"{rule.rule_type}_{rule.feature_key}"
                feature_counts[key] = feature_counts.get(key, 0) + 1
        
        for key, count in feature_counts.items():
            if count > 1:
                issues['warnings'].append(f"Duplicate feature rules for {key}: {count} rules")
        
        return issues