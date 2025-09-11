from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from ..utils import (
    log, Point3D, BoundingBox, IfcElementInfo, VerticalSpaceInfo, 
    VerticalSpaceType, Evidence
)
from ..geometry import SpatialContext


@dataclass
class DetectionCandidate:
    """检测候选项"""
    elements: List[IfcElementInfo]
    detection_method: str
    confidence: float
    geometric_continuity: float
    vertical_span: Dict[str, Any]
    consolidated_bbox: BoundingBox


class VerticalSpaceDetector:
    """垂直贯穿空间检测器"""
    
    def __init__(self):
        # 检测参数
        self.detection_params = {
            'horizontal_tolerance': 0.2,  # 水平位置容差 200mm
            'cross_section_consistency': 0.7,  # 截面一致性阈值
            'min_floors_spanned': 2,  # 最小跨越楼层数
            'typical_floor_height': 3.0,  # 典型层高
            'overlap_threshold': 0.8  # 重叠阈值
        }
        
        # 实体类型优先级（用于去重）
        self.entity_priority = {
            'IfcSpace': 1,
            'IfcOpeningElement': 2,
            'IfcVoid': 3,
            'IfcSlab': 4,
            'IfcBuildingElementProxy': 5
        }
        
        # 垂直空间类型关键词
        self.space_type_keywords = {
            VerticalSpaceType.ATRIUM: ['atrium', 'lobby', 'hall', '中庭', '大厅', '门厅'],
            VerticalSpaceType.SHAFT: ['shaft', 'duct', 'pipe', '竖井', '管井', '风井'],
            VerticalSpaceType.STAIRWELL: ['stair', 'staircase', 'stairwell', '楼梯', '楼梯间'],
            VerticalSpaceType.ELEVATOR_SHAFT: ['elevator', 'lift', '电梯', '电梯井'],
            VerticalSpaceType.VOID: ['void', 'opening', '空洞', '开洞']
        }
    
    def detect_vertical_spaces(self, all_elements: List[IfcElementInfo],
                             spatial_context: Optional[SpatialContext] = None) -> List[VerticalSpaceInfo]:
        """检测垂直贯穿空间"""
        try:
            log.info("Starting comprehensive vertical space detection")
            
            # Step 1: 多实体扫描策略
            candidates = self._multi_entity_scanning(all_elements)
            
            # Step 2: 几何连续性验证
            validated_candidates = self._validate_geometric_continuity(candidates)
            
            # Step 3: 整合与去重
            consolidated_spaces = self._consolidate_and_deduplicate(validated_candidates)
            
            # Step 4: 分类和最终验证
            vertical_spaces = self._classify_and_finalize(consolidated_spaces, spatial_context)
            
            log.info(f"Detected {len(vertical_spaces)} vertical penetrating spaces")
            return vertical_spaces
            
        except Exception as e:
            log.error(f"Error in vertical space detection: {e}")
            return []
    
    def _multi_entity_scanning(self, all_elements: List[IfcElementInfo]) -> List[DetectionCandidate]:
        """多实体扫描策略"""
        candidates = []
        
        # 1. 直接IfcSpace检测
        space_candidates = self._detect_direct_ifcspace(all_elements)
        candidates.extend(space_candidates)
        
        # 2. IfcOpeningElement分析
        opening_candidates = self._detect_opening_elements(all_elements)
        candidates.extend(opening_candidates)
        
        # 3. 隐式空洞检测
        void_candidates = self._detect_implicit_voids(all_elements)
        candidates.extend(void_candidates)
        
        # 4. 边界分析
        boundary_candidates = self._detect_boundary_discontinuities(all_elements)
        candidates.extend(boundary_candidates)
        
        log.debug(f"Multi-entity scanning found {len(candidates)} candidates")
        return candidates
    
    def _detect_direct_ifcspace(self, all_elements: List[IfcElementInfo]) -> List[DetectionCandidate]:
        """直接IfcSpace检测"""
        candidates = []
        
        spaces = [elem for elem in all_elements if elem.ifc_type == 'IfcSpace']
        
        for space in spaces:
            # 检查预定义类型
            predefined_type = space.properties.get('PredefinedType', '').upper()
            if predefined_type in ['SHAFT', 'ATRIUM', 'STAIRCASE', 'USERDEFINED']:
                candidate = self._create_candidate_from_space(space, 'direct_predefined_type')
                if candidate:
                    candidates.append(candidate)
                continue
            
            # 检查名称和描述中的关键词
            text_fields = [
                space.properties.get('Name', ''),
                space.properties.get('LongName', ''),
                space.properties.get('Description', '')
            ]
            combined_text = ' '.join(text_fields).lower()
            
            for space_type, keywords in self.space_type_keywords.items():
                if any(keyword in combined_text for keyword in keywords):
                    candidate = self._create_candidate_from_space(space, 'keyword_match')
                    if candidate:
                        candidates.append(candidate)
                    break
            
            # 检查垂直跨度
            bbox = space.bounding_box
            height = bbox.max_point.z - bbox.min_point.z
            floors_spanned = max(1, int(height / self.detection_params['typical_floor_height']))
            
            if floors_spanned >= self.detection_params['min_floors_spanned']:
                candidate = self._create_candidate_from_space(space, 'vertical_span_analysis')
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _detect_opening_elements(self, all_elements: List[IfcElementInfo]) -> List[DetectionCandidate]:
        """IfcOpeningElement分析"""
        candidates = []
        
        openings = [elem for elem in all_elements if elem.ifc_type == 'IfcOpeningElement']
        slabs = [elem for elem in all_elements if elem.ifc_type == 'IfcSlab']
        
        # 按楼层分组开洞
        floor_openings = defaultdict(list)
        for opening in openings:
            floor_level = self._estimate_floor_level(opening.bounding_box)
            floor_openings[floor_level].append(opening)
        
        # 查找跨楼层的连续开洞
        sorted_floors = sorted(floor_openings.keys())
        
        for i in range(len(sorted_floors) - 1):
            current_floor = sorted_floors[i]
            next_floor = sorted_floors[i + 1]
            
            current_openings = floor_openings[current_floor]
            next_openings = floor_openings[next_floor]
            
            # 查找位置对应的开洞
            for curr_opening in current_openings:
                for next_opening in next_openings:
                    if self._are_openings_aligned(curr_opening, next_opening):
                        # 找到连续的开洞，创建候选项
                        continuous_openings = self._trace_continuous_openings(
                            curr_opening, floor_openings, sorted_floors[i:]
                        )
                        
                        if len(continuous_openings) >= 2:
                            candidate = DetectionCandidate(
                                elements=continuous_openings,
                                detection_method='continuous_openings',
                                confidence=0.8,
                                geometric_continuity=self._calculate_opening_continuity(continuous_openings),
                                vertical_span=self._calculate_opening_span(continuous_openings),
                                consolidated_bbox=self._merge_bounding_boxes([op.bounding_box for op in continuous_openings])
                            )
                            candidates.append(candidate)
        
        return candidates
    
    def _detect_implicit_voids(self, all_elements: List[IfcElementInfo]) -> List[DetectionCandidate]:
        """隐式空洞检测"""
        candidates = []
        
        slabs = [elem for elem in all_elements if elem.ifc_type == 'IfcSlab']
        
        # 按楼层分组楼板
        floor_slabs = defaultdict(list)
        for slab in slabs:
            floor_level = self._estimate_floor_level(slab.bounding_box)
            floor_slabs[floor_level].append(slab)
        
        # 分析楼板开洞模式
        sorted_floors = sorted(floor_slabs.keys())
        
        for i in range(len(sorted_floors) - 1):
            current_floor = sorted_floors[i]
            next_floor = sorted_floors[i + 1]
            
            current_slabs = floor_slabs[current_floor]
            next_slabs = floor_slabs[next_floor]
            
            # 检测一致的缺失区域
            void_regions = self._detect_consistent_void_regions(current_slabs, next_slabs)
            
            for void_region in void_regions:
                # 追踪多层的空洞连续性
                continuous_voids = self._trace_void_continuity(
                    void_region, floor_slabs, sorted_floors[i:]
                )
                
                if len(continuous_voids) >= 2:
                    candidate = DetectionCandidate(
                        elements=[],  # 隐式空洞没有直接的IFC元素
                        detection_method='implicit_void_pattern',
                        confidence=0.6,
                        geometric_continuity=0.8,  # 基于模式的高连续性
                        vertical_span={
                            'total_height': len(continuous_voids) * self.detection_params['typical_floor_height'],
                            'floors_spanned': len(continuous_voids),
                            'void_region': void_region
                        },
                        consolidated_bbox=void_region
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _detect_boundary_discontinuities(self, all_elements: List[IfcElementInfo]) -> List[DetectionCandidate]:
        """边界分析检测"""
        candidates = []
        
        # 简化实现：基于空间边界关系推断垂直连通
        spaces = [elem for elem in all_elements if elem.ifc_type == 'IfcSpace']
        
        # 分析空间的垂直连通性
        for space in spaces:
            if 'relationships' in space.properties:
                boundary_rels = space.properties.get('relationships', {}).get('IfcRelSpaceBoundary', [])
                
                # 检查是否有垂直边界缺失（暗示垂直连通）
                has_vertical_discontinuity = self._analyze_boundary_discontinuity(boundary_rels)
                
                if has_vertical_discontinuity:
                    candidate = self._create_candidate_from_space(space, 'boundary_discontinuity')
                    if candidate:
                        candidates.append(candidate)
        
        return candidates
    
    def _validate_geometric_continuity(self, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
        """几何连续性验证"""
        validated = []
        
        for candidate in candidates:
            # 垂直对齐检查
            alignment_score = self._check_vertical_alignment(candidate)
            
            # 多楼层跨度验证
            span_validation = self._verify_multi_floor_span(candidate)
            
            # 几何完整性检查
            completeness_score = self._check_geometric_completeness(candidate)
            
            # 综合评分
            overall_score = (alignment_score + span_validation + completeness_score) / 3
            
            if overall_score >= 0.6:  # 通过验证阈值
                candidate.confidence *= overall_score
                candidate.geometric_continuity = overall_score
                validated.append(candidate)
        
        log.debug(f"Geometric validation: {len(validated)}/{len(candidates)} candidates passed")
        return validated
    
    def _consolidate_and_deduplicate(self, candidates: List[DetectionCandidate]) -> List[DetectionCandidate]:
        """整合与去重"""
        if not candidates:
            return []
        
        # 按空间位置分组
        spatial_groups = self._group_by_spatial_overlap(candidates)
        
        consolidated = []
        
        for group in spatial_groups:
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # 合并重叠的候选项
                merged_candidate = self._merge_candidates(group)
                consolidated.append(merged_candidate)
        
        log.debug(f"Consolidation: {len(candidates)} candidates → {len(consolidated)} consolidated")
        return consolidated
    
    def _classify_and_finalize(self, candidates: List[DetectionCandidate],
                              spatial_context: Optional[SpatialContext] = None) -> List[VerticalSpaceInfo]:
        """分类和最终验证"""
        vertical_spaces = []
        
        for candidate in candidates:
            # 确定垂直空间类型
            space_type = self._classify_vertical_space_type(candidate)
            
            # 计算面积影响
            area_impact = self._calculate_area_impact(candidate, space_type)
            
            # 收集证据
            evidence = self._collect_detection_evidence(candidate)
            
            # 创建垂直空间信息
            vertical_space = VerticalSpaceInfo(
                space_type=space_type,
                elements=candidate.elements,
                bounding_box=candidate.consolidated_bbox,
                floors_spanned=candidate.vertical_span.get('floors_spanned', 1),
                total_height=candidate.vertical_span.get('total_height', 0.0),
                area_per_floor=area_impact['area_per_floor'],
                detection_method=candidate.detection_method,
                confidence=candidate.confidence,
                evidence=evidence
            )
            
            vertical_spaces.append(vertical_space)
        
        return vertical_spaces
    
    # 辅助方法实现
    
    def _create_candidate_from_space(self, space: IfcElementInfo, method: str) -> Optional[DetectionCandidate]:
        """从空间创建候选项"""
        bbox = space.bounding_box
        height = bbox.max_point.z - bbox.min_point.z
        floors_spanned = max(1, int(height / self.detection_params['typical_floor_height']))
        
        if floors_spanned < self.detection_params['min_floors_spanned']:
            return None
        
        confidence_map = {
            'direct_predefined_type': 0.9,
            'keyword_match': 0.8,
            'vertical_span_analysis': 0.7,
            'boundary_discontinuity': 0.6
        }
        
        return DetectionCandidate(
            elements=[space],
            detection_method=method,
            confidence=confidence_map.get(method, 0.5),
            geometric_continuity=0.8,
            vertical_span={
                'total_height': height,
                'floors_spanned': floors_spanned
            },
            consolidated_bbox=bbox
        )
    
    def _estimate_floor_level(self, bbox: BoundingBox) -> float:
        """估算楼层标高"""
        z_center = (bbox.min_point.z + bbox.max_point.z) / 2
        return round(z_center / self.detection_params['typical_floor_height']) * self.detection_params['typical_floor_height']
    
    def _are_openings_aligned(self, opening1: IfcElementInfo, opening2: IfcElementInfo) -> bool:
        """检查两个开洞是否对齐"""
        bbox1, bbox2 = opening1.bounding_box, opening2.bounding_box
        
        # 水平位置对齐检查
        center1 = Point3D(
            (bbox1.min_point.x + bbox1.max_point.x) / 2,
            (bbox1.min_point.y + bbox1.max_point.y) / 2,
            (bbox1.min_point.z + bbox1.max_point.z) / 2
        )
        
        center2 = Point3D(
            (bbox2.min_point.x + bbox2.max_point.x) / 2,
            (bbox2.min_point.y + bbox2.max_point.y) / 2,
            (bbox2.min_point.z + bbox2.max_point.z) / 2
        )
        
        horizontal_distance = np.sqrt(
            (center1.x - center2.x) ** 2 + (center1.y - center2.y) ** 2
        )
        
        return horizontal_distance <= self.detection_params['horizontal_tolerance']
    
    def _trace_continuous_openings(self, start_opening: IfcElementInfo,
                                  floor_openings: Dict[float, List[IfcElementInfo]],
                                  floor_levels: List[float]) -> List[IfcElementInfo]:
        """追踪连续开洞"""
        continuous = [start_opening]
        current_opening = start_opening
        
        for i in range(1, len(floor_levels)):
            floor_level = floor_levels[i]
            if floor_level not in floor_openings:
                break
            
            # 在当前楼层找到对齐的开洞
            aligned_opening = None
            for opening in floor_openings[floor_level]:
                if self._are_openings_aligned(current_opening, opening):
                    aligned_opening = opening
                    break
            
            if aligned_opening:
                continuous.append(aligned_opening)
                current_opening = aligned_opening
            else:
                break
        
        return continuous
    
    def _calculate_opening_continuity(self, openings: List[IfcElementInfo]) -> float:
        """计算开洞连续性"""
        if len(openings) < 2:
            return 0.0
        
        # 简化实现：基于截面一致性
        areas = []
        for opening in openings:
            bbox = opening.bounding_box
            area = abs((bbox.max_point.x - bbox.min_point.x) * (bbox.max_point.y - bbox.min_point.y))
            areas.append(area)
        
        if not areas:
            return 0.0
        
        avg_area = sum(areas) / len(areas)
        variance = sum((area - avg_area) ** 2 for area in areas) / len(areas)
        consistency = 1.0 / (1.0 + variance / (avg_area ** 2)) if avg_area > 0 else 0.0
        
        return consistency
    
    def _calculate_opening_span(self, openings: List[IfcElementInfo]) -> Dict[str, Any]:
        """计算开洞跨度"""
        if not openings:
            return {'total_height': 0.0, 'floors_spanned': 0}
        
        z_coords = []
        for opening in openings:
            bbox = opening.bounding_box
            z_coords.extend([bbox.min_point.z, bbox.max_point.z])
        
        total_height = max(z_coords) - min(z_coords)
        floors_spanned = len(openings)
        
        return {
            'total_height': total_height,
            'floors_spanned': floors_spanned
        }
    
    def _merge_bounding_boxes(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """合并边界框"""
        if not bboxes:
            return BoundingBox(Point3D(0, 0, 0), Point3D(0, 0, 0))
        
        min_x = min(bbox.min_point.x for bbox in bboxes)
        min_y = min(bbox.min_point.y for bbox in bboxes)
        min_z = min(bbox.min_point.z for bbox in bboxes)
        
        max_x = max(bbox.max_point.x for bbox in bboxes)
        max_y = max(bbox.max_point.y for bbox in bboxes)
        max_z = max(bbox.max_point.z for bbox in bboxes)
        
        return BoundingBox(
            Point3D(min_x, min_y, min_z),
            Point3D(max_x, max_y, max_z)
        )
    
    def _detect_consistent_void_regions(self, slabs1: List[IfcElementInfo], 
                                      slabs2: List[IfcElementInfo]) -> List[BoundingBox]:
        """检测一致的空洞区域（简化实现）"""
        # 这里需要复杂的几何分析，简化为基本实现
        void_regions = []
        
        # 基于楼板缺失区域的简单检测
        # 实际实现需要更复杂的几何计算
        
        return void_regions
    
    def _trace_void_continuity(self, void_region: BoundingBox,
                              floor_slabs: Dict[float, List[IfcElementInfo]],
                              floor_levels: List[float]) -> List[BoundingBox]:
        """追踪空洞连续性"""
        continuous_voids = [void_region]
        
        # 简化实现
        for floor_level in floor_levels[1:]:
            # 检查该楼层是否有相同位置的空洞
            # 实际需要几何分析
            pass
        
        return continuous_voids
    
    def _analyze_boundary_discontinuity(self, boundary_rels: List[Any]) -> bool:
        """分析边界不连续性"""
        # 简化实现：检查是否缺少顶部或底部边界
        return len(boundary_rels) < 6  # 完整空间应该有6个面的边界
    
    def _check_vertical_alignment(self, candidate: DetectionCandidate) -> float:
        """检查垂直对齐"""
        if not candidate.elements:
            return 0.5
        
        # 计算所有元素的水平中心点
        centers = []
        for element in candidate.elements:
            bbox = element.bounding_box
            center = Point3D(
                (bbox.min_point.x + bbox.max_point.x) / 2,
                (bbox.min_point.y + bbox.max_point.y) / 2,
                (bbox.min_point.z + bbox.max_point.z) / 2
            )
            centers.append(center)
        
        if len(centers) < 2:
            return 0.8
        
        # 计算水平偏差
        avg_x = sum(c.x for c in centers) / len(centers)
        avg_y = sum(c.y for c in centers) / len(centers)
        
        max_deviation = max(
            np.sqrt((c.x - avg_x) ** 2 + (c.y - avg_y) ** 2) for c in centers
        )
        
        # 转换为对齐分数
        alignment_score = max(0.0, 1.0 - max_deviation / self.detection_params['horizontal_tolerance'])
        return alignment_score
    
    def _verify_multi_floor_span(self, candidate: DetectionCandidate) -> float:
        """验证多楼层跨度"""
        floors_spanned = candidate.vertical_span.get('floors_spanned', 1)
        
        if floors_spanned >= self.detection_params['min_floors_spanned']:
            return min(1.0, floors_spanned / 3.0)  # 3层以上给满分
        else:
            return 0.0
    
    def _check_geometric_completeness(self, candidate: DetectionCandidate) -> float:
        """检查几何完整性"""
        # 简化实现：基于检测方法给出完整性分数
        method_scores = {
            'direct_predefined_type': 0.9,
            'keyword_match': 0.8,
            'continuous_openings': 0.85,
            'implicit_void_pattern': 0.7,
            'boundary_discontinuity': 0.6
        }
        
        return method_scores.get(candidate.detection_method, 0.5)
    
    def _group_by_spatial_overlap(self, candidates: List[DetectionCandidate]) -> List[List[DetectionCandidate]]:
        """按空间重叠分组"""
        groups = []
        used = set()
        
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            
            group = [candidate]
            used.add(i)
            
            for j, other_candidate in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                
                if self._do_candidates_overlap(candidate, other_candidate):
                    group.append(other_candidate)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _do_candidates_overlap(self, candidate1: DetectionCandidate, 
                              candidate2: DetectionCandidate) -> bool:
        """检查候选项是否重叠"""
        bbox1 = candidate1.consolidated_bbox
        bbox2 = candidate2.consolidated_bbox
        
        # 简单的边界框重叠检查
        overlap_x = (bbox1.min_point.x <= bbox2.max_point.x and bbox2.min_point.x <= bbox1.max_point.x)
        overlap_y = (bbox1.min_point.y <= bbox2.max_point.y and bbox2.min_point.y <= bbox1.max_point.y)
        overlap_z = (bbox1.min_point.z <= bbox2.max_point.z and bbox2.min_point.z <= bbox1.max_point.z)
        
        return overlap_x and overlap_y and overlap_z
    
    def _merge_candidates(self, candidates: List[DetectionCandidate]) -> DetectionCandidate:
        """合并候选项"""
        # 选择优先级最高的检测方法
        best_candidate = max(candidates, key=lambda c: c.confidence)
        
        # 合并所有元素
        all_elements = []
        for candidate in candidates:
            all_elements.extend(candidate.elements)
        
        # 去重
        unique_elements = []
        seen_guids = set()
        for element in all_elements:
            if element.guid not in seen_guids:
                unique_elements.append(element)
                seen_guids.add(element.guid)
        
        # 合并边界框
        all_bboxes = [c.consolidated_bbox for c in candidates]
        merged_bbox = self._merge_bounding_boxes(all_bboxes)
        
        # 重新计算跨度
        height = merged_bbox.max_point.z - merged_bbox.min_point.z
        floors_spanned = max(1, int(height / self.detection_params['typical_floor_height']))
        
        return DetectionCandidate(
            elements=unique_elements,
            detection_method=best_candidate.detection_method,
            confidence=max(c.confidence for c in candidates),
            geometric_continuity=sum(c.geometric_continuity for c in candidates) / len(candidates),
            vertical_span={
                'total_height': height,
                'floors_spanned': floors_spanned
            },
            consolidated_bbox=merged_bbox
        )
    
    def _classify_vertical_space_type(self, candidate: DetectionCandidate) -> VerticalSpaceType:
        """分类垂直空间类型"""
        # 基于关键词匹配
        for element in candidate.elements:
            text_fields = [
                element.properties.get('Name', ''),
                element.properties.get('LongName', ''),
                element.properties.get('Description', ''),
                element.properties.get('PredefinedType', '')
            ]
            combined_text = ' '.join(text_fields).lower()
            
            for space_type, keywords in self.space_type_keywords.items():
                if any(keyword in combined_text for keyword in keywords):
                    return space_type
        
        # 基于几何特征推断
        bbox = candidate.consolidated_bbox
        width = bbox.max_point.x - bbox.min_point.x
        length = bbox.max_point.y - bbox.min_point.y
        height = bbox.max_point.z - bbox.min_point.z
        
        horizontal_area = width * length
        aspect_ratio = max(width, length) / min(width, length) if min(width, length) > 0 else 1.0
        
        # 基于尺寸特征分类
        if horizontal_area > 50.0 and aspect_ratio < 2.0:  # 大面积，相对方正
            return VerticalSpaceType.ATRIUM
        elif horizontal_area < 10.0 or aspect_ratio > 5.0:  # 小面积或细长
            return VerticalSpaceType.SHAFT
        else:
            return VerticalSpaceType.VOID  # 默认为一般空洞
    
    def _calculate_area_impact(self, candidate: DetectionCandidate, 
                              space_type: VerticalSpaceType) -> Dict[str, float]:
        """计算面积影响"""
        bbox = candidate.consolidated_bbox
        area_per_floor = abs((bbox.max_point.x - bbox.min_point.x) * 
                            (bbox.max_point.y - bbox.min_point.y))
        
        return {
            'area_per_floor': area_per_floor,
            'total_area_impact': area_per_floor * candidate.vertical_span.get('floors_spanned', 1)
        }
    
    def _collect_detection_evidence(self, candidate: DetectionCandidate) -> List[Evidence]:
        """收集检测证据"""
        evidence = []
        
        # 检测方法证据
        evidence.append(Evidence(
            text=f"检测方法: {candidate.detection_method}",
            source='vertical_space_detection',
            confidence=candidate.confidence
        ))
        
        # 几何连续性证据
        evidence.append(Evidence(
            text=f"几何连续性: {candidate.geometric_continuity:.3f}",
            source='geometric_validation',
            confidence=candidate.geometric_continuity
        ))
        
        # 垂直跨度证据
        floors_spanned = candidate.vertical_span.get('floors_spanned', 1)
        total_height = candidate.vertical_span.get('total_height', 0.0)
        evidence.append(Evidence(
            text=f"垂直跨度: {floors_spanned}层, 总高度: {total_height:.2f}m",
            source='span_analysis',
            confidence=0.8
        ))
        
        # 元素证据
        if candidate.elements:
            element_types = list(set(elem.ifc_type for elem in candidate.elements))
            evidence.append(Evidence(
                text=f"涉及IFC元素: {', '.join(element_types)}",
                source='ifc_elements',
                confidence=0.7
            ))
        
        return evidence