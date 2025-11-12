# IFC Semantic Ambiguity Test Set: Standard Attributes + Geometry Inference for Area Calculation

## Core Strategy

### Dual Differentiation Mechanism
1. **Standard Attribute Layer** (IFC Schema defined properties)
   - `PredefinedType` enumeration values
   - `IsExternal` boolean value
   - `Name` field (auxiliary validation only)
   
2. **Geometry Inference Layer** (Extracted from IFC geometry & relationships)
   - Spatial dimensions (height, width, area, volume)
   - Topological relationships (multi-floor spanning, containment, connectivity)
   - Location features (elevation, relative position)
   - Shape characteristics (aspect ratio, thickness)

### Decision Flow for Area Calculation
```
Input IFC Object
    ↓
Extract PredefinedType
    ↓
Is USERDEFINED/NOTDEFINED?
    ├→ No: Direct mapping (Low ambiguity)
    └→ Yes: Enter geometry inference (High ambiguity)
            ↓
        Extract geometric features
            ↓
        Apply ML classifier
            ↓
        Output function + confidence + area calculation rule
```

---

## IFC Standard Attributes Reference

### IfcSlab.PredefinedType (IFC4)
```python
enum_values = {
    'FLOOR': Floor slab,
    'ROOF': Roof slab,
    'LANDING': Landing platform,
    'BASESLAB': Foundation base slab,
    'USERDEFINED': User-defined (requires geometry inference),
    'NOTDEFINED': Not defined (requires geometry inference)
}
```

### IfcSpace.PredefinedType (IFC4)
```python
enum_values = {
    'SPACE': General space (requires geometry inference),
    'PARKING': Parking space,
    'GFA': Gross Floor Area,
    'INTERNAL': Internal space (requires geometry inference),
    'EXTERNAL': External space,
    'USERDEFINED': User-defined (requires geometry inference),
    'NOTDEFINED': Not defined (requires geometry inference)
}
```

### IfcWall.PredefinedType (IFC4)
```python
enum_values = {
    'STANDARD': Standard wall (needs IsExternal to distinguish),
    'POLYGONAL': Polygonal wall,
    'SHEAR': Shear wall,
    'ELEMENTEDWALL': Composite wall,
    'PLUMBINGWALL': Plumbing wall,
    'USERDEFINED': User-defined,
    'NOTDEFINED': Not defined
}
```

---

## Complete Test Set: 100 Annotated Samples

### Grouping Strategy
- **Group A (30 samples)**: PredefinedType clear, no inference needed
- **Group B (40 samples)**: PredefinedType ambiguous, needs geometry inference
- **Group C (20 samples)**: IfcOpeningElement fully dependent on geometry
- **Group D (10 samples)**: Cross-type equivalence (hardest)

---

## Group A: Clear Standard Attributes (30 samples)

| No. | IFC Type | PredefinedType | IsExternal | Direct Mapping | Building Code | Area Calculation Rule | Include in GFA |
|-----|----------|---------------|-----------|----------------|---------------|-----------------------|----------------|
| 1 | IfcSlab | FLOOR | FALSE | Interior floor slab | Floor assembly | Full area | YES |
| 2 | IfcSlab | ROOF | - | Roof slab | Roof assembly | Full area | NO |
| 3 | IfcSlab | LANDING | FALSE | Stair landing | Stair landing | Landing area only | YES |
| 4 | IfcSlab | BASESLAB | FALSE | Foundation base slab | Foundation slab | Not counted | NO |
| 5 | IfcSpace | PARKING | FALSE | Indoor parking | Parking structure | Full area | Partial (50%) |
| 6 | IfcSpace | EXTERNAL | TRUE | Exterior space | Exterior space | Not counted | NO |
| 7 | IfcSpace | INTERNAL | FALSE | Interior office space | Habitable space | Full area | YES |
| 8 | IfcSpace | GFA | - | Gross floor area | Building area | Full area | YES |
| 9 | IfcWall | STANDARD | TRUE | Exterior wall | Exterior wall | Wall thickness to centerline | Partial |
| 10 | IfcWall | STANDARD | FALSE | Interior wall | Interior partition | Not counted in floor area | NO |
| 11 | IfcWall | SHEAR | FALSE | Shear wall | Shear wall | To centerline | Partial |
| 12 | IfcWall | PLUMBINGWALL | FALSE | Plumbing wall | Service wall | Not counted | NO |
| 13 | IfcColumn | COLUMN | - | Column | Vertical support | Column footprint deducted | Deduct |
| 14 | IfcBeam | BEAM | - | Beam | Horizontal member | Not affecting floor area | NO |
| 15 | IfcStair | STRAIGHT_RUN_STAIR | - | Straight stair | Exit stair | Stair footprint | YES |
| 16 | IfcStair | QUARTER_WINDING_STAIR | - | L-shaped stair | Exit stair | Stair footprint | YES |
| 17 | IfcRamp | STRAIGHT_RAMP | - | Straight ramp | Accessible ramp | Ramp footprint | YES |
| 18 | IfcRoof | FLAT_ROOF | - | Flat roof | Roof assembly | Full area | NO |
| 19 | IfcRoof | SHED_ROOF | - | Shed roof | Roof assembly | Projected area | NO |
| 20 | IfcCovering | CEILING | - | Ceiling finish | Ceiling finish | Not counted | NO |
| 21 | IfcCovering | FLOORING | - | Floor finish | Floor finish | Not counted | NO |
| 22 | IfcRailing | HANDRAIL | - | Handrail | Handrail | Not counted | NO |
| 23 | IfcRailing | GUARDRAIL | - | Guardrail | Fall protection | Not counted | NO |
| 24 | IfcCurtainWall | USERDEFINED | TRUE | Curtain wall | Curtain wall | To outer face | Partial |
| 25 | IfcSlab | FLOOR | TRUE | Exterior floor (balcony) | Exterior floor | Partial area (50%) | Partial |
| 26 | IfcSpace | PARKING | TRUE | Outdoor parking | Open parking | Not counted | NO |
| 27 | IfcSlab | LANDING | TRUE | Exterior landing | Exterior platform | Partial area | Partial |
| 28 | IfcRoof | GABLE_ROOF | - | Gable roof | Roof assembly | Projected area | NO |
| 29 | IfcColumn | PILASTER | - | Engaged column | Wall stiffener | Not deducted | NO |
| 30 | IfcBeam | LINTEL | - | Lintel beam | Opening support | Not affecting area | NO |

---

## Group B: Requires Geometry Inference (40 samples)

### Feature Extraction Method
```python
# Geometry feature extraction for area calculation
features = {
    'height': space height (m),
    'width': space width (m),  
    'length': space length (m),
    'area': horizontal area (m²),
    'volume': volume (m³),
    'aspect_ratio': height/width,
    'thickness': slab thickness (mm),
    'elevation': elevation (m),
    'floor_count': number of floors spanned,
    'has_elevator': contains elevator,
    'has_stair': contains stair,
    'has_duct': contains ductwork,
    'has_pipe': contains piping,
    'is_top_floor': top floor location,
    'is_basement': basement location,
    'perimeter_area_ratio': perimeter²/area
}
```

| No. | IFC | PredefinedType | Key Geometry Features | Inference Rule | Inferred Function | Confidence | Area Rule | GFA |
|-----|-----|---------------|----------------------|---------------|-------------------|-----------|-----------|-----|
| 31 | IfcSpace | INTERNAL | H>20m, W<3m, has_elevator=T | Super tall+narrow+elevator | Elevator shaft | 95% | Not counted (shaft) | NO |
| 32 | IfcSpace | INTERNAL | H>12m, W>8m, floor_count≥3 | Super tall+wide+multi-floor | Atrium | 90% | Not counted (void) | NO |
| 33 | IfcSpace | INTERNAL | H>15m, W<2m, has_stair=T | Super tall+narrow+stair | Stairwell | 92% | Not counted (shaft) | NO |
| 34 | IfcSpace | INTERNAL | H>15m, aspect_ratio>10, has_duct=T | Super tall+slender+duct | HVAC shaft | 88% | Not counted (shaft) | NO |
| 35 | IfcSpace | INTERNAL | H>15m, aspect_ratio>15, has_pipe=T | Super tall+slender+pipe | Plumbing shaft | 88% | Not counted (shaft) | NO |
| 36 | IfcSpace | INTERNAL | H>15m, aspect_ratio>15, cables | Super tall+slender+cable | Electrical shaft | 85% | Not counted (shaft) | NO |
| 37 | IfcSpace | INTERNAL | H<4m, Area<30m², equipment | Normal height+small+equipment | Mechanical room | 80% | Counted as service | YES |
| 38 | IfcSpace | INTERNAL | H<4m, is_basement=T, refuse | Underground+refuse | Refuse room | 85% | Counted as service | YES |
| 39 | IfcSpace | INTERNAL | H<4m, electrical equipment | Normal height+electrical | Electrical room | 90% | Counted as service | YES |
| 40 | IfcSpace | INTERNAL | H<4m, fire equipment | Normal height+fire system | Fire control room | 92% | Counted as service | YES |
| 41 | IfcSpace | INTERNAL | H<4m, plumbing fixtures | Normal height+fixtures | Toilet room | 95% | Counted as service | YES |
| 42 | IfcSpace | INTERNAL | H<4m, elevator adjacent | Adjacent to elevator | Elevator lobby | 88% | Counted as circulation | YES |
| 43 | IfcSpace | INTERNAL | H<6m, is_basement=T, Area>500m² | Underground+large area | Underground parking | 85% | Counted partial (50%) | Partial |
| 44 | IfcSpace | INTERNAL | H<4m, refrigeration | Normal height+cold storage | Cold storage | 90% | Counted as storage | YES |
| 45 | IfcSpace | INTERNAL | H<4m, server racks | Normal height+IT equipment | Data center | 92% | Counted as equipment | YES |
| 46 | IfcSpace | INTERNAL | H<4m, clean rating | Normal height+clean | Clean room | 88% | Counted as production | YES |
| 47 | IfcSpace | USERDEFINED | H<6m, smoke protection | Smoke protection+stair | Refuge area | 85% | Counted as circulation | YES |
| 48 | IfcSpace | INTERNAL | H<4m, corridor | Linear space | Corridor | 90% | Counted as circulation | YES |
| 49 | IfcSpace | INTERNAL | H>4m, W>20m, Area>200m² | High ceiling+large | Assembly space | 88% | Counted as occupiable | YES |
| 50 | IfcSpace | INTERNAL | H<4m, Area>100m², open plan | Normal+large+open | Open office | 92% | Counted as occupiable | YES |
| 51 | IfcSlab | USERDEFINED | is_top_floor=T, thickness>500mm | Roof+super thick | Helipad | 95% | Not counted (roof) | NO |
| 52 | IfcSlab | USERDEFINED | is_external=T, cantilever | External+cantilever | Balcony | 92% | Counted partial (50%) | Partial |
| 53 | IfcSlab | USERDEFINED | Mid-floor, partial slab | Partial+mid-level | Mezzanine floor | 85% | Counted full | YES |
| 54 | IfcSlab | USERDEFINED | is_basement=T, Area>1000m² | Underground+large | Parking deck | 88% | Counted partial (50%) | Partial |
| 55 | IfcSlab | USERDEFINED | Connecting buildings, elevated | Bridge+elevated | Pedestrian bridge | 82% | Counted as circulation | YES |
| 56 | IfcSlab | USERDEFINED | Small area, equipment load | Partial+heavy load | Equipment platform | 80% | Not counted (service) | NO |
| 57 | IfcSlab | USERDEFINED | Grade level, loading area | Boundary+loading | Loading dock | 85% | Not counted (service) | NO |
| 58 | IfcSlab | USERDEFINED | is_basement=T, on grade | Underground+foundation | Basement slab | 90% | Counted if occupiable | Conditional |
| 59 | IfcSlab | FLOOR | Contains pool | Floor+pool | Pool floor slab | 92% | Not counted (pool) | NO |
| 60 | IfcSlab | USERDEFINED | Transfer level, thick | Transfer+thick | Transfer slab | 88% | Counted as structural | YES |
| 61 | IfcSlab | USERDEFINED | Exterior, grade | Exterior+grade | Patio/plaza | 85% | Not counted (exterior) | NO |
| 62 | IfcSlab | ROOF | Vegetated surface | Roof+green | Green roof | 80% | Not counted (roof) | NO |
| 63 | IfcSlab | USERDEFINED | Covered walkway | Covered+pedestrian | Covered walkway | 82% | Counted if enclosed | Conditional |
| 64 | IfcSlab | USERDEFINED | Canopy, no enclosure | Canopy+open | Canopy deck | 78% | Not counted (open) | NO |
| 65 | IfcWall | USERDEFINED | H>6m, independent foundation | Super tall+independent | Fire wall | 90% | To centerline | Partial |
| 66 | IfcWall | STANDARD | Unit separation | Between units | Demising wall | 85% | To centerline | Partial |
| 67 | IfcWall | USERDEFINED | is_basement=T, retaining | Underground+retaining | Retaining wall | 95% | Not counted | NO |
| 68 | IfcWall | USERDEFINED | is_top_floor=T, low height | Roof+low | Parapet | 92% | Not counted | NO |
| 69 | IfcWall | USERDEFINED | Acoustic requirement | Sound performance | Acoustic wall | 80% | To centerline | Partial |
| 70 | IfcStair | USERDEFINED | Symmetric layout, shared shaft | Symmetric+shared | Scissor stair | 88% | Stair footprint | YES |

---

## Group C: IfcOpeningElement (20 samples)

**Key Point**: IfcOpeningElement has no PredefinedType, 100% dependent on geometry inference

**Area Calculation Impact**: Openings affect net floor area calculation by creating voids

| No. | IFC | Geometry Feature 1 | Geometry Feature 2 | Geometry Feature 3 | Inferred Function | Confidence | Area Impact |
|-----|-----|-------------------|-------------------|-------------------|-------------------|-----------|-------------|
| 71 | IfcOpeningElement | H>12m | W>8m | Penetrates multiple floors | Atrium void | 90% | Deduct full void area |
| 72 | IfcOpeningElement | H>15m | W<3m | Contains elevator | Elevator shaft opening | 92% | Deduct shaft area |
| 73 | IfcOpeningElement | H>15m | W<2m | Contains stair | Stairwell opening | 90% | Deduct shaft area |
| 74 | IfcOpeningElement | H>12m | aspect>10 | Vertical ductwork | HVAC shaft opening | 85% | Deduct shaft area |
| 75 | IfcOpeningElement | H>12m | aspect>12 | Vertical piping | Plumbing shaft opening | 85% | Deduct shaft area |
| 76 | IfcOpeningElement | H>10m | aspect>15 | Cable vertical shaft | Electrical shaft opening | 83% | Deduct shaft area |
| 77 | IfcOpeningElement | In slab | D<0.5m | Circular | Small service penetration | 80% | Ignore (too small) |
| 78 | IfcOpeningElement | In slab | D>1m | Rectangular | Large equipment opening | 85% | Deduct opening area |
| 79 | IfcOpeningElement | Building boundary | Narrow gap | Vertical | Expansion joint | 78% | Not deducted |
| 80 | IfcOpeningElement | In slab | D<0.3m | Round | Floor drain opening | 70% | Ignore (too small) |
| 81 | IfcOpeningElement | Underground exterior wall | Light well | - | Window well | 75% | Not affecting floor area |
| 82 | IfcOpeningElement | In slab | Temporary | Rigging | Equipment rigging opening | 72% | Temporary - not deducted |
| 83 | IfcOpeningElement | Multi-floor | Large void | Central | Central void/atrium | 88% | Deduct full void |
| 84 | IfcOpeningElement | In slab | Stair penetration | - | Stair opening | 90% | Deduct stair opening |
| 85 | IfcOpeningElement | In slab | Escalator penetration | - | Escalator opening | 88% | Deduct escalator opening |
| 86 | IfcOpeningElement | In slab | Vertical circulation | - | Vertical circulation void | 85% | Deduct circulation void |
| 87 | IfcOpeningElement | In roof | Large area | Glass | Glazed roof opening | 78% | Not affecting floor area |
| 88 | IfcOpeningElement | Between floors | Vertical void | Open | Open floor connection | 82% | Deduct connection void |
| 89 | IfcOpeningElement | In slab | Loading access | - | Loading dock opening | 75% | Not deducted (service) |
| 90 | IfcOpeningElement | Partial floor | Balcony edge | - | Balcony void boundary | 70% | Defines balcony edge |

---

## Group D: Cross-Type Equivalence (10 samples) - Highest Difficulty

**Core Challenge**: Same function may be expressed with different IFC types, affecting area calculation consistency

| No. | Function | Expression 1 | Expression 2 | Geometry Discriminator | Confidence Threshold | Area Calculation Strategy |
|-----|----------|--------------|--------------|------------------------|---------------------|---------------------------|
| 91 | Atrium | IfcSpace (H>12m) | IfcOpeningElement (H>12m) | Height>12m + Width>8m | >85% | Deduct void area from floor |
| 92 | Elevator shaft | IfcSpace (narrow) | IfcOpeningElement (narrow) | Height>15m + Width<3m + elevator | >90% | Deduct shaft area from floor |
| 93 | Stairwell | IfcSpace | IfcOpeningElement | Height>12m + Width<3m + stair | >88% | Deduct stairwell area |
| 94 | HVAC shaft | IfcSpace | IfcOpeningElement | Height>10m + slender + duct | >80% | Deduct shaft area |
| 95 | Plumbing shaft | IfcSpace | IfcOpeningElement | Height>10m + slender + pipe | >80% | Deduct shaft area |
| 96 | Electrical shaft | IfcSpace | IfcOpeningElement | Height>10m + very slender + cable | >78% | Deduct shaft area |
| 97 | Light well | IfcSpace (basement) | IfcOpeningElement | Underground exterior wall + well | >75% | Not deducted from GFA |
| 98 | Equipment opening | IfcOpeningElement | IfcSlab (HasOpening) | In slab + small hole | >70% | Deduct if significant size |
| 99 | Bridge space | IfcSpace | IfcSlab (Bridge) | Connecting buildings + elevated | >75% | Count as circulation area |
| 100 | Central void | IfcSpace (multi-floor) | IfcOpeningElement (void) | Multi-floor + central + large | >80% | Deduct void from all floors |

---

## Geometry Inference Algorithm Framework

### Feature Engineering
```python
def extract_geometry_features(ifc_element):
    """Extract geometric features for ML classification and area calculation"""
    
    features = {}
    
    # Basic dimensions
    if ifc_element.is_a('IfcSpace'):
        bbox = get_bounding_box(ifc_element)
        features['height'] = bbox.z_max - bbox.z_min
        features['width'] = min(bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min)
        features['length'] = max(bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min)
        features['area'] = calculate_floor_area(ifc_element)
        features['volume'] = calculate_volume(ifc_element)
        
        # Shape features
        features['aspect_ratio'] = features['height'] / features['width'] if features['width'] > 0 else 0
        features['slenderness'] = features['height'] / features['area']**0.5 if features['area'] > 0 else 0
        
        # Topological features
        features['floor_count'] = count_spanned_floors(ifc_element)
        features['is_vertical_shaft'] = features['floor_count'] >= 2 and features['aspect_ratio'] > 5
        
        # Location features
        features['elevation'] = get_bottom_elevation(ifc_element)
        features['is_top_floor'] = is_at_top_floor(ifc_element)
        features['is_basement'] = features['elevation'] < 0
        
        # Containment relationships
        features['has_elevator'] = contains_element_type(ifc_element, 'IfcTransportElement')
        features['has_stair'] = contains_element_type(ifc_element, 'IfcStair')
        features['has_duct'] = contains_element_type(ifc_element, 'IfcDuctSegment')
        features['has_pipe'] = contains_element_type(ifc_element, 'IfcPipeSegment')
        
    elif ifc_element.is_a('IfcSlab'):
        features['thickness'] = get_thickness(ifc_element)
        features['area'] = calculate_surface_area(ifc_element)
        features['elevation'] = get_elevation(ifc_element)
        features['is_top_floor'] = is_at_top_floor(ifc_element)
        features['is_external'] = getattr(ifc_element, 'IsExternal', False)
        features['is_cantilever'] = check_cantilever(ifc_element)
        
    elif ifc_element.is_a('IfcOpeningElement'):
        bbox = get_bounding_box(ifc_element)
        features['height'] = bbox.z_max - bbox.z_min
        features['width'] = min(bbox.x_max - bbox.x_min, bbox.y_max - bbox.y_min)
        features['depth'] = get_depth(ifc_element)
        features['shape'] = classify_shape(ifc_element)  # 'rectangular', 'circular', etc.
        features['orientation'] = get_orientation(ifc_element)  # 'vertical', 'horizontal'
        features['host_type'] = get_host_element_type(ifc_element)  # 'Wall', 'Slab', 'Roof'
        features['floor_count'] = count_penetrated_floors(ifc_element)
        
    return features
```

### Classification Decision Tree for Area Calculation
```python
def classify_space_for_area_calculation(features):
    """Classify space function based on geometric features for area calculation rules"""
    
    # Super tall spaces (>12m) - typically excluded from floor area
    if features['height'] > 12:
        if features['aspect_ratio'] > 8:
            # Slender vertical shafts
            if features['has_elevator']:
                return 'Elevator Shaft', 0.95, 'EXCLUDE_SHAFT'
            elif features['has_stair']:
                return 'Stairwell', 0.92, 'EXCLUDE_SHAFT'
            elif features['has_duct']:
                return 'HVAC Shaft', 0.88, 'EXCLUDE_SHAFT'
            elif features['has_pipe']:
                return 'Plumbing Shaft', 0.88, 'EXCLUDE_SHAFT'
            else:
                return 'Generic Shaft', 0.70, 'EXCLUDE_SHAFT'
        elif features['width'] > 8 and features['floor_count'] >= 3:
            # Wide multi-floor space
            return 'Atrium', 0.90, 'EXCLUDE_VOID'
    
    # Normal height spaces (<6m) - typically included in floor area
    elif features['height'] < 6:
        if features['is_basement'] and features['area'] > 500:
            return 'Parking Garage', 0.85, 'INCLUDE_PARTIAL_50'
        elif features['area'] < 30:
            # Small rooms - service spaces
            return 'Service Room', 0.75, 'INCLUDE_FULL'
        elif features['has_elevator']:
            return 'Elevator Lobby', 0.88, 'INCLUDE_FULL'
        elif features['area'] > 100:
            return 'Occupiable Space', 0.85, 'INCLUDE_FULL'
    
    return 'Generic Space', 0.50, 'INCLUDE_FULL'
```

### Area Calculation Rules
```python
def calculate_floor_area(ifc_elements, floor_level):
    """Calculate floor area based on classified elements"""
    
    total_area = 0
    deductions = 0
    partial_areas = 0
    
    for elem in ifc_elements:
        features = extract_geometry_features(elem)
        function, confidence, area_rule = classify_space_for_area_calculation(features)
        
        if area_rule == 'INCLUDE_FULL':
            total_area += features['area']
        elif area_rule == 'INCLUDE_PARTIAL_50':
            partial_areas += features['area'] * 0.5
        elif area_rule == 'EXCLUDE_SHAFT':
            deductions += features['area']
        elif area_rule == 'EXCLUDE_VOID':
            deductions += features['area']
        # EXCLUDE rules don't add to total
    
    net_floor_area = total_area + partial_areas - deductions
    
    return {
        'gross_area': total_area + partial_areas,
        'deductions': deductions,
        'net_floor_area': net_floor_area,
        'breakdown': get_detailed_breakdown(ifc_elements)
    }
```

---

## Test Set Usage for Area Calculation System

### 1. Training ML Classifier
- Use Group A (30 samples) as baseline labeled data
- Use Group B (40 samples) for supervised learning
- Validate on Group C & D for edge cases

### 2. Accuracy Metrics
- **Group A target**: >95% accuracy (clear attributes)
- **Group B target**: >80% accuracy (geometry inference)
- **Group C target**: >75% accuracy (opening elements)
- **Group D target**: >70% accuracy (cross-type)

### 3. Area Calculation Validation
- Compare calculated GFA with architectural drawings
- Validate shaft/void deductions
- Check partial area inclusion (balconies, parking)
- Verify wall centerline calculations

### 4. Edge Case Handling
- Ambiguity threshold: Flag for manual review if confidence <70%
- Cross-type conflicts: Prioritize IfcSpace over IfcOpeningElement
- Missing attributes: Default to geometry-based classification