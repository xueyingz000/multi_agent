# Regulation Analysis Agent 使用说明

本工具用于解析不同地区（中国大陆、香港、美国、欧洲等）的建筑法规文本，从以下三个方面抽取并对比建筑面积计算差异：

- 层高相关的计算规则（影响全面积/半面积/不计面积）
- 顶盖及围护结构（阳台、雨篷、屋顶、电梯井、楼梯间、飘窗、连廊、中庭采光井）
- 使用功能/特殊部位（停车场、地下室、夹层、阁楼、设备用房、垃圾房）

工具以正则+关键词的轻量解析为基础，输出结构化 JSON 与 Markdown 摘要。

## 安装/环境

Python 3.9+

无需额外依赖（仅标准库）。

## 输入JSON格式

输入文件为一个 list，每个元素代表一个地区的法规来源，支持直接提供文本或文件路径（任选其一）。

```json
[
  {
    "region": "CN",                      
    "source_name": "民用建筑设计通则(节选)", 
    "text": "……包含层高、阳台、屋顶、地下室等关于是否计入建筑面积的条文原文……"
  },
  {
    "region": "HK",
    "source_name": "HK PNAP APP-151",
    "file": "/abs/path/to/hk_gfa_rules.txt"
  },
  {
    "region": "US",
    "source_name": "IBC/GFA local ordinance",
    "text": "…"
  },
  {
    "region": "EU",
    "source_name": "某成员国建筑面积计算指引",
    "text": "…"
  }
]
```

- `region`: 推荐使用短代号（CN/HK/US/EU/…）。
- `text` 与 `file` 二选一；如果同时提供，优先采用 `text`。

## 运行

```bash
python /Users/zhuxueying/ifc/regulation_analysis_agent.py \
  --input /abs/path/to/inputs.json \
  --out-json /abs/path/to/result.json \
  --out-md /abs/path/to/result.md
```

执行后会在终端打印 Markdown 摘要，并可将结构化结果与 Markdown 分别写入指定文件。

## 输出

- `result.json`: 结构化结果，包含每个地区解析出的规则与跨区域对比摘要。
- `result.md`: Markdown 摘要，三部分：
  1. 层高相关的面积计算规则对比（按 `full/half/excluded/unknown` 分组并显示阈值）
  2. 顶盖及围护结构（阳台/雨篷/屋顶/电梯等）对比表
  3. 使用功能/特殊部位（停车场/地下室等）对比表

## 方法说明

- 通过关键词检测判断条文属于“计入/按一半/不计入/未知”。
- 对层高相关语句提取区间或阈值（支持m/米/ft/feet），统一换算为米。
- 对特征/部位采用同义词匹配，按优先级聚合标签：excluded > half > full > conditional > unknown。
- 每条规则保留原文证据窗口，便于回溯。

## 限制与扩展

- 当前为关键词+正则的轻量方案，对跨条款引用、表格、复杂条件逻辑的覆盖有限。
- 可进一步接入向量检索/LLM做条文定位与条件解析；或增加地区定制词典与规则模板。
- 可扩充特征词表，或对“屋顶平台、架空层、雨篷宽度阈值”等加入更细分的抽取逻辑。

## 最佳实践

- 尽量提供法规原文的纯文本版本，避免复杂排版。
- 每个地区尽量聚焦与GFA相关的章节，提高抽取精度。
- 若需要审计详细证据，可解析 `result.json` 中的 `evidence` 字段。 