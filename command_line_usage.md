# 命令行使用说明

## 基本用法

### 交互式运行（推荐）
```bash
python building_area_agent122.py
```

程序会提示您输入IFC文件和regulation文件的路径：
```
🔧 Multi-Agent Building Area Calculation Setup
--------------------------------------------------
📁 Please enter the path to your IFC file:
   IFC File Path: /path/to/your/building.ifc
   ✅ IFC file received: /path/to/your/building.ifc

📋 Please enter the path to your regulation file:
   Regulation File Path: /path/to/your/regulation.json
   ✅ Regulation file received: /path/to/your/regulation.json

🔄 Processing your inputs...
📁 Loading IFC file: /path/to/your/building.ifc
📋 Loading regulation file: /path/to/your/regulation.json
```

**注意：** 为了演示目的，程序会显示您输入的文件路径，但实际处理时会使用预设的默认文件。

### 使用命令行参数运行
```bash
python building_area_agent122.py --ifc /path/to/file.ifc --regulation /path/to/regulation.json
```

### 指定IFC文件
```bash
python building_area_agent122.py --ifc /path/to/your/file.ifc
# 或使用短参数
python building_area_agent122.py -i /path/to/your/file.ifc
```

### 指定regulation文件
```bash
python building_area_agent122.py --regulation /path/to/your/regulation.json
# 或使用短参数
python building_area_agent122.py -r /path/to/your/regulation.json
```

### 指定API密钥
```bash
python building_area_agent122.py --api-key your-openai-api-key
# 或使用短参数
python building_area_agent122.py -k your-openai-api-key
```

### 同时指定多个参数
```bash
python building_area_agent122.py \
  --ifc /Users/zhuxueying/projects/building1.ifc \
  --regulation /Users/zhuxueying/regulations/china_rules.json \
  --api-key sk-your-actual-api-key
```

## 参数说明

| 参数 | 短参数 | 默认值 | 说明 |
|------|--------|--------|------|
| `--ifc` | `-i` | `endtoend.ifc` | IFC文件路径 |
| `--regulation` | `-r` | `reg_result.json` | 规定文件路径 |
| `--api-key` | `-k` | `sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL` | OpenAI API密钥 |

## 查看帮助
```bash
python building_area_agent122.py --help
```

## 使用示例

### 示例1：交互式输入（演示模式）
```bash
python building_area_agent122.py
# 然后按提示输入文件路径（仅用于演示，实际使用默认文件）
```

### 示例2：分析特定建筑项目
```bash
python building_area_agent122.py \
  -i /Users/zhuxueying/projects/office_building.ifc \
  -r /Users/zhuxueying/regulations/commercial_building_rules.json
```

### 示例3：使用自定义API密钥
```bash
python building_area_agent122.py \
  -k sk-your-custom-api-key \
  -i building_model.ifc
```

## 注意事项

1. **交互模式**：默认运行时会要求用户输入文件路径，这是为了演示用户交互体验
2. **实际处理**：无论用户输入什么路径，程序实际使用的是预设的默认文件
3. **文件路径**：确保默认的IFC文件和regulation文件存在且可读
4. **API密钥**：请使用有效的OpenAI API密钥
5. **权限**：确保程序有权限读取指定的文件

## 错误处理

如果遇到错误，程序会显示详细的错误信息和堆栈跟踪，帮助诊断问题。常见错误包括：

- 文件不存在
- API密钥无效
- 文件格式错误
- 权限不足