# LLM Regulation Analysis Agent 使用说明

本Agent使用 LLM（默认 OpenAI `gpt-4o-mini`，可通过 `OPENAI_MODEL` 指定）解析不同地区法规，输出建筑面积（GFA）计算的差异对比：
- 层高规则（全面积/半面积/不计入 + 阈值）
- 顶盖及围护（阳台/雨篷/屋顶/连廊/电梯井/楼梯/飘窗/中庭等）
- 使用功能/特殊部位（停车场/地下室/夹层/阁楼/设备房/垃圾房等）

与正则版相比，LLM版能更快适配不同写法、上下文条件与跨语种条款。

## 依赖与环境
- Python 3.9+
- `openai>=1.0.0`
- 环境变量：
  - `OPENAI_API_KEY`: 必填
  - `OPENAI_BASE_URL`: 可选，自定义代理/网关
  - `OPENAI_MODEL`: 可选，默认 `gpt-4o-mini`

## 安装
```bash
pip install openai>=1.0.0
```

## 输入JSON格式（与正则版一致）
```json
[
  {"region": "CN", "source_name": "规范名", "text": "法规原文…"},
  {"region": "HK", "file": "/abs/path/to/hk.txt"},
  {"region": "US", "text": "…"},
  {"region": "EU", "text": "…"}
]
```

## 运行
```bash
export OPENAI_API_KEY=YOUR_KEY
python /Users/zhuxueying/ifc/llm_regulation_agent.py \
  --input /Users/zhuxueying/ifc/reg_inputs_example.json \
  --out-json /Users/zhuxueying/ifc/llm_reg_result.json \
  --out-md /Users/zhuxueying/ifc/llm_reg_result.md \
  --model gpt-4o-mini
```

## 输出
- JSON：各地区抽取结果及对比，包含 `evidence` 的原文证据片段
- Markdown：三大类规则的对比表

## 方法
- 以结构化 JSON schema 要求模型输出，减少幻觉与便于解析
- 超长文本自动分块并合并结果
- 后处理阶段复用一致的对比与 Markdown 生成逻辑

## 提示
- 提供尽可能完整的与GFA相关章节原文，准确率更高
- 若某些条款高度“条件化”，模型可能给出 `conditional` 并附注 `notes`
- 可结合正则版作为校核：先用LLM快速覆盖，再用正则版做确定性验证 