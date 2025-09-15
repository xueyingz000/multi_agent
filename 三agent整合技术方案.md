# 三Agent联合Framework技术实现方案

## 系统架构设计

### 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web前端界面                              │
│                    (React + TypeScript)                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP/WebSocket
┌─────────────────────▼───────────────────────────────────────────┐
│                    API网关层                                    │
│                 (FastAPI + Nginx)                              │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  业务编排层                                     │
│               (Workflow Engine)                                │
├─────────────────────┼───────────────────────────────────────────┤
│  Agent 1            │  Agent 2            │  Agent 3            │
│  LLM Regulation     │  Semantic           │  Building Area      │
│  Agent              │  Alignment Agent    │  Agent              │
│                     │                     │                     │
│  ┌─────────────┐    │  ┌─────────────┐    │  ┌─────────────┐    │
│  │ PDF Parser  │    │  │ IFC Loader  │    │  │ Area Calc   │    │
│  │ LLM Client  │    │  │ Semantic    │    │  │ Result Gen  │    │
│  │ Rule Extract│    │  │ Matcher     │    │  │ Report Gen  │    │
│  └─────────────┘    │  │ Confidence  │    │  └─────────────┘    │
│                     │  │ Evaluator   │    │                     │
│                     │  └─────────────┘    │                     │
└─────────────────────┼───────────────────────────────────────────┤
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    数据存储层                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │
│  │ PostgreSQL  │ │ Redis Cache │ │ File Storage│ │ Vector DB   │ │
│  │ (项目数据)  │ │ (会话状态)  │ │ (文件存储)  │ │ (语义搜索)  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2. 核心组件设计

#### 2.1 业务编排层 (Workflow Engine)

```python
# workflow_engine.py
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass
import asyncio
import json

class WorkflowStage(Enum):
    REGULATION_ANALYSIS = "regulation_analysis"
    SEMANTIC_ALIGNMENT = "semantic_alignment"
    AREA_CALCULATION = "area_calculation"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowContext:
    project_id: str
    current_stage: WorkflowStage
    regulation_result: Dict[str, Any] = None
    alignment_result: Dict[str, Any] = None
    calculation_result: Dict[str, Any] = None
    error_message: str = None
    user_inputs: Dict[str, Any] = None

class WorkflowEngine:
    def __init__(self):
        self.regulation_agent = LLMRegulationAgent()
        self.alignment_agent = SemanticAlignmentAgent()
        self.area_agent = BuildingAreaAgent()
        self.contexts: Dict[str, WorkflowContext] = {}
    
    async def start_workflow(self, project_id: str, 
                           regulation_files: List[str],
                           ifc_file: str,
                           user_params: Dict[str, Any]) -> WorkflowContext:
        """启动三阶段工作流"""
        context = WorkflowContext(
            project_id=project_id,
            current_stage=WorkflowStage.REGULATION_ANALYSIS,
            user_inputs=user_params
        )
        self.contexts[project_id] = context
        
        try:
            # 第一阶段：法规分析
            await self._execute_regulation_analysis(context, regulation_files)
            
            # 第二阶段：语义对齐
            await self._execute_semantic_alignment(context, ifc_file)
            
            # 第三阶段：面积计算
            await self._execute_area_calculation(context)
            
            context.current_stage = WorkflowStage.COMPLETED
            
        except Exception as e:
            context.current_stage = WorkflowStage.FAILED
            context.error_message = str(e)
            
        return context
    
    async def _execute_regulation_analysis(self, context: WorkflowContext, 
                                         regulation_files: List[str]):
        """执行法规分析阶段"""
        context.current_stage = WorkflowStage.REGULATION_ANALYSIS
        
        # 调用regulation agent
        regulation_inputs = [
            {"region": context.user_inputs.get("region", "HK"),
             "source_path": file_path,
             "text": self._extract_pdf_text(file_path)}
            for file_path in regulation_files
        ]
        
        result = await self.regulation_agent.analyze_async(regulation_inputs)
        context.regulation_result = result
        
        # 保存中间结果
        await self._save_intermediate_result(context.project_id, 
                                           "regulation", result)
    
    async def _execute_semantic_alignment(self, context: WorkflowContext, 
                                        ifc_file: str):
        """执行语义对齐阶段"""
        context.current_stage = WorkflowStage.SEMANTIC_ALIGNMENT
        
        # 调用alignment agent
        result = await self.alignment_agent.align_async(
            ifc_file_path=ifc_file,
            regulation_rules=context.regulation_result,
            building_type=context.user_inputs.get("building_type", "office"),
            user_confirmations=context.user_inputs.get("confirmations", {})
        )
        
        context.alignment_result = result
        
        # 保存中间结果
        await self._save_intermediate_result(context.project_id, 
                                           "alignment", result)
    
    async def _execute_area_calculation(self, context: WorkflowContext):
        """执行面积计算阶段"""
        context.current_stage = WorkflowStage.AREA_CALCULATION
        
        # 调用area calculation agent
        result = await self.area_agent.calculate_async(
            regulation_rules=context.regulation_result,
            alignment_report=context.alignment_result["alignment_report"],
            classification_summary=context.alignment_result["classification_summary"]
        )
        
        context.calculation_result = result
        
        # 保存最终结果
        await self._save_final_result(context.project_id, result)
```

#### 2.2 统一的Agent接口

```python
# base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio

class BaseAgent(ABC):
    """所有Agent的基类"""
    
    def __init__(self, llm_client, config: Dict[str, Any]):
        self.llm_client = llm_client
        self.config = config
        self.logger = self._setup_logger()
    
    @abstractmethod
    async def process_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理方法"""
        pass
    
    @abstractmethod
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """输入验证"""
        pass
    
    def _setup_logger(self):
        # 设置日志记录
        pass

# 改造后的Regulation Agent
class LLMRegulationAgentV2(BaseAgent):
    async def process_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理法规分析"""
        regulation_files = inputs["regulation_files"]
        region = inputs.get("region", "HK")
        
        # 并行处理多个法规文件
        tasks = []
        for file_info in regulation_files:
            task = self._analyze_single_regulation(file_info)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # 合并结果
        merged_result = self._merge_regulation_results(results)
        return merged_result
    
    async def _analyze_single_regulation(self, file_info: Dict[str, Any]):
        # 单个法规文件的分析逻辑
        pass

# 改造后的Semantic Alignment Agent
class SemanticAlignmentAgentV2(BaseAgent):
    async def process_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理语义对齐"""
        ifc_file_path = inputs["ifc_file_path"]
        regulation_rules = inputs["regulation_rules"]
        
        # 1. 加载IFC文件
        ifc_elements = await self._load_ifc_async(ifc_file_path)
        
        # 2. 并行处理元素分类
        classification_tasks = []
        for element in ifc_elements:
            task = self._classify_element_async(element, regulation_rules)
            classification_tasks.append(task)
        
        classifications = await asyncio.gather(*classification_tasks)
        
        # 3. 生成对齐报告
        alignment_report = self._generate_alignment_report(classifications)
        classification_summary = self._generate_classification_summary(classifications)
        
        return {
            "alignment_report": alignment_report,
            "classification_summary": classification_summary
        }

# 改造后的Building Area Agent
class BuildingAreaAgentV2(BaseAgent):
    async def process_async(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """异步处理面积计算"""
        regulation_rules = inputs["regulation_rules"]
        alignment_report = inputs["alignment_report"]
        classification_summary = inputs["classification_summary"]
        
        # 1. 按规则分类元素
        categorized_elements = self._categorize_elements_by_rules(
            classification_summary, regulation_rules
        )
        
        # 2. 并行计算各类面积
        area_tasks = [
            self._calculate_full_area_async(categorized_elements["full"]),
            self._calculate_half_area_async(categorized_elements["half"]),
            self._calculate_excluded_area_async(categorized_elements["excluded"])
        ]
        
        full_area, half_area, excluded_area = await asyncio.gather(*area_tasks)
        
        # 3. 生成最终结果
        total_area = full_area + (half_area * 0.5)
        
        return {
            "total_area": total_area,
            "full_area": full_area,
            "half_area": half_area,
            "excluded_area": excluded_area,
            "detailed_breakdown": self._generate_detailed_breakdown(
                categorized_elements, full_area, half_area, excluded_area
            )
        }
```

#### 2.3 API层设计

```python
# api/main.py
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uuid

app = FastAPI(title="Building Area Calculation Framework")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

workflow_engine = WorkflowEngine()

@app.post("/api/projects")
async def create_project(project_data: Dict[str, Any]):
    """创建新项目"""
    project_id = str(uuid.uuid4())
    
    # 保存项目基本信息
    await save_project_info(project_id, project_data)
    
    return {"project_id": project_id, "status": "created"}

@app.post("/api/projects/{project_id}/files/regulation")
async def upload_regulation_files(project_id: str, 
                                 files: List[UploadFile] = File(...)):
    """上传法规文件"""
    file_paths = []
    for file in files:
        file_path = await save_uploaded_file(file, "regulation", project_id)
        file_paths.append(file_path)
    
    return {"uploaded_files": file_paths}

@app.post("/api/projects/{project_id}/files/ifc")
async def upload_ifc_file(project_id: str, file: UploadFile = File(...)):
    """上传IFC文件"""
    file_path = await save_uploaded_file(file, "ifc", project_id)
    return {"ifc_file_path": file_path}

@app.post("/api/projects/{project_id}/start")
async def start_calculation(project_id: str, 
                          background_tasks: BackgroundTasks,
                          params: Dict[str, Any]):
    """启动计算流程"""
    # 获取项目文件路径
    regulation_files = await get_project_regulation_files(project_id)
    ifc_file = await get_project_ifc_file(project_id)
    
    # 后台启动工作流
    background_tasks.add_task(
        workflow_engine.start_workflow,
        project_id, regulation_files, ifc_file, params
    )
    
    return {"status": "started", "project_id": project_id}

@app.get("/api/projects/{project_id}/status")
async def get_project_status(project_id: str):
    """获取项目状态"""
    context = workflow_engine.contexts.get(project_id)
    if not context:
        return {"error": "Project not found"}
    
    return {
        "project_id": project_id,
        "current_stage": context.current_stage.value,
        "regulation_completed": context.regulation_result is not None,
        "alignment_completed": context.alignment_result is not None,
        "calculation_completed": context.calculation_result is not None,
        "error_message": context.error_message
    }

@app.get("/api/projects/{project_id}/results/{stage}")
async def get_stage_results(project_id: str, stage: str):
    """获取特定阶段的结果"""
    context = workflow_engine.contexts.get(project_id)
    if not context:
        return {"error": "Project not found"}
    
    if stage == "regulation":
        return context.regulation_result
    elif stage == "alignment":
        return context.alignment_result
    elif stage == "calculation":
        return context.calculation_result
    else:
        return {"error": "Invalid stage"}

@app.post("/api/projects/{project_id}/review")
async def submit_review(project_id: str, review_data: Dict[str, Any]):
    """提交人工审核结果"""
    context = workflow_engine.contexts.get(project_id)
    if not context:
        return {"error": "Project not found"}
    
    # 更新用户确认信息
    if "confirmations" not in context.user_inputs:
        context.user_inputs["confirmations"] = {}
    
    context.user_inputs["confirmations"].update(review_data)
    
    # 如果在对齐阶段，重新执行后续步骤
    if context.current_stage == WorkflowStage.SEMANTIC_ALIGNMENT:
        await workflow_engine._execute_semantic_alignment(
            context, await get_project_ifc_file(project_id)
        )
        await workflow_engine._execute_area_calculation(context)
    
    return {"status": "review_submitted"}

# WebSocket支持实时更新
from fastapi import WebSocket

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str):
    await websocket.accept()
    
    # 实时推送项目状态更新
    while True:
        context = workflow_engine.contexts.get(project_id)
        if context:
            await websocket.send_json({
                "type": "status_update",
                "stage": context.current_stage.value,
                "timestamp": datetime.now().isoformat()
            })
        
        await asyncio.sleep(1)  # 每秒更新一次
```

#### 2.4 前端集成方案

```typescript
// frontend/src/services/api.ts
class BuildingAreaAPI {
  private baseURL = process.env.REACT_APP_API_BASE_URL;
  private ws: WebSocket | null = null;
  
  async createProject(projectData: ProjectData): Promise<string> {
    const response = await fetch(`${this.baseURL}/api/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(projectData)
    });
    const result = await response.json();
    return result.project_id;
  }
  
  async uploadRegulationFiles(projectId: string, files: FileList): Promise<string[]> {
    const formData = new FormData();
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });
    
    const response = await fetch(
      `${this.baseURL}/api/projects/${projectId}/files/regulation`,
      { method: 'POST', body: formData }
    );
    const result = await response.json();
    return result.uploaded_files;
  }
  
  async startCalculation(projectId: string, params: CalculationParams): Promise<void> {
    await fetch(`${this.baseURL}/api/projects/${projectId}/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params)
    });
  }
  
  connectWebSocket(projectId: string, onUpdate: (data: any) => void): void {
    this.ws = new WebSocket(`${this.baseURL.replace('http', 'ws')}/ws/${projectId}`);
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onUpdate(data);
    };
  }
  
  async getStageResults(projectId: string, stage: string): Promise<any> {
    const response = await fetch(
      `${this.baseURL}/api/projects/${projectId}/results/${stage}`
    );
    return await response.json();
  }
}

// frontend/src/components/WorkflowManager.tsx
import React, { useState, useEffect } from 'react';
import { BuildingAreaAPI } from '../services/api';

const WorkflowManager: React.FC = () => {
  const [currentStage, setCurrentStage] = useState<string>('regulation_analysis');
  const [projectId, setProjectId] = useState<string>('');
  const [results, setResults] = useState<any>({});
  const api = new BuildingAreaAPI();
  
  useEffect(() => {
    if (projectId) {
      // 连接WebSocket获取实时更新
      api.connectWebSocket(projectId, (data) => {
        setCurrentStage(data.stage);
        // 更新UI状态
      });
    }
  }, [projectId]);
  
  const handleStageComplete = async (stage: string) => {
    const stageResults = await api.getStageResults(projectId, stage);
    setResults(prev => ({ ...prev, [stage]: stageResults }));
  };
  
  return (
    <div className="workflow-manager">
      <StageIndicator currentStage={currentStage} />
      
      {currentStage === 'regulation_analysis' && (
        <RegulationAnalysisStage 
          onComplete={() => handleStageComplete('regulation')}
        />
      )}
      
      {currentStage === 'semantic_alignment' && (
        <SemanticAlignmentStage 
          regulationResults={results.regulation}
          onComplete={() => handleStageComplete('alignment')}
        />
      )}
      
      {currentStage === 'area_calculation' && (
        <AreaCalculationStage 
          alignmentResults={results.alignment}
          onComplete={() => handleStageComplete('calculation')}
        />
      )}
    </div>
  );
};
```

### 3. 部署方案

#### 3.1 Docker容器化

```dockerfile
# Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# Dockerfile.frontend
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

#### 3.2 Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/building_area
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    volumes:
      - ./uploads:/app/uploads
  
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
  
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=building_area
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend

volumes:
  postgres_data:
  redis_data:
```

### 4. 性能优化策略

#### 4.1 异步处理
- 所有Agent都支持异步处理
- 使用消息队列处理长时间运行的任务
- WebSocket实现实时状态更新

#### 4.2 缓存策略
- Redis缓存中间结果
- 文件上传后的预处理结果缓存
- LLM调用结果缓存

#### 4.3 并发处理
- 多个法规文件并行分析
- IFC元素并行分类
- 面积计算并行执行

#### 4.4 资源管理
- 连接池管理数据库连接
- 文件上传大小限制
- 内存使用监控和清理

### 5. 监控和日志

```python
# monitoring.py
import logging
from prometheus_client import Counter, Histogram, Gauge
import time

# 定义监控指标
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_PROJECTS = Gauge('active_projects_total', 'Number of active projects')
AGENT_PROCESSING_TIME = Histogram('agent_processing_seconds', 'Agent processing time', ['agent_type'])

class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # 记录请求
            REQUEST_COUNT.labels(
                method=scope["method"], 
                endpoint=scope["path"]
            ).inc()
            
            # 执行请求
            await self.app(scope, receive, send)
            
            # 记录处理时间
            REQUEST_DURATION.observe(time.time() - start_time)
        else:
            await self.app(scope, receive, send)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

这个技术方案提供了一个完整的、可扩展的三Agent联合framework实现方案，包括了系统架构、核心组件、API设计、前端集成、部署方案和性能优化等各个方面。