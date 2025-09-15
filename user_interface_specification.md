# 三Agent建筑面积计算框架 - 用户界面技术规范

## 1. 整体架构设计

### 1.1 前端技术栈
```
┌─────────────────────────────────────────────────────────────┐
│                    前端架构                                  │
├─────────────────────────────────────────────────────────────┤
│ UI框架: React 18 + TypeScript                               │
│ 状态管理: Redux Toolkit + RTK Query                         │
│ 样式: Tailwind CSS + Ant Design                            │
│ 3D可视化: Three.js + React Three Fiber                     │
│ 图表: Recharts + D3.js                                     │
│ 文件处理: react-dropzone                                    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 后端API设计
```
┌─────────────────────────────────────────────────────────────┐
│                    API端点设计                               │
├─────────────────────────────────────────────────────────────┤
│ POST /api/v1/projects                    # 创建新项目       │
│ POST /api/v1/projects/{id}/files         # 上传文件         │
│ POST /api/v1/projects/{id}/analyze       # 开始分析         │
│ GET  /api/v1/projects/{id}/status        # 获取进度         │
│ GET  /api/v1/projects/{id}/results       # 获取结果         │
│ POST /api/v1/projects/{id}/review        # 人工审查         │
│ GET  /api/v1/projects/{id}/reports       # 导出报告         │
└─────────────────────────────────────────────────────────────┘
```

## 2. 核心界面组件设计

### 2.1 项目初始化组件 (ProjectInitializer)

```typescript
interface ProjectInitializerProps {
  onProjectCreate: (project: ProjectConfig) => void;
}

interface ProjectConfig {
  name: string;
  buildingType: 'residential' | 'commercial' | 'industrial' | 'mixed';
  region: 'china' | 'hongkong' | 'europe' | 'us';
  ifcFile: File;
  regulationFiles: File[];
}

// 组件状态
interface ProjectInitializerState {
  config: Partial<ProjectConfig>;
  uploadProgress: Record<string, number>;
  validationErrors: string[];
  isSubmitting: boolean;
}
```

**界面布局:**
```jsx
<div className="project-initializer">
  <FileUploadZone 
    accept=".ifc"
    onUpload={handleIfcUpload}
    progress={uploadProgress.ifc}
  />
  
  <RegulationFilesUpload 
    multiple
    accept=".pdf,.docx"
    onUpload={handleRegulationUpload}
    progress={uploadProgress.regulations}
  />
  
  <ProjectConfigForm 
    config={config}
    onChange={handleConfigChange}
    errors={validationErrors}
  />
  
  <ActionButtons 
    onSubmit={handleSubmit}
    onReset={handleReset}
    disabled={isSubmitting}
  />
</div>
```

### 2.2 进度监控组件 (ProgressMonitor)

```typescript
interface AgentProgress {
  agentId: 'regulation' | 'alignment' | 'calculation';
  agentName: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number; // 0-100
  currentTask: string;
  estimatedTimeRemaining: number; // seconds
  logs: LogEntry[];
}

interface LogEntry {
  timestamp: string;
  level: 'info' | 'warning' | 'error';
  message: string;
  details?: any;
}

interface ProgressMonitorState {
  agents: AgentProgress[];
  overallProgress: number;
  isPolling: boolean;
}
```

**界面布局:**
```jsx
<div className="progress-monitor">
  <OverallProgressBar progress={overallProgress} />
  
  <div className="agents-grid">
    {agents.map(agent => (
      <AgentCard 
        key={agent.agentId}
        agent={agent}
        onViewLogs={() => setSelectedAgent(agent.agentId)}
      />
    ))}
  </div>
  
  <RealTimeLogViewer 
    logs={selectedAgent ? getAgentLogs(selectedAgent) : []}
    autoScroll={true}
  />
</div>
```

### 2.3 结果展示组件 (ResultsViewer)

```typescript
interface CalculationResults {
  totalArea: number;
  floorDetails: FloorDetail[];
  summary: AreaSummary;
  qualityMetrics: QualityMetrics;
  reviewItems: ReviewItem[];
}

interface FloorDetail {
  floorId: string;
  floorName: string;
  elevation: number;
  height: number;
  coefficient: number;
  slabArea: number;
  calculatedArea: number;
  elements: ElementDetail[];
}

interface QualityMetrics {
  confidenceDistribution: {
    high: number;
    medium: number;
    low: number;
  };
  reviewRequired: number;
  potentialVariance: {
    min: number;
    max: number;
  };
}
```

**界面布局:**
```jsx
<div className="results-viewer">
  <ResultsSummaryCard 
    totalArea={results.totalArea}
    qualityMetrics={results.qualityMetrics}
  />
  
  <Tabs>
    <TabPane tab="分层明细" key="floors">
      <FloorDetailsTable 
        floors={results.floorDetails}
        onFloorSelect={handleFloorSelect}
      />
    </TabPane>
    
    <TabPane tab="3D可视化" key="3d">
      <Building3DViewer 
        ifcModel={ifcModel}
        areaAnnotations={areaAnnotations}
        onElementSelect={handleElementSelect}
      />
    </TabPane>
    
    <TabPane tab="质量分析" key="quality">
      <QualityAnalysisCharts 
        metrics={results.qualityMetrics}
        reviewItems={results.reviewItems}
      />
    </TabPane>
  </Tabs>
  
  <ActionToolbar 
    onExport={handleExport}
    onReview={handleReview}
    onRecalculate={handleRecalculate}
  />
</div>
```

## 3. 数据流和状态管理

### 3.1 Redux Store结构

```typescript
interface RootState {
  project: {
    current: Project | null;
    config: ProjectConfig;
    status: 'idle' | 'uploading' | 'analyzing' | 'completed' | 'error';
  };
  
  agents: {
    regulation: AgentState;
    alignment: AgentState;
    calculation: AgentState;
  };
  
  results: {
    regulation: RegulationAnalysisResult | null;
    alignment: AlignmentResult | null;
    calculation: CalculationResult | null;
  };
  
  ui: {
    activeStep: number;
    selectedFloor: string | null;
    selectedElement: string | null;
    showReviewPanel: boolean;
  };
}

interface AgentState {
  status: AgentStatus;
  progress: number;
  currentTask: string;
  logs: LogEntry[];
  error: string | null;
}
```

### 3.2 实时数据更新

```typescript
// WebSocket连接管理
class ProgressWebSocket {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect(projectId: string) {
    this.ws = new WebSocket(`ws://localhost:8000/ws/projects/${projectId}`);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      store.dispatch(updateAgentProgress(data));
    };
    
    this.ws.onclose = () => {
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        setTimeout(() => this.connect(projectId), 2000);
        this.reconnectAttempts++;
      }
    };
  }
}

// RTK Query API定义
const projectApi = createApi({
  reducerPath: 'projectApi',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api/v1/',
  }),
  tagTypes: ['Project', 'Results'],
  endpoints: (builder) => ({
    createProject: builder.mutation<Project, ProjectConfig>({
      query: (config) => ({
        url: 'projects',
        method: 'POST',
        body: config,
      }),
      invalidatesTags: ['Project'],
    }),
    
    getProjectStatus: builder.query<ProjectStatus, string>({
      query: (projectId) => `projects/${projectId}/status`,
      providesTags: ['Project'],
    }),
    
    getResults: builder.query<CalculationResults, string>({
      query: (projectId) => `projects/${projectId}/results`,
      providesTags: ['Results'],
    }),
  }),
});
```

## 4. 用户交互流程

### 4.1 文件上传流程

```typescript
const handleFileUpload = async (files: File[], type: 'ifc' | 'regulation') => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  
  try {
    const response = await fetch(`/api/v1/projects/${projectId}/files`, {
      method: 'POST',
      body: formData,
      onUploadProgress: (progressEvent) => {
        const progress = (progressEvent.loaded / progressEvent.total) * 100;
        dispatch(updateUploadProgress({ type, progress }));
      },
    });
    
    if (response.ok) {
      dispatch(setUploadComplete({ type, files }));
      showNotification('文件上传成功', 'success');
    }
  } catch (error) {
    dispatch(setUploadError({ type, error: error.message }));
    showNotification('文件上传失败', 'error');
  }
};
```

### 4.2 分析启动流程

```typescript
const startAnalysis = async () => {
  try {
    // 1. 验证输入
    const validation = validateProjectConfig(projectConfig);
    if (!validation.isValid) {
      showValidationErrors(validation.errors);
      return;
    }
    
    // 2. 启动分析
    const response = await projectApi.startAnalysis(projectId).unwrap();
    
    // 3. 建立WebSocket连接
    progressWebSocket.connect(projectId);
    
    // 4. 开始轮询状态
    dispatch(startPolling(projectId));
    
    // 5. 更新UI状态
    dispatch(setAnalysisStarted());
    
  } catch (error) {
    showNotification('启动分析失败', 'error');
  }
};
```

### 4.3 人工审查流程

```typescript
interface ReviewItem {
  elementId: string;
  elementType: string;
  issue: string;
  suggestions: string[];
  currentClassification: string;
  confidence: number;
}

const ReviewPanel: React.FC<{ items: ReviewItem[] }> = ({ items }) => {
  const [reviewDecisions, setReviewDecisions] = useState<Record<string, string>>({});
  
  const handleReviewDecision = (elementId: string, decision: string) => {
    setReviewDecisions(prev => ({ ...prev, [elementId]: decision }));
  };
  
  const submitReview = async () => {
    try {
      await projectApi.submitReview({
        projectId,
        decisions: reviewDecisions,
      }).unwrap();
      
      // 重新计算
      dispatch(startRecalculation());
      
    } catch (error) {
      showNotification('提交审查失败', 'error');
    }
  };
  
  return (
    <div className="review-panel">
      {items.map(item => (
        <ReviewItemCard 
          key={item.elementId}
          item={item}
          onDecision={(decision) => handleReviewDecision(item.elementId, decision)}
        />
      ))}
      
      <Button 
        type="primary" 
        onClick={submitReview}
        disabled={Object.keys(reviewDecisions).length !== items.length}
      >
        提交审查结果
      </Button>
    </div>
  );
};
```

## 5. 3D可视化集成

### 5.1 IFC模型加载

```typescript
import { IFCLoader } from 'web-ifc-three';
import { Scene, PerspectiveCamera, WebGLRenderer } from 'three';

class IFCViewer {
  private scene: Scene;
  private camera: PerspectiveCamera;
  private renderer: WebGLRenderer;
  private loader: IFCLoader;
  
  constructor(container: HTMLElement) {
    this.initializeScene(container);
    this.loader = new IFCLoader();
  }
  
  async loadIFC(url: string) {
    const model = await this.loader.loadAsync(url);
    this.scene.add(model);
    return model;
  }
  
  highlightElements(elementIds: string[], color: string) {
    elementIds.forEach(id => {
      const element = this.findElementById(id);
      if (element) {
        element.material.color.set(color);
      }
    });
  }
  
  addAreaAnnotations(annotations: AreaAnnotation[]) {
    annotations.forEach(annotation => {
      const label = this.createAreaLabel(annotation);
      this.scene.add(label);
    });
  }
}
```

### 5.2 面积标注显示

```typescript
interface AreaAnnotation {
  elementId: string;
  position: [number, number, number];
  area: number;
  classification: string;
  confidence: number;
}

const AreaAnnotationComponent: React.FC<{ annotation: AreaAnnotation }> = ({ annotation }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  const color = useMemo(() => {
    if (annotation.confidence > 0.8) return '#4CAF50'; // 绿色 - 高置信度
    if (annotation.confidence > 0.5) return '#FF9800'; // 橙色 - 中等置信度
    return '#F44336'; // 红色 - 低置信度
  }, [annotation.confidence]);
  
  return (
    <group position={annotation.position}>
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.1]} />
        <meshBasicMaterial color={color} />
      </mesh>
      
      <Html>
        <div className="area-label">
          <div className="area-value">{annotation.area.toFixed(2)} ㎡</div>
          <div className="classification">{annotation.classification}</div>
          <div className="confidence">{(annotation.confidence * 100).toFixed(0)}%</div>
        </div>
      </Html>
    </group>
  );
};
```

## 6. 报告生成和导出

### 6.1 报告模板

```typescript
interface ReportTemplate {
  type: 'pdf' | 'excel' | 'word';
  sections: ReportSection[];
  styling: ReportStyling;
}

interface ReportSection {
  title: string;
  content: 'summary' | 'floor_details' | 'quality_analysis' | 'review_items';
  includeCharts: boolean;
  include3DViews: boolean;
}

const generateReport = async (results: CalculationResults, template: ReportTemplate) => {
  const reportGenerator = new ReportGenerator(template);
  
  // 添加封面
  reportGenerator.addCoverPage({
    title: '建筑面积计算报告',
    projectName: project.name,
    date: new Date().toLocaleDateString(),
    totalArea: results.totalArea,
  });
  
  // 添加汇总信息
  reportGenerator.addSummarySection(results.summary);
  
  // 添加分层明细
  reportGenerator.addFloorDetailsSection(results.floorDetails);
  
  // 添加质量分析
  reportGenerator.addQualityAnalysisSection(results.qualityMetrics);
  
  // 添加3D截图
  if (template.sections.some(s => s.include3DViews)) {
    const screenshots = await capture3DViews();
    reportGenerator.add3DViewsSection(screenshots);
  }
  
  return reportGenerator.generate();
};
```

这个技术规范提供了完整的用户界面实现方案，确保用户能够直观地了解和控制整个建筑面积计算流程。