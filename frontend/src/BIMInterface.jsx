import React, { useState, useRef, useEffect } from 'react';
import {
    Box, Layers, ChevronRight, ChevronDown, Search, Filter,
    Download, Upload, Maximize2, MousePointer2,
    Edit3, Scissors, RotateCcw, FileText, Info, Square,
    BookOpen, X, Loader2,
    // --- 新增图标 ---
    AlertTriangle, ListChecks, CheckCircle, BrainCircuit, Activity, Calculator, FileSpreadsheet
} from 'lucide-react';
import IfcViewer from './IfcViewer';

const API_BASE_URL = process.env.BACKEND_URL || "http://localhost:8000";

// --- 组件：结构树节点 (保持不变) ---
const TreeNode = ({ node, onSelectNode, depth = 0 }) => {
    // Default expansion: Only expand the root (Building), collapse everything else (Levels)
    const [expanded, setExpanded] = useState(depth < 1);
    const hasChildren = node.children && node.children.length > 0;

    const getIcon = (type) => {
        if (type === 'IFCPROJECT') return <Box size={14} className="text-purple-400" />;
        if (type === 'IFCSITE') return <Box size={14} className="text-orange-400" />;
        if (type === 'IFCBUILDING') return <Box size={14} className="text-blue-400" />;
        if (type === 'IFCBUILDINGSTOREY') return <Layers size={14} className="text-yellow-400" />;
        if (type === 'IFCSPACE') return <Square size={14} className="text-emerald-400 transform rotate-45" />;
        return <Box size={14} className="text-gray-500" />;
    };

    const getNodeName = (node) => {
        if (!node) return 'Untitled';

        let name = null;
        if (node.Name && node.Name.value) name = node.Name.value;
        else if (typeof node.Name === 'string' && node.Name.length > 0) name = node.Name;
        else if (node.name && typeof node.name === 'string') name = node.name;
        else if (node.LongName && node.LongName.value) name = node.LongName.value;

        // Handle specific "Default" cases or missing names
        if (!name || name === 'Default' || name === 'Untitled') {
            if (node.type === 'IFCPROJECT') return 'Project Model';
            if (node.type === 'IFCSITE') return 'Site';
            if (node.type === 'IFCBUILDING') return 'Building';
        }

        if (name) return name;

        // Fallback to Type + ID
        const typeStr = node.type || 'Element';
        const idStr = node.expressID || '?';
        return `${typeStr} (${idStr})`;
    };

    return (
        <div>
            <div
                className="flex items-center py-1 cursor-pointer hover:bg-[#2d333b] text-gray-400 hover:text-white transition-colors border-l-2 border-transparent hover:border-emerald-500"
                style={{ paddingLeft: `${depth * 12 + 8}px` }}
                onClick={() => setExpanded(!expanded)}
            >
                <span className="mr-1 w-4 h-4 flex items-center justify-center">
                    {hasChildren && (expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />)}
                </span>
                <span className="mr-2">{getIcon(node.type)}</span>
                <span className="text-xs truncate">{getNodeName(node)}</span>
            </div>
            {expanded && hasChildren && (
                <div>
                    {node.children.map((child) => (
                        <TreeNode key={child.expressID} node={child} onSelectNode={onSelectNode} depth={depth + 1} />
                    ))}
                </div>
            )}
        </div>
    );
};

const BIMInterface = () => {
    // --- 状态管理 ---
    const [ifcFile, setIfcFile] = useState(null);
    const [projectName, setProjectName] = useState("Untitled Project");
    const [treeData, setTreeData] = useState(null);
    const [selectedProps, setSelectedProps] = useState(null);
    const [selectedPsets, setSelectedPsets] = useState(null);

    // --- 新增：法规 Agent 相关状态 ---
    const [showRuleModal, setShowRuleModal] = useState(false); // 控制弹窗
    const [isAnalyzingRule, setIsAnalyzingRule] = useState(false); // Agent 1 思考中
    const [activeRules, setActiveRules] = useState(null); // Agent 1 的输出结果
    const [ruleFileName, setRuleFileName] = useState("Shanghai Standard (2024)"); // 当前法规名

    // --- 新增：语义 Agent 相关状态 ---
    const [isAnalyzingSemantic, setIsAnalyzingSemantic] = useState(false);
    const [semanticResults, setSemanticResults] = useState(null);
    const [reviewQueue, setReviewQueue] = useState([]); // 复核队列
    const [activeTab, setActiveTab] = useState('structure'); // 'structure' | 'review'
    const [agent2Data, setAgent2Data] = useState(null); // 当前选中构件的 Agent 2 分析数据
    const abortControllerRef = useRef(null); // 用于取消请求

    // --- 新增：Agent 3 (面积计算) 相关状态 ---
    const [isCalculating, setIsCalculating] = useState(false);
    const [calculationResults, setCalculationResults] = useState(null);
    const [selectedId, setSelectedId] = useState(null); // Highlighting ID for IfcViewer

    // 视口尺寸
    const viewportRef = useRef(null);
    const [viewportSize, setViewportSize] = useState({ w: 800, h: 600 });

    useEffect(() => {
        const handleResize = () => {
            if (viewportRef.current) {
                setViewportSize({ w: viewportRef.current.offsetWidth, h: viewportRef.current.offsetHeight });
            }
        };
        window.addEventListener('resize', handleResize);
        handleResize();
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // --- 修改：处理 IFC 上传 (同步传给后端) ---
    const handleFileUpload = async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setIfcFile(file); // 前端渲染
            // Set project name from filename (remove .ifc extension)
            const name = file.name.replace(/\.ifc$/i, '');
            setProjectName(name);

            // 后端同步上传
            const formData = new FormData();
            formData.append("file", file);
            try {
                const uploadUrl = `${API_BASE_URL}/upload/ifc`;
                console.log(`📤 Uploading IFC to backend: ${uploadUrl}`);
                await fetch(uploadUrl, { method: "POST", body: formData });
            } catch (err) {
                console.error("IFC upload failed (Check if server.py is running)", err);
            }
        }
    };

    // --- 新增：处理法规上传 (触发 Agent 1) ---
    const handleRegulationUpload = async (e) => {
        if (e.target.files && e.target.files[0]) {
            const file = e.target.files[0];
            setIsAnalyzingRule(true);
            setRuleFileName(file.name);

            const formData = new FormData();
            formData.append("file", file);
            formData.append("region_name", file.name);

            try {
                console.log("📤 Uploading PDF to Agent 1...");
                const res = await fetch(`${API_BASE_URL}/upload/regulation`, {
                    method: "POST",
                    body: formData
                });
                const json = await res.json();

                if (json.status === "success") {
                    console.log("✅ Agent 1 Analysis Result:", json.data);
                    setActiveRules(json.data); // 保存规则到状态
                }
            } catch (err) {
                console.error("Agent 1 failed:", err);
                alert("Regulation analysis failed. Check console.");
            } finally {
                setIsAnalyzingRule(false);
            }
        }
    };

    // --- 新增：触发 Agent 2 (语义分析) ---
    const handleSemanticAnalysis = async () => {
        // 如果正在分析，则视为“停止”操作
        if (isAnalyzingSemantic) {
            // 1. 中止前端请求
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
            }
            // 2. 发送后端停止信号
            try {
                await fetch(`${API_BASE_URL}/analyze/stop`, { method: "POST" });
                console.log("🛑 Stop signal sent to backend.");
            } catch (err) {
                console.error("Failed to send stop signal:", err);
            }
            setIsAnalyzingSemantic(false);
            return;
        }

        if (!ifcFile) {
            alert("Please upload an IFC file first.");
            return;
        }
        if (!activeRules) {
            alert("Please select/upload regulation rules first.");
            return;
        }

        setIsAnalyzingSemantic(true);
        setSemanticResults(null); // 清空旧结果
        setReviewQueue([]);

        // 创建新的 AbortController
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;

        try {
            console.log("🚀 Starting Agent 2 Semantic Analysis...");
            const res = await fetch(`${API_BASE_URL}/analyze/semantic`, {
                method: "POST",
                signal: signal // 绑定取消信号
            });
            const json = await res.json();

            if (json.status === "success") {
                console.log("✅ Semantic Analysis Complete:", json);
                setSemanticResults(json.meta);
                setReviewQueue(json.data.hitl_queue || []); // 填充队列
                if (json.data.hitl_queue?.length > 0) {
                    setActiveTab('review'); // 自动切换到复核 Tab
                }
                alert(json.message); // 显示完成或停止消息
            } else {
                alert("Analysis failed: " + json.message);
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                console.log("🛑 Semantic analysis aborted by user.");
                alert("Analysis stopped.");
            } else {
                console.error("Semantic analysis failed:", err);
                alert("Semantic analysis failed. Check console.");
            }
        } finally {
            setIsAnalyzingSemantic(false);
            abortControllerRef.current = null;
        }
    };

    const handleModelLoaded = (_, structure) => {
        if (!structure) return;

        // Helper function to find the first IFCBUILDING node
        const findBuilding = (node) => {
            if (node.type === 'IFCBUILDING') return node;
            if (node.children && node.children.length > 0) {
                for (let child of node.children) {
                    const found = findBuilding(child);
                    if (found) return found;
                }
            }
            return null;
        };

        const buildingNode = findBuilding(structure);

        if (buildingNode) {
            setTreeData(buildingNode);
        } else {
            // Fallback: If no building found, use root or its child
            setTreeData(structure.children && structure.children.length > 0 ? structure.children[0] : structure);
        }
    };

    const handleSelection = async (id, props) => {
        console.log("🖱️ 3D Selection:", id, props);
        setSelectedId(id); // Sync state
        setSelectedProps(props || null);
        setAgent2Data(null); // 重置之前的分析数据

        // 如果选中了构件，获取详细信息
        if (props && props.GlobalId) {
            try {
                console.log("🔍 Fetching analysis for:", props.GlobalId.value);
                // 调用后端 /analyze/element
                const res = await fetch(`${API_BASE_URL}/analyze/element`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ element_guid: props.GlobalId.value })
                });
                if (res.ok) {
                    const data = await res.json();
                    console.log("✅ Analysis data received:", data);
                    setAgent2Data(data);
                } else {
                    console.warn("⚠️ No analysis data found for this element.");
                }
            } catch (err) {
                console.error("Failed to fetch element analysis:", err);
            }
        }
    };

    // 新增：处理复核队列点击
    const handleReviewItemClick = async (item) => {
        console.log("📋 Review Item Clicked:", item);

        // Highlight in 3D Viewer
        if (item.express_id) {
            setSelectedId(item.express_id);
        }

        // 模拟选中属性（为了右侧面板能显示）
        const mockProps = {
            GlobalId: { value: item.guid },
            Name: { value: item.name },
            constructor: { name: 'IfcElement' } // 简化处理
        };
        setSelectedProps(mockProps);
        setAgent2Data(null);

        // 获取详细分析数据
        try {
            const res = await fetch(`${API_BASE_URL}/analyze/element`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ element_guid: item.guid })
            });
            if (res.ok) {
                const data = await res.json();
                console.log("✅ Analysis data received (from list):", data);
                setAgent2Data(data);
                // Also update express_id if backend returned it (just in case item didn't have it)
                if (data.express_id) setSelectedId(data.express_id);
            }
        } catch (err) {
            console.error("Failed to fetch element analysis:", err);
        }
    };

    // --- 新增：触发 Agent 3 (面积计算) ---
    const handleCalculateArea = async () => {
        if (!semanticResults) {
            alert("Please run Semantic Analysis first.");
            return;
        }

        setIsCalculating(true);
        try {
            console.log("🚀 Starting Area Calculation...");
            const res = await fetch(`${API_BASE_URL}/calculate/area`, { method: "POST" });
            const json = await res.json();

            if (json.status === "success") {
                console.log("✅ Calculation Complete:", json.data);
                setCalculationResults(json.data);
                setActiveTab('calculation'); // 自动切换到计算结果 Tab
            } else {
                alert("Calculation failed: " + json.message);
            }
        } catch (err) {
            console.error("Calculation error:", err);
            alert("Calculation error. Check console.");
        } finally {
            setIsCalculating(false);
        }
    };

    // --- 新增：导出 Excel ---
    const handleExportReport = async () => {
        if (!calculationResults) return;

        try {
            console.log("📥 Exporting Report...");
            const res = await fetch(`${API_BASE_URL}/export/report`);
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = "Area_Calculation_Report.xlsx";
                document.body.appendChild(a);
                a.click();
                a.remove();
            } else {
                alert("Export failed");
            }
        } catch (err) {
            console.error("Export error:", err);
        }
    };

    // --- 新增：处理 Approve ---
    const handleApprove = async () => {
        if (!agent2Data || !agent2Data.element_id) return;
        try {
            console.log("✅ Approving element:", agent2Data.element_id);
            const res = await fetch("/analyze/approve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ element_guid: agent2Data.element_id })
            });
            if (res.ok) {
                const json = await res.json();
                console.log("Approved:", json);
                // Update local state
                setAgent2Data(prev => ({ ...prev, is_dirty: false }));
                // Update review queue
                setReviewQueue(prev => prev.filter(item => item.guid !== agent2Data.element_id));

                // Update semanticResults summary count
                if (semanticResults) {
                    setSemanticResults(prev => ({
                        ...prev,
                        needs_review: Math.max(0, prev.needs_review - 1)
                    }));
                }
            } else {
                alert("Approve failed");
            }
        } catch (err) {
            console.error("Approve failed:", err);
        }
    };

    // --- 新增：处理 Edit ---
    const handleEdit = async () => {
        if (!agent2Data || !agent2Data.element_id) return;

        // Simple prompt for now
        const newType = prompt("Enter new semantic type (e.g., EXTERIOR_WALL, BALCONY):", agent2Data.type);
        if (newType && newType !== agent2Data.type) {
            try {
                console.log("✏️ Editing element:", agent2Data.element_id, "to", newType);
                const res = await fetch("/analyze/update", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        element_guid: agent2Data.element_id,
                        new_type: newType,
                        reason: "User manual edit"
                    })
                });
                if (res.ok) {
                    const json = await res.json();
                    console.log("Updated:", json);

                    // Update local state
                    setAgent2Data(prev => ({ ...prev, type: newType, is_dirty: false }));

                    // Update review queue
                    setReviewQueue(prev => prev.filter(item => item.guid !== agent2Data.element_id));

                    // Update semanticResults summary count
                    if (semanticResults) {
                        setSemanticResults(prev => ({
                            ...prev,
                            needs_review: Math.max(0, prev.needs_review - 1)
                        }));
                    }
                } else {
                    alert("Update failed");
                }
            } catch (err) {
                console.error("Update failed:", err);
            }
        }
    };

    return (
        <div className="flex flex-col h-screen bg-[#0b0c0e] text-gray-300 font-sans text-sm overflow-hidden">

            {/* ==================== A. 顶部导航栏 ==================== */}
            <header className="h-16 border-b border-gray-800 bg-[#111316] flex items-center px-4 justify-between shrink-0">

                <div className="flex items-center space-x-6">
                    {/* ChatBIM Branding */}
                    <div className="flex items-center mr-2">
                        <span className="text-xl font-bold text-white tracking-tight">
                            ChatBIM
                        </span>
                    </div>

                    <div className="h-8 w-px bg-gray-700 mx-2"></div>

                    <div className="flex flex-col">
                        <span className="text-xs text-gray-500 uppercase tracking-wider">Project</span>
                        <span className="text-white font-bold text-lg tracking-tight">{projectName}</span>
                    </div>

                    <label className="flex items-center px-3 py-1.5 bg-[#2d333b] hover:bg-[#363c45] text-white rounded border border-gray-600 cursor-pointer transition-colors text-xs font-medium">
                        <Upload size={14} className="mr-2 text-emerald-400" />
                        {ifcFile ? "File Loaded" : "Import IFC"}
                        <input type="file" accept=".ifc" className="hidden" onChange={handleFileUpload} />
                    </label>

                    <div className="h-8 w-px bg-gray-700"></div>

                    {/* 修改：法规选择变成按钮 */}
                    <div className="flex flex-col">
                        <span className="text-xs text-gray-500">Regulation Rules</span>
                        <button
                            onClick={() => setShowRuleModal(true)}
                            className="flex items-center text-emerald-400 cursor-pointer hover:text-emerald-300 focus:outline-none"
                        >
                            <BookOpen size={14} className="mr-1.5" />
                            <span className="truncate max-w-[200px]">{ruleFileName}</span>
                            <ChevronDown size={14} className="ml-1" />
                        </button>
                    </div>

                    <div className="h-8 w-px bg-gray-700"></div>

                    {/* Start Analysis Button (Gradient & Loading) */}
                    <button
                        onClick={handleSemanticAnalysis}
                        disabled={!ifcFile || !activeRules}
                        className={`flex items-center px-4 py-2 rounded-md shadow-lg transition-all transform hover:scale-105 active:scale-95 font-semibold text-xs tracking-wide
                            ${!ifcFile || !activeRules
                                ? 'bg-[#2d333b] text-gray-500 cursor-not-allowed opacity-50 border border-gray-600'
                                : isAnalyzingSemantic
                                    ? 'bg-red-600 hover:bg-red-500 text-white shadow-red-900/50' // 正在分析时显示红色 Stop 样式
                                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-900/50'
                            }`}
                    >
                        {isAnalyzingSemantic ? (
                            <Square size={16} className="mr-2 fill-current" />
                        ) : (
                            <BrainCircuit size={16} className="mr-2" />
                        )}
                        {isAnalyzingSemantic ? "STOP ANALYSIS" : "START ANALYSIS"}
                    </button>

                    {/* Agent 3: Calculate Area Button */}
                    {semanticResults && (
                        <button
                            onClick={handleCalculateArea}
                            disabled={isCalculating}
                            className={`flex items-center px-4 py-2 rounded-md shadow-lg transition-all transform hover:scale-105 active:scale-95 font-semibold text-xs tracking-wide ml-4
                                ${isCalculating
                                    ? 'bg-blue-800 text-white cursor-wait'
                                    : 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/50'
                                }`}
                        >
                            {isCalculating ? <Loader2 size={16} className="mr-2 animate-spin" /> : <Calculator size={16} className="mr-2" />}
                            {isCalculating ? "CALCULATING..." : "CALCULATE AREA"}
                        </button>
                    )}
                </div>

                {/* Dashboard Stats */}
                <div className="flex items-center space-x-6">
                    {semanticResults && (
                        <div className="flex items-center space-x-4 mr-4">
                            <div className="flex flex-col items-end">
                                <span className="text-[10px] text-gray-500 uppercase font-bold">Processed</span>
                                <span className="text-lg font-mono text-emerald-400 leading-none">{semanticResults.total}</span>
                            </div>
                            <div className="h-6 w-px bg-gray-700"></div>
                            <div className="flex flex-col items-end">
                                <span className="text-[10px] text-gray-500 uppercase font-bold">Needs Review</span>
                                <span className={`text-lg font-mono leading-none ${semanticResults.needs_review > 0 ? 'text-orange-500' : 'text-gray-400'}`}>
                                    {semanticResults.needs_review}
                                </span>
                            </div>
                        </div>
                    )}

                    <div className="flex items-center bg-[#1a1d21] rounded-lg border border-gray-700 px-4 py-2 space-x-4 shadow-inner">
                        <div className="flex flex-col items-center">
                            <span className="text-[10px] text-gray-500">Total GFA</span>
                            <span className="text-sm font-bold text-white">
                                {calculationResults ? `${calculationResults.total_area} m²` : "---"}
                            </span>
                        </div>
                    </div>

                    <button
                        onClick={handleExportReport}
                        disabled={!calculationResults}
                        title="Export Calculation Report"
                        className={`p-2 rounded border transition-colors
                            ${calculationResults
                                ? 'bg-[#1a1d21] hover:bg-[#25282e] text-emerald-400 border-emerald-900/50 hover:border-emerald-500'
                                : 'bg-[#1a1d21] text-gray-600 border-gray-800 cursor-not-allowed'
                            }`}
                    >
                        <FileSpreadsheet size={16} />
                    </button>
                </div>
            </header>

            {/* ==================== 主体内容区域 ==================== */}
            <div className="flex flex-1 overflow-hidden relative">

                {/* B. 左侧面板 (Tabbed) */}
                <aside className="w-80 bg-[#111316] border-r border-gray-800 flex flex-col">
                    {/* Tabs */}
                    <div className="flex border-b border-gray-800">
                        <button
                            onClick={() => setActiveTab('structure')}
                            className={`flex-1 py-3 text-xs font-bold uppercase tracking-wide flex items-center justify-center transition-colors
                                ${activeTab === 'structure' ? 'bg-[#1a1d21] text-emerald-500 border-b-2 border-emerald-500' : 'text-gray-500 hover:text-gray-300 hover:bg-[#16181d]'}`}
                        >
                            <Layers size={14} className="mr-2" />
                            Structure
                        </button>
                        <button
                            onClick={() => setActiveTab('review')}
                            className={`flex-1 py-3 text-xs font-bold uppercase tracking-wide flex items-center justify-center transition-colors relative
                                ${activeTab === 'review' ? 'bg-[#1a1d21] text-orange-500 border-b-2 border-orange-500' : 'text-gray-500 hover:text-gray-300 hover:bg-[#16181d]'}`}
                        >
                            <ListChecks size={14} className="mr-2" />
                            Review
                            {reviewQueue.length > 0 && (
                                <span className="absolute top-2 right-4 w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
                            )}
                        </button>
                        <button
                            onClick={() => setActiveTab('calculation')}
                            className={`flex-1 py-3 text-xs font-bold uppercase tracking-wide flex items-center justify-center transition-colors
                                ${activeTab === 'calculation' ? 'bg-[#1a1d21] text-blue-500 border-b-2 border-blue-500' : 'text-gray-500 hover:text-gray-300 hover:bg-[#16181d]'}`}
                        >
                            <Calculator size={14} className="mr-2" />
                            Results
                        </button>
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 overflow-y-auto bg-[#0b0c0e]">
                        {activeTab === 'structure' && (
                            <>
                                {/* 搜索栏保持不变 */}
                                <div className="p-3">
                                    {/* ... (原有搜索代码) ... */}
                                    <div className="relative">
                                        <Search size={14} className="absolute left-3 top-2.5 text-gray-500" />
                                        <input type="text" placeholder="Search structure..." className="w-full bg-[#16181d] border border-gray-700 rounded py-2 pl-9 pr-3 text-xs text-white focus:border-emerald-500 focus:outline-none transition-colors" />
                                        <Filter size={14} className="absolute right-3 top-2.5 text-gray-500" />
                                    </div>
                                </div>
                                <div className="px-2 py-2">
                                    {!treeData && (
                                        <div className="text-center mt-10 text-gray-600 text-xs flex flex-col items-center">
                                            <Loader2 size={24} className="mb-2 animate-spin opacity-20" />
                                            {ifcFile ? "Parsing Structure..." : "Please Import IFC File"}
                                        </div>
                                    )}
                                    {treeData && <TreeNode node={treeData} />}
                                </div>
                            </>
                        )}

                        {activeTab === 'review' && (
                            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                                {reviewQueue.length === 0 ? (
                                    <div className="text-gray-500 text-center mt-10 text-xs">No items to review.</div>
                                ) : (
                                    reviewQueue.map((item, idx) => (
                                        <div
                                            key={idx}
                                            onClick={() => handleReviewItemClick(item)}
                                            className="bg-[#1a1d21] p-3 rounded border border-gray-700 hover:border-orange-500 cursor-pointer transition-colors group"
                                        >
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="font-bold text-white text-xs">{item.name}</span>
                                                <AlertTriangle size={12} className="text-orange-500" />
                                            </div>
                                            <div className="text-[10px] text-gray-500 space-y-1">
                                                <div className="flex justify-between">
                                                    <span>Type:</span>
                                                    <span className="text-gray-300">{item.type}</span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span>Reason:</span>
                                                    <span className="text-orange-400">{item.reason}</span>
                                                </div>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        )}

                        {activeTab === 'calculation' && (
                            <div className="p-4 space-y-4">
                                {!calculationResults ? (
                                    <div className="text-center mt-10 text-gray-600 text-xs">
                                        No calculation results yet. <br /> Click "Calculate Area" to start.
                                    </div>
                                ) : (
                                    <>
                                        {/* Summary */}
                                        <div className="bg-[#1a1d21] p-3 rounded border border-gray-700 shadow-lg">
                                            <div className="flex items-center mb-2">
                                                <Activity size={14} className="text-emerald-400 mr-2" />
                                                <h3 className="text-white font-bold text-xs uppercase tracking-wider">Project Summary</h3>
                                            </div>
                                            <div className="space-y-2">
                                                <div className="flex justify-between items-end border-b border-gray-800 pb-2">
                                                    <span className="text-gray-500 text-xs">Total GFA</span>
                                                    <span className="text-xl font-mono text-emerald-400 font-bold">{calculationResults.total_area} m²</span>
                                                </div>
                                                <div className="text-[10px] text-gray-500 pt-1">
                                                    {calculationResults.summary_text}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Story List */}
                                        <div className="space-y-2">
                                            <div className="text-xs text-gray-500 uppercase font-bold tracking-wider mb-2 pl-1">Story Breakdown</div>
                                            {calculationResults.stories.map((story, idx) => (
                                                <div key={idx} className="bg-[#1a1d21] p-3 rounded border border-gray-700 hover:border-blue-500 transition-colors">
                                                    <div className="flex justify-between items-center mb-2">
                                                        <span className="font-bold text-white text-sm">{story.story_name}</span>
                                                        <span className="font-mono text-blue-400 font-bold">{story.calculated_area} m²</span>
                                                    </div>

                                                    <div className="grid grid-cols-2 gap-y-1 gap-x-2 text-[10px] text-gray-400 bg-[#111316] p-2 rounded">
                                                        <div className="flex justify-between">
                                                            <span>Base Area:</span>
                                                            <span className="text-gray-300">{story.base_area}</span>
                                                        </div>
                                                        <div className="flex justify-between">
                                                            <span>Height Coeff:</span>
                                                            <span className={story.height_coefficient < 1 ? "text-orange-400 font-bold" : "text-gray-300"}>
                                                                {story.height_coefficient}
                                                            </span>
                                                        </div>
                                                        <div className="flex justify-between col-span-2 border-t border-gray-800 pt-1 mt-1">
                                                            <span>Adjustments:</span>
                                                            <span className={story.adjustments_area !== 0 ? "text-yellow-400" : "text-gray-500"}>
                                                                {story.adjustments_area > 0 ? "+" : ""}{story.adjustments_area}
                                                            </span>
                                                        </div>
                                                    </div>

                                                    {story.adjustment_details.length > 0 && (
                                                        <div className="mt-2 text-[10px] text-gray-500 pl-2 border-l-2 border-gray-800">
                                                            {story.adjustment_details.map((d, i) => (
                                                                <div key={i}>• {d}</div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </div>
                        )}
                    </div>


                </aside>

                {/* C. 中间视口 (保持不变) */}
                <main ref={viewportRef} className="flex-1 relative bg-[#0f1115] overflow-hidden">
                    <div className="absolute inset-0 z-0">
                        <IfcViewer
                            file={ifcFile}
                            onLoaded={handleModelLoaded}
                            onSelect={handleSelection}
                            width={viewportSize.w}
                            height={viewportSize.h}
                            selectedId={selectedId}
                        />
                    </div>
                    {!ifcFile && (
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                            <div className="bg-black/50 backdrop-blur px-6 py-4 rounded border border-gray-700 text-white flex flex-col items-center">
                                <Upload size={32} className="mb-2 text-emerald-500" />
                                <span>Drag & Drop or Click Import to Start</span>
                            </div>
                        </div>
                    )}
                    {/* ... 3D View Toggle, Context Toolbar, Legend 保持不变 ... */}
                </main>

                {/* ==================== D. 右侧面板：属性与计算逻辑 ==================== */}
                <aside className="w-80 bg-[#111316] border-l border-gray-800 overflow-y-auto">
                    <div className="p-3 border-b border-gray-800 flex justify-between items-center text-emerald-500 font-medium">
                        <span>Properties & Logic</span>
                        <Info size={16} />
                    </div>

                    {!selectedProps ? (
                        <div className="p-8 text-center text-gray-600 text-xs italic">
                            Select an element in the 3D view to see details.
                        </div>
                    ) : (
                        <div className="p-4 space-y-6">
                            {/* 选中对象头信息 */}
                            <div className="bg-[#1a1d21] p-3 rounded border border-gray-700">
                                <span className="text-xs text-gray-500 block mb-1">Entity Name</span>
                                <h2 className="text-white font-bold text-md break-all flex items-center">
                                    <span className="w-2 h-2 rounded-full bg-yellow-500 mr-2"></span>
                                    {selectedProps.Name ? selectedProps.Name.value : 'Unnamed'}
                                </h2>
                                <span className="text-[10px] text-gray-500 mt-1 block font-mono">{selectedProps.GlobalId ? selectedProps.GlobalId.value : ''}</span>
                            </div>



                            {/* 新增：如果语义分析已完成，显示 Agent 2 结果 (已移至 Dashboard 和 Review Tab，这里保留基本信息) */}
                            {semanticResults && !agent2Data && (
                                <div className="mt-6">
                                    <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Semantic Overview</h3>
                                    <div className="bg-[#16181d] border border-gray-700 p-3 rounded text-xs text-gray-400">
                                        Select an element to view detailed analysis.
                                    </div>
                                </div>
                            )}

                            {/* 新增：Agent 2 Analysis Card (Dynamic) */}
                            {agent2Data && (
                                <div className={`mt-6 rounded border p-4 shadow-lg ${agent2Data.is_dirty ? 'bg-orange-900/10 border-orange-500/50' : 'bg-blue-900/10 border-blue-500/50'}`}>
                                    <div className="flex justify-between items-center mb-3">
                                        <h3 className={`text-xs font-bold uppercase flex items-center ${agent2Data.is_dirty ? 'text-orange-400' : 'text-blue-400'}`}>
                                            {agent2Data.is_dirty ? <AlertTriangle size={14} className="mr-2" /> : <BrainCircuit size={14} className="mr-2" />}
                                            Agent 2 Analysis
                                        </h3>
                                        <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold ${agent2Data.is_dirty ? 'bg-orange-500 text-white' : 'bg-blue-500 text-white'}`}>
                                            Factor: {agent2Data.calc_factor}
                                        </span>
                                    </div>

                                    <div className="space-y-3">
                                        <div>
                                            <span className="text-[10px] text-gray-500 uppercase font-bold block mb-1">Inferred Type</span>
                                            <div className="text-sm font-medium text-white flex items-center">
                                                {agent2Data.type}
                                                {agent2Data.is_dirty && (
                                                    <span className="ml-2 text-[10px] text-red-400 border border-red-500/30 px-1 rounded">Low Confidence</span>
                                                )}
                                            </div>
                                        </div>

                                        <div>
                                            <span className="text-[10px] text-gray-500 uppercase font-bold block mb-1">Reasoning Chain</span>
                                            <div className="text-xs text-gray-300 leading-relaxed bg-black/20 p-2 rounded border border-white/5">
                                                {agent2Data.reason}
                                            </div>
                                        </div>

                                        {agent2Data.is_dirty && (
                                            <div className="flex space-x-2 pt-2">
                                                <button
                                                    onClick={handleApprove}
                                                    className="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white text-xs py-1.5 rounded transition-colors"
                                                >
                                                    Approve
                                                </button>
                                                <button
                                                    onClick={handleEdit}
                                                    className="flex-1 bg-[#2d333b] hover:bg-[#363c45] text-gray-300 text-xs py-1.5 rounded border border-gray-600 transition-colors"
                                                >
                                                    Edit
                                                </button>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* 原有几何信息保持不变 */}
                            <div>
                                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Geometry</h3>
                                {/* ... Geometry Details ... */}
                                <div className="grid grid-cols-2 gap-y-2 text-sm">
                                    <div className="text-gray-400">Type</div>
                                    <div className="text-right text-gray-200">
                                        {selectedProps.constructor && selectedProps.constructor.name.replace('Ifc', '')}
                                    </div>

                                </div>
                            </div>

                            {/* 动态属性集展示 */}
                            {selectedPsets && selectedPsets.length > 0 && (
                                <div className="space-y-4">
                                    {selectedPsets.map((pset, index) => (
                                        <div key={index}>
                                            <h3 className="text-xs font-bold text-emerald-500 uppercase mb-2 border-b border-gray-800 pb-1">
                                                {pset.Name ? pset.Name.value : 'Property Set'}
                                            </h3>
                                            <div className="grid grid-cols-2 gap-y-1 text-xs">
                                                {pset.HasProperties && pset.HasProperties.map((prop, pIdx) => (
                                                    <React.Fragment key={pIdx}>
                                                        <div className="text-gray-400 truncate pr-2" title={prop.Name ? prop.Name.value : ''}>
                                                            {prop.Name ? prop.Name.value : 'Unknown'}
                                                        </div>
                                                        <div className="text-right text-gray-200 truncate">
                                                            {prop.NominalValue ? String(prop.NominalValue.value) : '-'}
                                                        </div>
                                                    </React.Fragment>
                                                ))}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </aside>

                {/* ==================== E. 新增：法规管理弹窗 (Agent 1 UI) ==================== */}
                {showRuleModal && (
                    <div className="absolute inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
                        <div className="bg-[#1a1d21] w-[600px] max-h-[80vh] rounded-lg border border-gray-700 shadow-2xl flex flex-col">

                            {/* Header */}
                            <div className="p-4 border-b border-gray-700 flex justify-between items-center">
                                <h3 className="text-white font-bold flex items-center">
                                    <BookOpen size={18} className="mr-2 text-emerald-500" />
                                    Regulation Manager (Agent 1)
                                </h3>
                                <button onClick={() => setShowRuleModal(false)} className="text-gray-500 hover:text-white"><X size={18} /></button>
                            </div>

                            {/* Content */}
                            <div className="p-6 flex-1 overflow-y-auto">

                                {/* 1. Upload Section */}
                                <div className="mb-6">
                                    <label className="block text-xs font-bold text-gray-500 uppercase mb-2">Upload Regulation PDF</label>
                                    <div className="border-2 border-dashed border-gray-700 hover:border-emerald-500/50 rounded-lg p-8 text-center transition-colors relative">
                                        {isAnalyzingRule ? (
                                            <div className="flex flex-col items-center animate-pulse">
                                                <Loader2 size={32} className="text-emerald-500 animate-spin mb-2" />
                                                <span className="text-emerald-400 font-medium">Agent 1 is reasoning...</span>
                                                <span className="text-xs text-gray-500 mt-1">Extracting area rules</span>
                                            </div>
                                        ) : (
                                            <>
                                                <Upload size={32} className="mx-auto text-gray-500 mb-2" />
                                                <p className="text-gray-300 text-sm">Drag & drop PDF here</p>
                                                <input type="file" accept=".pdf" className="absolute inset-0 opacity-0 cursor-pointer" onChange={handleRegulationUpload} />
                                            </>
                                        )}
                                    </div>
                                </div>

                                {/* 2. Analysis Results (Agent Output) */}
                                {activeRules && (
                                    <div className="space-y-4 animate-fade-in-up">
                                        <div className="flex items-center justify-between">
                                            <h4 className="text-emerald-400 font-bold text-sm">Extraction Results</h4>
                                            <span className="text-[10px] bg-emerald-900/50 text-emerald-400 px-2 py-0.5 rounded border border-emerald-900">Verified by LLM</span>
                                        </div>

                                        {/* Height Rules */}
                                        <div className="bg-[#111316] p-3 rounded border border-gray-700">
                                            <span className="text-xs text-gray-500 block mb-2 font-bold">1. Height Requirements</span>
                                            <div className="space-y-2">
                                                {activeRules.height_requirements?.map((rule, idx) => (
                                                    <div key={idx} className="flex justify-between items-center text-xs bg-[#1a1d21] p-2 rounded">
                                                        <span className="text-gray-300">{rule.description}</span>
                                                        <span className={`font-bold px-1.5 rounded ${rule.coefficient === 1 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                                                            k={rule.coefficient}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Enclosure Rules */}
                                        <div className="bg-[#111316] p-3 rounded border border-gray-700">
                                            <span className="text-xs text-gray-500 block mb-2 font-bold">2. Enclosure Requirements</span>
                                            <div className="space-y-2">
                                                {activeRules.enclosure_requirements?.map((rule, idx) => (
                                                    <div key={idx} className="flex justify-between items-center text-xs bg-[#1a1d21] p-2 rounded">
                                                        <span className="text-gray-300">{rule.description}</span>
                                                        <span className={`font-bold px-1.5 rounded ${rule.coefficient === 1 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                                                            k={rule.coefficient}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Special Space Rules */}
                                        <div className="bg-[#111316] p-3 rounded border border-gray-700">
                                            <span className="text-xs text-gray-500 block mb-2 font-bold">3. Special Space Requirements</span>
                                            <div className="space-y-2">
                                                {activeRules.special_space_requirements?.map((rule, idx) => (
                                                    <div key={idx} className="flex justify-between items-center text-xs bg-[#1a1d21] p-2 rounded">
                                                        <span className="text-gray-300">{rule.description}</span>
                                                        <span className={`font-bold px-1.5 rounded ${rule.coefficient === 1 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'}`}>
                                                            k={rule.coefficient}
                                                        </span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Reasoning Trace */}
                                        <div className="bg-[#111316] p-3 rounded border border-gray-700">
                                            <span className="text-xs text-gray-500 block mb-1 font-bold">🧠 Reasoning Trace</span>
                                            <p className="text-[10px] text-gray-400 leading-relaxed font-mono">
                                                {activeRules.reasoning_trace}
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
};

export default BIMInterface;