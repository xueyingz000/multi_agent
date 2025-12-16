import { useState, useRef, useEffect } from 'react';
import {
    Box, Layers, ChevronRight, ChevronDown, Search, Filter,
    Download, Upload, Maximize2, MousePointer2,
    Edit3, Scissors, RotateCcw, FileText, Info, CheckSquare, Square
} from 'lucide-react';
import IfcViewer from './IfcViewer';

// --- 组件：结构树节点 (保持动态递归) ---
const TreeNode = ({ node, onSelectNode, depth = 0 }) => {
    const [expanded, setExpanded] = useState(depth < 2);
    const hasChildren = node.children && node.children.length > 0;

    const getIcon = (type) => {
        if (type === 'IFCBUILDING') return <Box size={14} className="text-blue-400" />;
        if (type === 'IFCBUILDINGSTOREY') return <Layers size={14} className="text-yellow-400" />;
        if (type === 'IFCSPACE') return <Square size={14} className="text-emerald-400 transform rotate-45" />;
        return <Box size={14} className="text-gray-500" />;
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
                <span className="text-xs truncate">{node.Name ? node.Name.value : 'Untitled'}</span>
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
    const [treeData, setTreeData] = useState(null);
    const [selectedProps, setSelectedProps] = useState(null); // 选中的构件属性

    // 视口尺寸
    const viewportRef = useRef(null);
    const [viewportSize, setViewportSize] = useState({ w: 800, h: 600 });

    useEffect(() => {
        const handleResize = () => {
            if (viewportRef.current) {
                setViewportSize({
                    w: viewportRef.current.offsetWidth,
                    h: viewportRef.current.offsetHeight
                });
            }
        };
        window.addEventListener('resize', handleResize);
        handleResize(); // Init
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // --- 事件处理 ---
    const handleFileUpload = (e) => {
        if (e.target.files && e.target.files[0]) {
            setIfcFile(e.target.files[0]);
        }
    };

    const handleModelLoaded = (_, structure) => {
        if (structure && structure.children && structure.children.length > 0) {
            setTreeData(structure.children[0]);
        }
    };

    const handleSelection = (_, props) => {
        setSelectedProps(props || null);
    };

    return (
        <div className="flex flex-col h-screen bg-[#0b0c0e] text-gray-300 font-sans text-sm overflow-hidden">

            {/* ==================== A. 顶部导航栏 (还原了仪表盘和进度条) ==================== */}
            <header className="h-16 border-b border-gray-800 bg-[#111316] flex items-center px-4 justify-between shrink-0">

                {/* 左侧：项目信息 + 上传按钮 */}
                <div className="flex items-center space-x-6">
                    <div className="flex flex-col">
                        <span className="text-xs text-gray-500 uppercase tracking-wider">Project</span>
                        <span className="text-white font-bold text-lg tracking-tight">Shanghai Tower_BIM_v3</span>
                    </div>

                    {/* 这里是上传按钮 */}
                    <label className="flex items-center px-3 py-1.5 bg-[#2d333b] hover:bg-[#363c45] text-white rounded border border-gray-600 cursor-pointer transition-colors text-xs font-medium">
                        <Upload size={14} className="mr-2 text-emerald-400" />
                        {ifcFile ? "File Loaded" : "Import IFC"}
                        <input type="file" accept=".ifc" className="hidden" onChange={handleFileUpload} />
                    </label>

                    <div className="h-8 w-px bg-gray-700"></div>
                    <div className="flex flex-col">
                        <span className="text-xs text-gray-500">Regulation</span>
                        <div className="flex items-center text-emerald-400 cursor-pointer hover:text-emerald-300">
                            <span>Shanghai Standard (2024)</span>
                            <ChevronDown size={14} className="ml-1" />
                        </div>
                    </div>
                </div>

                {/* 中间：核心仪表盘 (The Dashboard) */}
                <div className="flex items-center bg-[#1a1d21] rounded-lg border border-gray-700 px-6 py-2 space-x-8 shadow-inner">
                    <div className="flex flex-col items-center">
                        <span className="text-xs text-gray-500">Total GFA</span>
                        <span className="text-xl font-bold text-white">125,000 m²</span>
                    </div>
                    <div className="flex flex-col w-48">
                        <div className="flex justify-between text-xs mb-1">
                            <span className="text-gray-400">FAR (Current/Max)</span>
                            <span className="text-emerald-400 font-mono">3.5 / 4.0</span>
                        </div>
                        {/* 进度条 */}
                        <div className="w-full bg-gray-700 rounded-full h-2">
                            <div className="bg-gradient-to-r from-emerald-600 to-emerald-400 h-2 rounded-full" style={{ width: '87.5%' }}></div>
                        </div>
                    </div>
                </div>

                {/* 右侧：操作按钮 */}
                <div className="flex items-center space-x-3">
                    <button className="flex items-center px-4 py-2 bg-[#1a1d21] hover:bg-[#25282e] text-gray-300 rounded border border-gray-700 transition-colors">
                        <Download size={16} className="mr-2" />
                        Report
                    </button>
                </div>
            </header>

            {/* ==================== 主体内容区域 ==================== */}
            <div className="flex flex-1 overflow-hidden">

                {/* ==================== B. 左侧面板：结构树与过滤器 ==================== */}
                <aside className="w-80 bg-[#111316] border-r border-gray-800 flex flex-col">
                    <div className="p-3 border-b border-gray-800 flex justify-between items-center text-emerald-500 font-medium">
                        <span>Structure & Filters</span>
                        <Layers size={16} />
                    </div>

                    {/* 搜索 */}
                    <div className="p-3">
                        <div className="relative">
                            <Search size={14} className="absolute left-3 top-2.5 text-gray-500" />
                            <input
                                type="text"
                                placeholder="Search zones..."
                                className="w-full bg-[#0b0c0e] border border-gray-700 rounded py-2 pl-9 pr-3 text-xs focus:border-emerald-500 focus:outline-none text-white"
                            />
                            <Filter size={14} className="absolute right-3 top-2.5 text-gray-500 cursor-pointer hover:text-white" />
                        </div>
                    </div>

                    {/* 动态结构树区域 */}
                    <div className="flex-1 overflow-y-auto px-2 py-2">
                        {!treeData && (
                            <div className="text-center mt-10 text-gray-600 text-xs">
                                {ifcFile ? "Parsing Structure..." : "Please Import IFC File"}
                            </div>
                        )}
                        {treeData && <TreeNode node={treeData} />}
                    </div>

                    {/* 底部过滤器 (还原复选框) */}
                    <div className="p-4 bg-[#16181d] border-t border-gray-800">
                        <h4 className="text-xs text-gray-500 uppercase font-bold mb-3">Filter Visibility</h4>
                        <div className="space-y-2">
                            <label className="flex items-center space-x-2 cursor-pointer group">
                                <CheckSquare size={16} className="text-emerald-500" />
                                <span className="text-gray-300 group-hover:text-white">GFA Areas (Factor &gt; 0)</span>
                            </label>
                            <label className="flex items-center space-x-2 cursor-pointer group">
                                <Square size={16} className="text-gray-600" />
                                <span className="text-gray-400 group-hover:text-white">Half-GFA Only (0.5)</span>
                            </label>
                        </div>
                    </div>
                </aside>

                {/* ==================== C. 中间视口：3D 可视化交互区 ==================== */}
                <main ref={viewportRef} className="flex-1 relative bg-[#0f1115] overflow-hidden">

                    {/* 1. 真实的 3D 引擎 (作为背景层) */}
                    <div className="absolute inset-0 z-0">
                        <IfcViewer
                            file={ifcFile}
                            onLoaded={handleModelLoaded}
                            onSelect={handleSelection}
                            width={viewportSize.w}
                            height={viewportSize.h}
                        />
                    </div>

                    {/* 2. 浮动提示：如果没有文件 */}
                    {!ifcFile && (
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                            <div className="bg-black/50 backdrop-blur px-6 py-4 rounded border border-gray-700 text-white flex flex-col items-center">
                                <Upload size={32} className="mb-2 text-emerald-500" />
                                <span>Drag & Drop or Click Import to Start</span>
                            </div>
                        </div>
                    )}

                    {/* 3. 顶部视图切换控件 (UI Overlay) */}
                    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 flex bg-[#1a1d21] border border-gray-700 rounded p-1 shadow-lg z-10">
                        <button className="px-4 py-1.5 bg-[#2d333b] text-white rounded text-xs font-medium">3D View</button>
                        <button className="px-4 py-1.5 text-gray-400 hover:text-white rounded text-xs font-medium">2D Plan</button>
                        <div className="w-px bg-gray-700 mx-1 h-full"></div>
                        <span className="px-3 py-1.5 text-gray-500 text-xs flex items-center">Level: L2</span>
                    </div>

                    {/* 4. 浮动工具栏 (Context Toolbar) - 仅选中时显示 */}
                    {selectedProps && (
                        <div className="absolute top-[20%] right-[25%] bg-[#1a1d21]/95 backdrop-blur border border-emerald-500/50 rounded-lg shadow-2xl p-2 w-80 animate-fade-in-up z-20">
                            <div className="flex justify-between items-center mb-2 pb-2 border-b border-gray-700">
                                <span className="text-xs font-bold text-white">Boundary Editor</span>
                                <button onClick={() => setSelectedProps(null)} className="text-gray-500 hover:text-white">✕</button>
                            </div>

                            {/* Snap Controls */}
                            <div className="grid grid-cols-2 gap-1 mb-2">
                                <button className="text-xs bg-[#2d333b] hover:bg-emerald-900/50 text-gray-300 hover:text-emerald-400 py-1 px-2 rounded flex items-center justify-center border border-transparent hover:border-emerald-500/30">
                                    <MousePointer2 size={12} className="mr-1" /> Center Line
                                </button>
                                <button className="text-xs bg-emerald-900/30 text-emerald-400 border border-emerald-500/50 py-1 px-2 rounded flex items-center justify-center font-medium">
                                    <Maximize2 size={12} className="mr-1" /> Wall Exterior
                                </button>
                            </div>

                            {/* Action Tools */}
                            <div className="flex space-x-1">
                                <button className="flex-1 bg-[#0b0c0e] hover:bg-gray-800 p-2 rounded border border-gray-700 flex flex-col items-center group">
                                    <Edit3 size={16} className="text-gray-400 group-hover:text-white mb-1" />
                                    <span className="text-[10px] text-gray-500">Draw</span>
                                </button>
                                <button className="flex-1 bg-[#0b0c0e] hover:bg-gray-800 p-2 rounded border border-gray-700 flex flex-col items-center group">
                                    <Scissors size={16} className="text-gray-400 group-hover:text-white mb-1" />
                                    <span className="text-[10px] text-gray-500">Split</span>
                                </button>
                                <button className="flex-1 bg-[#0b0c0e] hover:bg-gray-800 p-2 rounded border border-gray-700 flex flex-col items-center group">
                                    <RotateCcw size={16} className="text-gray-400 group-hover:text-white mb-1" />
                                    <span className="text-[10px] text-gray-500">Reset</span>
                                </button>
                            </div>
                        </div>
                    )}

                    {/* 5. 底部图例 */}
                    <div className="absolute bottom-4 left-4 bg-[#1a1d21]/80 p-2 rounded border border-gray-700 flex space-x-4 text-xs z-10">
                        <div className="flex items-center"><div className="w-3 h-3 bg-red-500/80 mr-2 rounded-sm"></div>1.0 Full</div>
                        <div className="flex items-center"><div className="w-3 h-3 bg-yellow-500/80 mr-2 rounded-sm"></div>0.5 Half</div>
                    </div>
                </main>

                {/* ==================== D. 右侧面板：属性与计算逻辑 ==================== */}
                <aside className="w-80 bg-[#111316] border-l border-gray-800 overflow-y-auto">
                    <div className="p-3 border-b border-gray-800 flex justify-between items-center text-emerald-500 font-medium">
                        <span>Properties & Logic</span>
                        <Info size={16} />
                    </div>

                    {!selectedProps ? (
                        <div className="p-8 text-center text-gray-600 text-xs italic">
                            Select an element in the 3D view to see calculation details.
                        </div>
                    ) : (
                        <div className="p-4 space-y-6">

                            {/* 选中对象头信息 */}
                            <div className="bg-[#1a1d21] p-3 rounded border border-gray-700">
                                <span className="text-xs text-gray-500 block mb-1">Entity Name</span>
                                <h2 className="text-white font-bold text-md break-all flex items-center">
                                    <span className="w-2 h-2 rounded-full bg-yellow-500 mr-2"></span>
                                    {selectedProps.Name ? selectedProps.Name.value : 'Unnamed Element'}
                                </h2>
                                <span className="text-[10px] text-gray-500 mt-1 block font-mono">{selectedProps.GlobalId ? selectedProps.GlobalId.value : ''}</span>
                            </div>

                            {/* 1. 几何信息 (混合真实数据与 UI 布局) */}
                            <div>
                                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Geometry</h3>
                                <div className="grid grid-cols-2 gap-y-2 text-sm">
                                    <div className="text-gray-400">Type</div>
                                    <div className="text-right text-gray-200">
                                        {selectedProps.constructor && selectedProps.constructor.name.replace('Ifc', '')}
                                    </div>
                                    {/* 以下数据暂为模拟，需后端计算返回 */}
                                    <div className="text-gray-400">Clear Height</div>
                                    <div className="text-right text-gray-200 font-mono">2.8m</div>
                                    <div className="text-gray-400">Depth</div>
                                    <div className="text-right text-gray-200 font-mono">1.5m</div>
                                </div>
                            </div>

                            <hr className="border-gray-800" />

                            {/* 2. 计算结果 (高亮核心) */}
                            <div>
                                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Calculation</h3>
                                <div className="bg-[#1f2329] p-3 rounded border border-gray-700 space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-gray-400">Physical Area</span>
                                        <span className="text-white font-mono">6.0 m²</span>
                                    </div>

                                    <div className="flex justify-between items-center bg-yellow-500/10 p-1 -mx-1 rounded border border-yellow-500/30">
                                        <span className="text-yellow-500 font-medium">Factor</span>
                                        <span className="text-yellow-400 font-bold font-mono">0.5</span>
                                    </div>

                                    <div className="flex justify-between border-t border-gray-700 pt-2 mt-2">
                                        <span className="text-emerald-400 font-bold">GFA (Calculated)</span>
                                        <span className="text-emerald-400 font-bold text-lg font-mono">3.0 m²</span>
                                    </div>
                                </div>
                            </div>

                            {/* 3. 判定逻辑 */}
                            <div>
                                <h3 className="text-xs font-bold text-gray-500 uppercase mb-2">Logic Trace</h3>
                                <div className="bg-[#0b0c0e] p-3 rounded text-xs text-gray-300 border border-gray-800 leading-relaxed">
                                    <p className="mb-2">
                                        <span className="text-gray-500">Condition 1:</span> Enclosed? <span className="text-red-400 font-mono">NO</span>
                                    </p>
                                    <p className="border-t border-gray-800 pt-2 mt-2 text-gray-400 italic">
                                        "Measured from structural floor plate projection. Since open-style, apply factor 0.5."
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* 底部法规引用卡片 */}
                    <div className="mt-auto p-4 bg-[#1f2329] border-t border-emerald-900/30">
                        <div className="flex items-start text-emerald-500 mb-2">
                            <FileText size={14} className="mt-0.5 mr-2 shrink-0" />
                            <span className="text-xs font-bold">Standard Reference</span>
                        </div>
                        <p className="text-[10px] text-gray-400 leading-relaxed">
                            根据《上海市房产测绘规范(2024版)》第3.2.3条：未封闭的阳台、挑廊，按其围护结构外围水平投影面积的一半计算。
                        </p>
                    </div>
                </aside>

            </div>
        </div>
    );
};

export default BIMInterface;