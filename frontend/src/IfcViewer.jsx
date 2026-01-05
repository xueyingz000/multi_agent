import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { IFCLoader } from 'web-ifc-three';
import { IFCRELCONTAINEDINSPATIALSTRUCTURE } from 'web-ifc';

const IfcViewer = ({ file, onLoaded, onSelect, width, height, selectedId }) => {
    const mountRef = useRef(null);
    const [loading, setLoading] = useState(false);

    // 保持对 Three.js 核心对象的引用
    const sceneRef = useRef(null);
    const cameraRef = useRef(null);
    const rendererRef = useRef(null);
    const ifcLoaderRef = useRef(null);
    const modelRef = useRef(null);
    const highlightMatRef = useRef(null);
    const subsetRef = useRef(null);
    const controlsRef = useRef(null);

    // --- 新增：处理外部传入的 selectedId ---
    useEffect(() => {
        if (!selectedId || !modelRef.current || !ifcLoaderRef.current || !sceneRef.current || !highlightMatRef.current) {
            return;
        }

        // 1. 清除现有高亮
        if (subsetRef.current) {
            sceneRef.current.remove(subsetRef.current);
            if (subsetRef.current.geometry) subsetRef.current.geometry.dispose();
            subsetRef.current = null;
        }
        ifcLoaderRef.current.ifcManager.removeSubset(modelRef.current.modelID, highlightMatRef.current);

        // 2. 创建新高亮
        try {
            const id = parseInt(selectedId); // 确保是数字
            const subset = ifcLoaderRef.current.ifcManager.createSubset({
                modelID: modelRef.current.modelID,
                ids: [id],
                material: highlightMatRef.current,
                scene: sceneRef.current,
                removePrevious: true
            });
            // 复制模型变换到高亮子集
            subset.position.copy(modelRef.current.position);
            subset.rotation.copy(modelRef.current.rotation);
            subset.scale.copy(modelRef.current.scale);

            subsetRef.current = subset;

            // 可选：聚焦到选中构件 (这里暂时不自动聚焦，以免打断用户视角)
        } catch (err) {
            console.error("Error highlighting element:", err);
        }
    }, [selectedId]);

    // 初始化场景 (ComponentDidMount)
    useEffect(() => {
        if (!mountRef.current) return;

        // 1. Scene Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f1115); // 深色背景
        sceneRef.current = scene;

        // 2. Camera
        const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(10, 10, 10);
        cameraRef.current = camera;

        // 3. Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, logarithmicDepthBuffer: true }); // Enable logarithmicDepthBuffer
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        mountRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // 4. Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
        directionalLight.position.set(10, 10, 5);
        scene.add(directionalLight);

        // Grid
        const grid = new THREE.GridHelper(50, 50, 0x444444, 0x222222);
        scene.add(grid);

        // 5. Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controlsRef.current = controls;

        // 6. IFC Loader Setup
        const ifcLoader = new IFCLoader();
        // 指向本地的 wasm 路径
        ifcLoader.ifcManager.setWasmPath('./');
        ifcLoader.ifcManager.useWebWorkers(false); // 强制单线程

        // Removed setupThreeMeshBVH call as three-mesh-bvh is not installed
        // ifcLoader.ifcManager.setupThreeMeshBVH(...);

        ifcLoaderRef.current = ifcLoader;

        // 高亮材质
        const highlightMaterial = new THREE.MeshLambertMaterial({
            transparent: true,
            opacity: 0.6,
            color: 0x10b981, // Emerald Green
            depthTest: false
        });
        highlightMatRef.current = highlightMaterial;

        // 7. Raycaster Setup
        const raycaster = new THREE.Raycaster();
        raycaster.firstHitOnly = true;
        const mouse = new THREE.Vector2();

        const handleDoubleClick = async (event) => {
            if (!modelRef.current || !rendererRef.current || !cameraRef.current) return;

            const rect = rendererRef.current.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, cameraRef.current);
            // 开启递归查找，以防模型是 Group 结构
            const intersects = raycaster.intersectObjects([modelRef.current], true);

            if (intersects.length > 0) {
                const index = intersects[0].faceIndex;
                const geometry = intersects[0].object.geometry;
                const id = ifcLoader.ifcManager.getExpressId(geometry, index);

                // 高亮
                const subset = ifcLoader.ifcManager.createSubset({
                    modelID: modelRef.current.modelID,
                    ids: [id],
                    material: highlightMaterial,
                    scene: scene,
                    removePrevious: true
                });
                // 复制模型变换到高亮子集
                subset.position.copy(modelRef.current.position);
                subset.rotation.copy(modelRef.current.rotation);
                subset.scale.copy(modelRef.current.scale);

                subsetRef.current = subset;

                try {
                    // 获取属性并传给父组件
                    const props = await ifcLoader.ifcManager.getItemProperties(modelRef.current.modelID, id);
                    // 获取属性集 (Property Sets)
                    const psets = await ifcLoader.ifcManager.getPropertySets(modelRef.current.modelID, id, true);

                    onSelect(id, props, psets);
                } catch (err) {
                    console.error("Error fetching element properties:", err);
                }
            } else {
                // 清除高亮
                // 1. Explicitly remove previous subset mesh
                if (subsetRef.current) {
                    scene.remove(subsetRef.current);
                    if (subsetRef.current.geometry) subsetRef.current.geometry.dispose();
                    subsetRef.current = null;
                }

                // 2. Also call manager (without scene parameter to just remove from internal map if needed, 
                // but actually removeSubset with scene is what removes it from scene graph usually. 
                // Since we did it manually above, we can just ensure manager state is clean)
                ifcLoader.ifcManager.removeSubset(modelRef.current.modelID, highlightMaterial);

                onSelect(null, null);
            }
        };

        renderer.domElement.addEventListener('dblclick', handleDoubleClick);

        // Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Cleanup
        return () => {
            if (mountRef.current && renderer.domElement) {
                mountRef.current.removeChild(renderer.domElement);
                renderer.domElement.removeEventListener('dblclick', handleDoubleClick);
            }
            renderer.dispose();
        };
    }, []); // Run once on mount

    // 监听文件变化
    useEffect(() => {
        if (file && ifcLoaderRef.current && sceneRef.current) {
            setLoading(true);
            const url = URL.createObjectURL(file);

            // 清理旧模型
            if (modelRef.current) {
                sceneRef.current.remove(modelRef.current);
                modelRef.current = null;
            }

            const loadModel = async () => {
                // 配置加载选项
                // 使用 correct method name: applyWebIfcConfig (Ifc, not IFC)
                await ifcLoaderRef.current.ifcManager.applyWebIfcConfig({
                    COORDINATE_TO_ORIGIN: true,
                    USE_FAST_BOOLS: false // Disable fast bools to ensure complex geometry (windows/doors) are processed correctly
                });

                ifcLoaderRef.current.load(url, async (ifcModel) => {
                    modelRef.current = ifcModel;
                    sceneRef.current.add(ifcModel);

                    // --- 强制双面材质 ---
                    // 遍历所有材质并设置 side = DoubleSide，防止因法线反转导致的面不可见
                    if (ifcModel.material) {
                        if (Array.isArray(ifcModel.material)) {
                            ifcModel.material.forEach(mat => {
                                mat.side = THREE.DoubleSide;
                                // 确保材质不完全透明
                                if (mat.opacity < 0.1) mat.opacity = 0.3;
                                mat.transparent = mat.opacity < 1;
                            });
                        } else {
                            ifcModel.material.side = THREE.DoubleSide;
                        }
                    }

                    // --- 自动居中模型 ---
                    // 使用 Box3 计算包围盒，兼容 Mesh 和 Group
                    const box = new THREE.Box3().setFromObject(ifcModel);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const radius = size.length() / 2;

                    if (!box.isEmpty()) {
                        // 将模型移至原点，但保持底部在 y=0 (让网格作为地面)
                        ifcModel.position.x = -center.x;
                        ifcModel.position.y = -box.min.y; // 底部对齐
                        ifcModel.position.z = -center.z;

                        // 调整相机位置
                        if (cameraRef.current && controlsRef.current) {
                            const fitOffset = radius * 2.5 || 50;
                            // 稍微抬高视角
                            cameraRef.current.position.set(fitOffset, fitOffset / 2 + size.y / 2, fitOffset);
                            // 看向模型中心
                            cameraRef.current.lookAt(0, size.y / 2, 0);
                            controlsRef.current.target.set(0, size.y / 2, 0);
                            controlsRef.current.update();
                        }
                    }

                    // 获取空间结构树 (Building -> Storey -> Space)
                    try {
                        const structure = await ifcLoaderRef.current.ifcManager.getSpatialStructure(ifcModel.modelID);

                        // --- 调试：检查构件是否存在 ---
                        const allWindows = await ifcLoaderRef.current.ifcManager.getAllItemsOfType(ifcModel.modelID, 2520696781 /* IFCWINDOW */, false); // IFCWINDOW ID might vary, using type name if possible or just log generic
                        // Better to use string types if imported or available, but web-ifc exports integer constants.
                        // Let's rely on getAllItemsOfType being correct.
                        // 2520696781 is IFCWINDOW? No, constants are small integers.
                        // We need to import constants. But for now, let's just log structure enrichment which processes contained elements.

                        console.log("Model ID:", ifcModel.modelID);

                        // --- 新增：获取包含关系 (Storey/Space -> Elements) ---
                        const rels = await ifcLoaderRef.current.ifcManager.getAllItemsOfType(ifcModel.modelID, IFCRELCONTAINEDINSPATIALSTRUCTURE, true);
                        const elementsMap = {};
                        for (const rel of rels) {
                            const parentId = rel.RelatingStructure.value;
                            const childIds = rel.RelatedElements.map(r => r.value);
                            if (!elementsMap[parentId]) elementsMap[parentId] = [];
                            elementsMap[parentId].push(...childIds);
                        }

                        // --- 递归获取节点属性 (Name, GlobalId) 并附加构件 ---
                        const enrichNode = async (node) => {
                            if (!node) return;
                            try {
                                // 只为没有 Name 的节点获取属性
                                if (!node.Name || !node.Name.value) {
                                    const props = await ifcLoaderRef.current.ifcManager.getItemProperties(ifcModel.modelID, node.expressID);
                                    if (props) {
                                        if (props.Name) node.Name = props.Name;
                                        if (props.LongName) node.LongName = props.LongName;
                                        if (props.GlobalId) node.GlobalId = props.GlobalId;
                                    }
                                }
                            } catch (e) {
                                console.warn("Failed to fetch properties for node:", node.expressID);
                            }

                            // 1. 先递归处理现有的空间子节点
                            if (node.children && node.children.length > 0) {
                                await Promise.all(node.children.map(child => enrichNode(child)));
                            }

                            // 2. 附加包含的构件 (Walls, Windows, Roofs, etc.)
                            const containedIds = elementsMap[node.expressID];
                            if (containedIds && containedIds.length > 0) {
                                if (!node.children) node.children = [];

                                // 并行获取构件信息
                                const elementNodes = await Promise.all(containedIds.map(async (id) => {
                                    try {
                                        const props = await ifcLoaderRef.current.ifcManager.getItemProperties(ifcModel.modelID, id);
                                        const type = await ifcLoaderRef.current.ifcManager.getIfcType(ifcModel.modelID, id);
                                        return {
                                            expressID: id,
                                            type: type, // e.g. 'IFCWINDOW', 'IFCROOF'
                                            Name: props.Name,
                                            GlobalId: props.GlobalId,
                                            children: []
                                        };
                                    } catch (e) {
                                        return null;
                                    }
                                }));

                                // 过滤掉失败的，并加入到 children
                                node.children.push(...elementNodes.filter(n => n));
                            }
                        };

                        console.log("🌳 Enriching spatial structure...");
                        await enrichNode(structure);
                        console.log("✅ Structure enriched:", structure);

                        onLoaded(ifcModel, structure);
                    } catch (err) {
                        console.error("Error loading structure:", err);
                        onLoaded(ifcModel, null);
                    }

                    setLoading(false);
                });
            };

            loadModel();
        }
    }, [file]); // 依赖 file 变化

    // 监听尺寸变化
    useEffect(() => {
        if (cameraRef.current && rendererRef.current) {
            cameraRef.current.aspect = width / height;
            cameraRef.current.updateProjectionMatrix();
            rendererRef.current.setSize(width, height);
        }
    }, [width, height]);

    return (
        <div ref={mountRef} className="relative w-full h-full">
            {loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50 text-emerald-500 font-bold">
                    Parsing IFC...
                </div>
            )}
        </div>
    );
};

export default IfcViewer;