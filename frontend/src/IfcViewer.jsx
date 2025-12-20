import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { IFCLoader } from 'web-ifc-three';

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
        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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

        // 6. IFC Loader Setup
        const ifcLoader = new IFCLoader();
        // 指向本地的 wasm 路径
        ifcLoader.ifcManager.setWasmPath('./');
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

            ifcLoaderRef.current.load(url, async (ifcModel) => {
                modelRef.current = ifcModel;
                sceneRef.current.add(ifcModel);

                // 获取空间结构树 (Building -> Storey -> Space)
                try {
                    const structure = await ifcLoaderRef.current.ifcManager.getSpatialStructure(ifcModel.modelID);

                    // --- 递归获取节点属性 (Name, GlobalId) ---
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

                        if (node.children && node.children.length > 0) {
                            await Promise.all(node.children.map(child => enrichNode(child)));
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