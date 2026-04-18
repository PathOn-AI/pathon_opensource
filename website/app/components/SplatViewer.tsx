"use client";

import { Canvas, useThree } from "@react-three/fiber";
import {
  GizmoHelper,
  GizmoViewport,
  Grid,
  OrbitControls,
} from "@react-three/drei";
import {
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

type FitFn = () => void;

function SparkSceneSetup() {
  const { gl, scene } = useThree();

  useEffect(() => {
    const spark = new SparkRenderer({ renderer: gl });
    scene.add(spark);
    return () => {
      scene.remove(spark);
    };
  }, [gl, scene]);

  return null;
}

type SplatProps = {
  url: string;
  onReady?: () => void;
  onFitReady?: (fit: FitFn) => void;
};

function Splat({ url, onReady, onFitReady }: SplatProps) {
  const { camera, controls } = useThree() as unknown as {
    camera: THREE.PerspectiveCamera;
    controls: { target: THREE.Vector3; update: () => void } | null;
  };

  const [mesh, setMesh] = useState<SplatMesh | null>(null);
  const meshRef = useRef<SplatMesh | null>(null);

  const fitToScene = useCallback(() => {
    const m = meshRef.current;
    if (!m) return;
    if (!m.packedSplats && !m.extSplats) return;

    m.position.set(0, 0, 0);
    m.updateMatrixWorld(true);

    const bbox = m.getBoundingBox(false);
    const center = new THREE.Vector3();
    const size = new THREE.Vector3();
    bbox.getCenter(center);
    bbox.getSize(size);

    // eslint-disable-next-line no-console
    console.log("[SplatViewer] bbox", {
      center: center.toArray(),
      size: size.toArray(),
      length: size.length(),
    });

    m.position.sub(center);

    const radius = Math.max(0.1, size.length() * 0.5);
    const distance =
      (radius / Math.sin((camera.fov * Math.PI) / 360)) * 1.3;

    const dir = new THREE.Vector3(1, 0.7, 1).normalize();
    camera.position.copy(dir).multiplyScalar(distance);
    camera.near = Math.max(0.001, distance / 1000);
    camera.far = Math.max(distance * 100, 1000);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();

    if (controls?.target) {
      controls.target.set(0, 0, 0);
      controls.update?.();
    }
  }, [camera, controls]);

  useEffect(() => {
    onFitReady?.(fitToScene);
  }, [fitToScene, onFitReady]);

  useEffect(() => {
    let cancelled = false;
    const m = new SplatMesh({ url });
    setMesh(m);
    m.initialized
      .then((loaded) => {
        if (cancelled) return;
        if (!loaded.packedSplats && !loaded.extSplats) return;
        meshRef.current = loaded;
        fitToScene();
        onReady?.();
      })
      .catch((err) => {
        console.error("[SplatViewer] failed to load:", err);
      });

    return () => {
      cancelled = true;
      meshRef.current = null;
      m.dispose();
    };
  }, [url, fitToScene, onReady]);

  if (!mesh) return null;
  return <primitive object={mesh as unknown as THREE.Object3D} />;
}

type SplatViewerProps = {
  url: string;
};

export default function SplatViewer({ url }: SplatViewerProps) {
  const [ready, setReady] = useState(false);
  const onReady = useRef(() => setReady(true)).current;
  const fitRef = useRef<FitFn | null>(null);

  useEffect(() => {
    setReady(false);
  }, [url]);

  const onFitReady = useCallback((fit: FitFn) => {
    fitRef.current = fit;
  }, []);

  return (
    <div className="relative w-full h-full">
      <Canvas
        camera={{ position: [0, 0, 3], fov: 60, near: 0.01, far: 1000 }}
        gl={{ antialias: false, premultipliedAlpha: true }}
        style={{ background: "#0a0a0a" }}
      >
        <SparkSceneSetup />
        <Suspense fallback={null}>
          <Splat url={url} onReady={onReady} onFitReady={onFitReady} />
        </Suspense>
        <axesHelper args={[1]} />
        <Grid
          args={[20, 20]}
          cellColor="#3a3a3a"
          sectionColor="#606060"
          sectionSize={5}
          fadeDistance={30}
          fadeStrength={1}
          infiniteGrid
        />
        <OrbitControls enableDamping makeDefault />
        <GizmoHelper alignment="top-right" margin={[64, 64]}>
          <GizmoViewport
            axisColors={["#e53e3e", "#38a169", "#3182ce"]}
            labelColor="white"
          />
        </GizmoHelper>
      </Canvas>
      <button
        onClick={() => fitRef.current?.()}
        className="absolute bottom-3 right-3 z-10 bg-black/60 hover:bg-black/80 backdrop-blur text-white text-sm rounded px-3 py-1.5 transition-colors"
        title="Re-fit camera to the splat"
      >
        Fit to scene
      </button>
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none text-white/80 text-sm">
          Decoding splat…
        </div>
      )}
    </div>
  );
}
