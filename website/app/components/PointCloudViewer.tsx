"use client";

import React, { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";

interface PointCloudData {
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
}

interface PointCloudViewerProps {
  pointCloud: PointCloudData;
}

function PointCloudPoints({
  positions,
  colors,
  pointSize,
}: {
  positions: Float32Array;
  colors: Float32Array;
  pointSize: number;
}) {
  const pointsRef = useRef<THREE.Points>(null);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }, [positions, colors]);

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial size={pointSize} vertexColors sizeAttenuation />
    </points>
  );
}

function Axis({
  start,
  end,
  color,
}: {
  start: [number, number, number];
  end: [number, number, number];
  color: string;
}) {
  const line = useMemo(() => {
    const geometry = new THREE.BufferGeometry().setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array([...start, ...end]), 3)
    );
    return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));
  }, [start, end, color]);

  return <primitive object={line} />;
}

function CoordinateAxes({ size = 0.2 }: { size?: number }) {
  return (
    <group>
      <Axis start={[0, 0, 0]} end={[size, 0, 0]} color="#ef4444" />
      <Axis start={[0, 0, 0]} end={[0, size, 0]} color="#22c55e" />
      <Axis start={[0, 0, 0]} end={[0, 0, size]} color="#3b82f6" />
    </group>
  );
}

function GridPlane({
  size = 2,
  divisions = 20,
  position = [0, 0, -0.05] as [number, number, number],
}) {
  const grid = useMemo(() => {
    return new THREE.GridHelper(size, divisions, "#444444", "#333333");
  }, [size, divisions]);

  return (
    <primitive
      object={grid}
      position={position}
      rotation={[Math.PI / 2, 0, 0]}
    />
  );
}

function CameraController({ pointCloud }: { pointCloud: PointCloudData }) {
  const { camera } = useThree();
  const initialized = useRef(false);

  useEffect(() => {
    if (initialized.current) return;
    if (!pointCloud || pointCloud.positions.length === 0) return;

    initialized.current = true;

    let minX = Infinity,
      minY = Infinity,
      minZ = Infinity;
    let maxX = -Infinity,
      maxY = -Infinity,
      maxZ = -Infinity;

    for (let i = 0; i < pointCloud.positions.length; i += 3) {
      minX = Math.min(minX, pointCloud.positions[i]);
      maxX = Math.max(maxX, pointCloud.positions[i]);
      minY = Math.min(minY, pointCloud.positions[i + 1]);
      maxY = Math.max(maxY, pointCloud.positions[i + 1]);
      minZ = Math.min(minZ, pointCloud.positions[i + 2]);
      maxZ = Math.max(maxZ, pointCloud.positions[i + 2]);
    }

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const sizeX = maxX - minX;
    const sizeY = maxY - minY;
    const sizeZ = maxZ - minZ;
    const maxSize = Math.max(sizeX, sizeY, sizeZ, 0.5);

    const distance = maxSize * 1.8;
    camera.position.set(
      centerX + distance * 0.6,
      centerY + distance * 0.6,
      centerZ + distance * 0.8
    );
    camera.lookAt(centerX, centerY, centerZ);
    camera.up.set(0, 0, 1);
    camera.updateProjectionMatrix();
  }, [pointCloud, camera]);

  return null;
}

const PointCloudViewer: React.FC<PointCloudViewerProps> = ({ pointCloud }) => {
  const [pointSize, setPointSize] = useState(0.003);

  return (
    <div className="relative w-full" style={{ height: "600px" }}>
      <Canvas
        camera={{ fov: 50, near: 0.001, far: 1000 }}
        style={{ background: "#1a1a2e" }}
      >
        <ambientLight intensity={0.6} />
        <pointLight position={[5, 5, 5]} intensity={1} />
        <directionalLight position={[-5, 5, 5]} intensity={0.5} />
        <directionalLight position={[0, -5, 5]} intensity={0.3} />

        <CameraController pointCloud={pointCloud} />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          rotateSpeed={0.5}
          zoomSpeed={0.8}
          panSpeed={0.5}
        />

        <CoordinateAxes />
        <GridPlane />

        {pointCloud.positions.length > 0 && (
          <PointCloudPoints
            positions={pointCloud.positions}
            colors={pointCloud.colors}
            pointSize={pointSize}
          />
        )}
      </Canvas>

      {/* Axis legend */}
      <div className="absolute top-4 left-4 bg-black/70 text-white text-xs p-3 rounded-lg">
        <p className="font-semibold mb-2">Axes</p>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-1 rounded"
              style={{ backgroundColor: "#ef4444" }}
            ></div>
            <span>X</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-1 rounded"
              style={{ backgroundColor: "#22c55e" }}
            ></div>
            <span>Y</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-1 rounded"
              style={{ backgroundColor: "#3b82f6" }}
            ></div>
            <span>Z</span>
          </div>
        </div>
      </div>

      {/* Point size control */}
      <div className="absolute top-4 right-4 bg-black/70 text-white text-xs p-3 rounded-lg">
        <label className="flex items-center gap-2">
          <span>Point Size:</span>
          <input
            type="range"
            min="0.001"
            max="0.01"
            step="0.001"
            value={pointSize}
            onChange={(e) => setPointSize(parseFloat(e.target.value))}
            className="w-20"
          />
          <span>{pointSize.toFixed(3)}</span>
        </label>
      </div>

      {/* Controls help */}
      <div className="absolute bottom-4 left-4 bg-black/50 text-white text-xs p-2 rounded">
        <p>Left click + drag: Rotate</p>
        <p>Right click + drag: Pan</p>
        <p>Scroll: Zoom</p>
      </div>
    </div>
  );
};

export default PointCloudViewer;
