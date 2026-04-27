"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line, Html } from "@react-three/drei";
import * as THREE from "three";
import type { Point3 } from "@/lib/geometry/pca";

interface TrajectoryThreeProps {
  points: Point3[];
  /** Optional preview thumbnails per step (data URLs); same length as points. */
  previews?: Array<string | null>;
  height?: number;
}

function PreviewBillboard({ position, dataUrl }: { position: Point3; dataUrl: string }) {
  const texture = useMemo(() => {
    const loader = new THREE.TextureLoader();
    return loader.load(dataUrl);
  }, [dataUrl]);
  // Slight lift above the step marker so it doesn't sit on top of the path.
  const lifted: Point3 = [position[0], position[1] + 0.18, position[2]];
  return (
    <sprite position={lifted} scale={[0.3, 0.3, 0.3]}>
      <spriteMaterial attach="material" map={texture} transparent />
    </sprite>
  );
}

/**
 * Normalise points to a unit cube around origin so the camera framing is
 * stable regardless of latent magnitude (which varies wildly across models).
 */
interface NormaliseResult {
  points: Point3[];
  /** Largest range across axes before scaling — useful for diagnostics. */
  span: number;
}

function normalise(points: Point3[]): NormaliseResult {
  if (points.length === 0) return { points: [], span: 0 };
  const min: Point3 = [Infinity, Infinity, Infinity];
  const max: Point3 = [-Infinity, -Infinity, -Infinity];
  for (const p of points) {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  }
  const range: Point3 = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const span = Math.max(range[0], range[1], range[2]);
  // If the path collapsed to a single point, fall back to laying the steps
  // out along the X axis so the user at least sees something.
  if (span === 0 || !Number.isFinite(span)) {
    return {
      points: points.map((_, i) => [
        (i / Math.max(1, points.length - 1)) * 2 - 1,
        0,
        0,
      ]),
      span: 0,
    };
  }
  const scale = 1.6 / span;
  return {
    points: points.map((p) => [
      (p[0] - (min[0] + max[0]) / 2) * scale,
      (p[1] - (min[1] + max[1]) / 2) * scale,
      (p[2] - (min[2] + max[2]) / 2) * scale,
    ]),
    span,
  };
}

function Path({ points }: { points: Point3[] }) {
  const positions = useMemo<[number, number, number][]>(() => points.map((p) => [p[0], p[1], p[2]]), [points]);
  if (positions.length < 2) return null;
  return (
    <Line
      points={positions}
      color="#7c2d36"
      lineWidth={2}
      transparent
      opacity={0.85}
    />
  );
}

function StepMarkers({ points }: { points: Point3[] }) {
  const last = points.length - 1;
  return (
    <>
      {points.map((p, i) => {
        const isFirst = i === 0;
        const isLast = i === last;
        const colour = isFirst ? "#c9a227" : isLast ? "#7c2d36" : "#666";
        return (
          <mesh key={i} position={p}>
            <sphereGeometry args={[0.025, 16, 16]} />
            <meshStandardMaterial color={colour} />
          </mesh>
        );
      })}
    </>
  );
}

function AutoRotate() {
  const ref = useRef<THREE.Group>(null);
  useFrame((_, delta) => {
    if (ref.current) ref.current.rotation.y += delta * 0.15;
  });
  return null;
}

export function TrajectoryThree({ points, previews, height = 480 }: TrajectoryThreeProps) {
  const { points: norm, span } = useMemo(() => normalise(points), [points]);

  return (
    <div style={{ height }} className="rounded-sm border border-parchment bg-cream/30 overflow-hidden">
      <Canvas camera={{ position: [2.5, 2, 2.5], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.6} />
        <directionalLight position={[-5, -3, -5]} intensity={0.3} />
        <gridHelper args={[4, 8, "#d6d6d6", "#ececec"]} position={[0, -1.2, 0]} />
        <Path points={norm} />
        <StepMarkers points={norm} />
        {previews?.map((url, i) =>
          url && i < norm.length ? (
            <PreviewBillboard key={i} position={norm[i]} dataUrl={url} />
          ) : null,
        )}
        {norm.length > 0 && (
          <Html position={norm[0]} center>
            <div
              className="font-sans whitespace-nowrap pointer-events-none select-none"
              style={{ color: "#c9a227", fontSize: "9px", letterSpacing: "0.08em", transform: "translate(0, -10px)", opacity: 0.7 }}
            >
              start
            </div>
          </Html>
        )}
        {norm.length > 1 && (
          <Html position={norm[norm.length - 1]} center>
            <div
              className="font-sans whitespace-nowrap pointer-events-none select-none"
              style={{ color: "#7c2d36", fontSize: "9px", letterSpacing: "0.08em", transform: "translate(0, 10px)", opacity: 0.7 }}
            >
              end
            </div>
          </Html>
        )}
        <AutoRotate />
        <OrbitControls enablePan enableZoom enableRotate />
      </Canvas>
    </div>
  );
}
