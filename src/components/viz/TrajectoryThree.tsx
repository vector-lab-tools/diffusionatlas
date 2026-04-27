"use client";

import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Line, Html } from "@react-three/drei";
import * as THREE from "three";
import type { Point3 } from "@/lib/geometry/pca";

export interface TrajectoryLayerInput {
  id: string;
  label: string;
  colour: string;
  points: Point3[];
  previews?: Array<string | null>;
}

interface TrajectoryThreeProps {
  /** Single-layer compatibility: existing call sites pass these directly. */
  points?: Point3[];
  previews?: Array<string | null>;
  /** Multi-layer mode: pass an array of named/coloured layers to overlay. */
  layers?: TrajectoryLayerInput[];
  height?: number;
  /**
   * Show a preview thumbnail every Nth step. 1 = every step (dense),
   * higher values thin the swarm so the curve itself stays readable.
   * Endpoints (first and last preview) are always shown regardless.
   */
  previewStride?: number;
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

function Path({ points, color = "#7c2d36" }: { points: Point3[]; color?: string }) {
  const positions = useMemo<[number, number, number][]>(() => points.map((p) => [p[0], p[1], p[2]]), [points]);
  if (positions.length < 2) return null;
  return (
    <Line
      points={positions}
      color={color}
      lineWidth={2}
      transparent
      opacity={0.85}
    />
  );
}

function StepMarkers({ points, color = "#666" }: { points: Point3[]; color?: string }) {
  return (
    <>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.022, 12, 12]} />
          <meshStandardMaterial color={color} />
        </mesh>
      ))}
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

export function TrajectoryThree({ points, previews, layers, height = 480, previewStride = 1 }: TrajectoryThreeProps) {
  // Build a unified list of layers. Legacy single-layer callers still work.
  const renderLayers = useMemo<TrajectoryLayerInput[]>(() => {
    if (layers && layers.length > 0) return layers;
    if (points && points.length > 0) {
      return [{ id: "default", label: "trajectory", colour: "#7c2d36", points, previews }];
    }
    return [];
  }, [layers, points, previews]);

  // Normalise across ALL layers' points jointly so they share a frame.
  const allPoints = useMemo(
    () => renderLayers.flatMap((l) => l.points),
    [renderLayers],
  );
  const { points: normAll } = useMemo(() => normalise(allPoints), [allPoints]);

  // Slice the normalised points back into per-layer segments.
  const normalisedLayers = useMemo(() => {
    let cursor = 0;
    return renderLayers.map((l) => {
      const slice = normAll.slice(cursor, cursor + l.points.length);
      cursor += l.points.length;
      return { ...l, points: slice };
    });
  }, [renderLayers, normAll]);

  return (
    <div style={{ height }} className="rounded-sm border border-parchment bg-cream/30 overflow-hidden">
      <Canvas camera={{ position: [2.5, 2, 2.5], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.6} />
        <directionalLight position={[-5, -3, -5]} intensity={0.3} />
        <gridHelper args={[4, 8, "#d6d6d6", "#ececec"]} position={[0, -1.2, 0]} />
        {normalisedLayers.map((layer) => (
          <group key={layer.id}>
            <Path points={layer.points} color={layer.colour} />
            <StepMarkers points={layer.points} color={layer.colour} />
            {(() => {
              const stride = Math.max(1, Math.floor(previewStride));
              const previewIdxs = (layer.previews ?? [])
                .map((url, i) => ({ url, i }))
                .filter(({ url }) => Boolean(url));
              // Always include the last available preview so the "destination"
              // thumbnail anchors the end of the curve, even when stride skips it.
              const lastIdx = previewIdxs.length > 0 ? previewIdxs[previewIdxs.length - 1].i : -1;
              return previewIdxs
                .filter(({ i }) => i % stride === 0 || i === lastIdx)
                .map(({ url, i }) =>
                  url && i < layer.points.length ? (
                    <PreviewBillboard key={`${layer.id}-${i}`} position={layer.points[i]} dataUrl={url} />
                  ) : null,
                );
            })()}
            {layer.points.length > 0 && (
              <Html position={layer.points[0]} center>
                <div
                  className="font-sans whitespace-nowrap pointer-events-none select-none"
                  style={{ color: layer.colour, fontSize: "9px", letterSpacing: "0.08em", transform: "translate(0, -10px)", opacity: 0.6 }}
                >
                  {normalisedLayers.length === 1 ? "start" : `${layer.label} ·`}
                </div>
              </Html>
            )}
          </group>
        ))}
        <AutoRotate />
        <OrbitControls enablePan enableZoom enableRotate />
      </Canvas>
    </div>
  );
}
