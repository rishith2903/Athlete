import React, { useRef, useState, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, Html } from '@react-three/drei';
import * as THREE from 'three';

/**
 * 3D Muscle Body Component
 * Interactive 3D human body showing muscle groups with color-coded intensity
 */
const Body3D = ({ muscleData = {}, onMuscleClick }) => {
    return (
        <div className="w-full h-[500px] bg-gradient-to-b from-gray-900 to-gray-800 rounded-xl overflow-hidden">
            <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
                <Suspense fallback={<LoadingFallback />}>
                    <ambientLight intensity={0.5} />
                    <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} />
                    <pointLight position={[-10, -10, -10]} intensity={0.5} />

                    <HumanBody muscleData={muscleData} onMuscleClick={onMuscleClick} />

                    <OrbitControls
                        enablePan={false}
                        minDistance={2.5}
                        maxDistance={6}
                        autoRotate
                        autoRotateSpeed={0.5}
                    />
                    <Environment preset="city" />
                </Suspense>
            </Canvas>

            {/* Legend */}
            <div className="absolute bottom-4 left-4 flex gap-3 text-xs bg-black/50 px-3 py-2 rounded-lg">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-gray-500"></div>
                    <span className="text-gray-300">Rest</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-green-400"></div>
                    <span className="text-gray-300">Light</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-yellow-400"></div>
                    <span className="text-gray-300">Moderate</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-orange-500"></div>
                    <span className="text-gray-300">Heavy</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-red-500"></div>
                    <span className="text-gray-300">Intense</span>
                </div>
            </div>
        </div>
    );
};

// Loading fallback
const LoadingFallback = () => (
    <Html center>
        <div className="text-white text-sm">Loading 3D Model...</div>
    </Html>
);

// Human Body Component made of primitives
const HumanBody = ({ muscleData, onMuscleClick }) => {
    const groupRef = useRef();

    // Subtle breathing animation
    useFrame((state) => {
        if (groupRef.current) {
            groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.02;
        }
    });

    const getColor = (muscle) => {
        const intensity = muscleData[muscle] || 0;
        if (intensity === 0) return '#6B7280'; // gray
        if (intensity < 25) return '#4ADE80'; // green
        if (intensity < 50) return '#FACC15'; // yellow
        if (intensity < 75) return '#FB923C'; // orange
        return '#EF4444'; // red
    };

    return (
        <group ref={groupRef}>
            {/* Head */}
            <mesh position={[0, 1.6, 0]}>
                <sphereGeometry args={[0.18, 32, 32]} />
                <meshStandardMaterial color="#E5C4B0" roughness={0.7} />
            </mesh>

            {/* Neck */}
            <mesh position={[0, 1.35, 0]}>
                <cylinderGeometry args={[0.08, 0.1, 0.15, 16]} />
                <meshStandardMaterial color="#E5C4B0" roughness={0.7} />
            </mesh>

            {/* ===== TORSO ===== */}

            {/* Chest - Left */}
            <MuscleGroup
                name="chest"
                label="Chest"
                position={[-0.12, 1.05, 0.12]}
                geometry={<sphereGeometry args={[0.15, 16, 16]} />}
                color={getColor('chest')}
                scale={[1, 0.7, 0.6]}
                onMuscleClick={onMuscleClick}
            />
            {/* Chest - Right */}
            <MuscleGroup
                name="chest"
                label="Chest"
                position={[0.12, 1.05, 0.12]}
                geometry={<sphereGeometry args={[0.15, 16, 16]} />}
                color={getColor('chest')}
                scale={[1, 0.7, 0.6]}
                onMuscleClick={onMuscleClick}
            />

            {/* Abs */}
            <MuscleGroup
                name="abs"
                label="Abs"
                position={[0, 0.75, 0.1]}
                geometry={<boxGeometry args={[0.22, 0.35, 0.12]} />}
                color={getColor('abs')}
                onMuscleClick={onMuscleClick}
            />

            {/* Obliques */}
            <MuscleGroup
                name="obliques"
                label="Obliques"
                position={[-0.18, 0.75, 0.05]}
                geometry={<boxGeometry args={[0.08, 0.3, 0.15]} />}
                color={getColor('obliques')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="obliques"
                label="Obliques"
                position={[0.18, 0.75, 0.05]}
                geometry={<boxGeometry args={[0.08, 0.3, 0.15]} />}
                color={getColor('obliques')}
                onMuscleClick={onMuscleClick}
            />

            {/* Core/Torso Base */}
            <mesh position={[0, 0.85, 0]}>
                <capsuleGeometry args={[0.2, 0.5, 8, 16]} />
                <meshStandardMaterial color="#E5C4B0" roughness={0.7} transparent opacity={0.3} />
            </mesh>

            {/* ===== BACK ===== */}

            {/* Lats */}
            <MuscleGroup
                name="lats"
                label="Lats"
                position={[-0.2, 0.9, -0.08]}
                geometry={<boxGeometry args={[0.12, 0.4, 0.1]} />}
                color={getColor('lats')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="lats"
                label="Lats"
                position={[0.2, 0.9, -0.08]}
                geometry={<boxGeometry args={[0.12, 0.4, 0.1]} />}
                color={getColor('lats')}
                onMuscleClick={onMuscleClick}
            />

            {/* Traps */}
            <MuscleGroup
                name="traps"
                label="Traps"
                position={[0, 1.2, -0.05]}
                geometry={<boxGeometry args={[0.35, 0.15, 0.1]} />}
                color={getColor('traps')}
                onMuscleClick={onMuscleClick}
            />

            {/* Lower Back */}
            <MuscleGroup
                name="lower_back"
                label="Lower Back"
                position={[0, 0.6, -0.12]}
                geometry={<boxGeometry args={[0.2, 0.25, 0.1]} />}
                color={getColor('lower_back')}
                onMuscleClick={onMuscleClick}
            />

            {/* ===== SHOULDERS ===== */}

            {/* Shoulders */}
            <MuscleGroup
                name="shoulders"
                label="Shoulders"
                position={[-0.32, 1.15, 0]}
                geometry={<sphereGeometry args={[0.1, 16, 16]} />}
                color={getColor('shoulders')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="shoulders"
                label="Shoulders"
                position={[0.32, 1.15, 0]}
                geometry={<sphereGeometry args={[0.1, 16, 16]} />}
                color={getColor('shoulders')}
                onMuscleClick={onMuscleClick}
            />

            {/* ===== ARMS ===== */}

            {/* Biceps */}
            <MuscleGroup
                name="biceps"
                label="Biceps"
                position={[-0.38, 0.9, 0.04]}
                geometry={<capsuleGeometry args={[0.06, 0.2, 8, 16]} />}
                color={getColor('biceps')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="biceps"
                label="Biceps"
                position={[0.38, 0.9, 0.04]}
                geometry={<capsuleGeometry args={[0.06, 0.2, 8, 16]} />}
                color={getColor('biceps')}
                onMuscleClick={onMuscleClick}
            />

            {/* Triceps */}
            <MuscleGroup
                name="triceps"
                label="Triceps"
                position={[-0.38, 0.9, -0.04]}
                geometry={<capsuleGeometry args={[0.055, 0.2, 8, 16]} />}
                color={getColor('triceps')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="triceps"
                label="Triceps"
                position={[0.38, 0.9, -0.04]}
                geometry={<capsuleGeometry args={[0.055, 0.2, 8, 16]} />}
                color={getColor('triceps')}
                onMuscleClick={onMuscleClick}
            />

            {/* Forearms */}
            <MuscleGroup
                name="forearms"
                label="Forearms"
                position={[-0.4, 0.6, 0]}
                geometry={<capsuleGeometry args={[0.04, 0.22, 8, 16]} />}
                color={getColor('forearms')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="forearms"
                label="Forearms"
                position={[0.4, 0.6, 0]}
                geometry={<capsuleGeometry args={[0.04, 0.22, 8, 16]} />}
                color={getColor('forearms')}
                onMuscleClick={onMuscleClick}
            />

            {/* ===== LEGS ===== */}

            {/* Glutes */}
            <MuscleGroup
                name="glutes"
                label="Glutes"
                position={[-0.1, 0.42, -0.08]}
                geometry={<sphereGeometry args={[0.12, 16, 16]} />}
                color={getColor('glutes')}
                scale={[1, 0.8, 0.8]}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="glutes"
                label="Glutes"
                position={[0.1, 0.42, -0.08]}
                geometry={<sphereGeometry args={[0.12, 16, 16]} />}
                color={getColor('glutes')}
                scale={[1, 0.8, 0.8]}
                onMuscleClick={onMuscleClick}
            />

            {/* Quads */}
            <MuscleGroup
                name="quads"
                label="Quads"
                position={[-0.12, 0.05, 0.05]}
                geometry={<capsuleGeometry args={[0.08, 0.35, 8, 16]} />}
                color={getColor('quads')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="quads"
                label="Quads"
                position={[0.12, 0.05, 0.05]}
                geometry={<capsuleGeometry args={[0.08, 0.35, 8, 16]} />}
                color={getColor('quads')}
                onMuscleClick={onMuscleClick}
            />

            {/* Hamstrings */}
            <MuscleGroup
                name="hamstrings"
                label="Hamstrings"
                position={[-0.12, 0.05, -0.05]}
                geometry={<capsuleGeometry args={[0.07, 0.35, 8, 16]} />}
                color={getColor('hamstrings')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="hamstrings"
                label="Hamstrings"
                position={[0.12, 0.05, -0.05]}
                geometry={<capsuleGeometry args={[0.07, 0.35, 8, 16]} />}
                color={getColor('hamstrings')}
                onMuscleClick={onMuscleClick}
            />

            {/* Calves */}
            <MuscleGroup
                name="calves"
                label="Calves"
                position={[-0.12, -0.45, -0.02]}
                geometry={<capsuleGeometry args={[0.05, 0.25, 8, 16]} />}
                color={getColor('calves')}
                onMuscleClick={onMuscleClick}
            />
            <MuscleGroup
                name="calves"
                label="Calves"
                position={[0.12, -0.45, -0.02]}
                geometry={<capsuleGeometry args={[0.05, 0.25, 8, 16]} />}
                color={getColor('calves')}
                onMuscleClick={onMuscleClick}
            />
        </group>
    );
};

// Individual Muscle Group Component with hover/click effects
const MuscleGroup = ({ name, label, position, geometry, color, scale = [1, 1, 1], onMuscleClick }) => {
    const meshRef = useRef();
    const [hovered, setHovered] = useState(false);

    useFrame(() => {
        if (meshRef.current && hovered) {
            meshRef.current.scale.x = scale[0] * 1.1;
            meshRef.current.scale.y = scale[1] * 1.1;
            meshRef.current.scale.z = scale[2] * 1.1;
        } else if (meshRef.current) {
            meshRef.current.scale.x = THREE.MathUtils.lerp(meshRef.current.scale.x, scale[0], 0.1);
            meshRef.current.scale.y = THREE.MathUtils.lerp(meshRef.current.scale.y, scale[1], 0.1);
            meshRef.current.scale.z = THREE.MathUtils.lerp(meshRef.current.scale.z, scale[2], 0.1);
        }
    });

    return (
        <mesh
            ref={meshRef}
            position={position}
            scale={scale}
            onClick={() => onMuscleClick?.(name, label)}
            onPointerOver={(e) => {
                e.stopPropagation();
                setHovered(true);
                document.body.style.cursor = 'pointer';
            }}
            onPointerOut={() => {
                setHovered(false);
                document.body.style.cursor = 'auto';
            }}
        >
            {geometry}
            <meshStandardMaterial
                color={hovered ? '#FFFFFF' : color}
                roughness={0.4}
                metalness={0.2}
                emissive={hovered ? color : '#000000'}
                emissiveIntensity={hovered ? 0.5 : 0}
            />
            {hovered && (
                <Html position={[0, 0.2, 0]} center>
                    <div className="bg-black/80 text-white px-2 py-1 rounded text-sm whitespace-nowrap">
                        {label}
                    </div>
                </Html>
            )}
        </mesh>
    );
};

export default Body3D;
