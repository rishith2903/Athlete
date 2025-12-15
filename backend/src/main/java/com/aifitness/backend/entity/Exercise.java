package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.index.TextIndexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Exercise entity representing an exercise in the exercise library.
 * Contains exercise details, muscle targeting, equipment needed, and
 * instructions.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "exercises")
public class Exercise {

    @Id
    private String id;

    @TextIndexed
    private String name;

    @Indexed
    private String category; // CHEST, BACK, SHOULDERS, LEGS, ARMS, CORE, CARDIO, FULL_BODY

    @Builder.Default
    private Set<String> primaryMuscles = new HashSet<>(); // chest, lats, quads, etc.

    @Builder.Default
    private Set<String> secondaryMuscles = new HashSet<>();

    @Builder.Default
    private Set<String> equipment = new HashSet<>(); // barbell, dumbbells, machine, bodyweight, etc.

    private String exerciseType; // STRENGTH, CARDIO, FLEXIBILITY, PLYOMETRIC

    private String difficulty; // BEGINNER, INTERMEDIATE, ADVANCED

    private String instructions; // How to perform the exercise

    private List<String> tips; // Form tips

    private String videoUrl; // Demo video URL

    private String imageUrl; // Exercise image

    private String animationUrl; // GIF animation

    // Tracking defaults
    private String trackingType; // WEIGHT_REPS, REPS_ONLY, TIME, DISTANCE

    private Integer defaultSets;

    private Integer defaultReps;

    private Integer defaultRestSeconds;

    // Metadata
    private boolean isCompound; // Compound vs isolation exercise

    private boolean isBilateral; // Works both sides equally

    private String movementPattern; // PUSH, PULL, SQUAT, HINGE, CARRY, ROTATION

    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    private LocalDateTime updatedAt;

    private boolean isActive = true;
}
