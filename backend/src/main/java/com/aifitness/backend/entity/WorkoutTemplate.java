package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * WorkoutTemplate entity for saving reusable workout routines.
 * Users can create templates and start workouts from them.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "workout_templates")
public class WorkoutTemplate {

    @Id
    private String id;

    @Indexed
    private String userId;

    private String name; // e.g., "Push Day", "Upper Body"

    private String description;

    private String category; // PUSH, PULL, LEGS, UPPER, LOWER, FULL_BODY, CARDIO, CUSTOM

    @Builder.Default
    private List<TemplateExercise> exercises = new ArrayList<>();

    private Integer estimatedDuration; // in minutes

    private String difficulty; // BEGINNER, INTERMEDIATE, ADVANCED

    private boolean isPublic; // Can other users see/copy this template

    private Integer timesUsed; // How many times this template was used

    private String lastUsedDate;

    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    private LocalDateTime updatedAt;

    private boolean isActive = true;

    /**
     * Nested class for template exercise configuration.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class TemplateExercise {
        private String exerciseId;
        private String exerciseName; // Denormalized
        private Integer order;
        private Integer targetSets;
        private String targetReps; // Can be "8-12" or "10"
        private Integer restSeconds;
        private String supersetGroup; // For grouping supersets
        private String notes; // Notes for this exercise
    }
}
