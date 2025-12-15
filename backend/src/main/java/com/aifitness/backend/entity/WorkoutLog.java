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
 * WorkoutLog entity representing a completed workout session.
 * Contains all exercises performed with their sets, reps, and weights.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "workout_logs")
public class WorkoutLog {

    @Id
    private String id;

    @Indexed
    private String userId;

    private String name; // Workout name (e.g., "Push Day", "Leg Day")

    private String templateId; // If started from a template

    @Indexed
    private LocalDateTime startTime;

    private LocalDateTime endTime;

    private Integer durationMinutes; // Total workout duration

    @Builder.Default
    private List<ExerciseLog> exercises = new ArrayList<>();

    private String notes; // Workout notes

    private Integer rating; // User's rating 1-5

    private String mood; // How the user felt: GREAT, GOOD, OKAY, TIRED, EXHAUSTED

    // Calculated stats
    private Integer totalSets;

    private Integer totalReps;

    private Double totalVolume; // Sum of (weight × reps) for all sets

    private Integer estimatedCalories;

    @CreatedDate
    private LocalDateTime createdAt;

    @LastModifiedDate
    private LocalDateTime updatedAt;

    /**
     * Nested class representing a single exercise within a workout log.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class ExerciseLog {
        private String exerciseId;
        private String exerciseName; // Denormalized for quick access
        private Integer order; // Order in the workout
        private String supersetGroup; // For supersets (e.g., "A", "B")

        @Builder.Default
        private List<SetLog> sets = new ArrayList<>();

        private String notes; // Notes for this exercise
    }

    /**
     * Nested class representing a single set within an exercise.
     */
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SetLog {
        private Integer setNumber;
        private Double weight; // in kg or lbs based on user preference
        private Integer reps;
        private Integer duration; // For timed exercises (seconds)
        private Double distance; // For cardio (meters or miles)
        private Integer rpe; // Rate of Perceived Exertion 1-10
        private Integer rir; // Reps In Reserve
        private boolean isWarmup;
        private boolean isDropSet;
        private boolean isFailure; // To failure
        private boolean completed;
        private boolean isPR; // Personal Record achieved
    }
}
