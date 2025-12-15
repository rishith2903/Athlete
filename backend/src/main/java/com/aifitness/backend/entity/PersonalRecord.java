package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.CompoundIndex;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

/**
 * PersonalRecord entity for tracking user's best performances.
 * Tracks PRs for different record types (1RM, max reps, max volume, etc.)
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "personal_records")
@CompoundIndex(name = "user_exercise_type_idx", def = "{'userId': 1, 'exerciseId': 1, 'recordType': 1}")
public class PersonalRecord {

    @Id
    private String id;

    @Indexed
    private String userId;

    @Indexed
    private String exerciseId;

    private String exerciseName; // Denormalized for quick access

    private String recordType; // ONE_REP_MAX, MAX_REPS, MAX_VOLUME, MAX_WEIGHT, BEST_TIME

    private Double value; // The record value

    private String unit; // kg, lbs, reps, seconds, etc.

    // Context of when PR was achieved
    private Double weight; // Weight used (for max reps PR)

    private Integer reps; // Reps done (for 1RM PR)

    private String workoutLogId; // Reference to the workout where PR was set

    private LocalDateTime achievedAt;

    // Previous record for comparison
    private Double previousValue;

    private LocalDateTime previousAchievedAt;

    @CreatedDate
    private LocalDateTime createdAt;
}
