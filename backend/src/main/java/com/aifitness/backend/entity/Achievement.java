package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

/**
 * Achievement entity for gamification system.
 * Tracks user achievements/badges earned.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "achievements")
public class Achievement {

    @Id
    private String id;

    @Indexed
    private String userId;

    private String achievementType; // FIRST_WORKOUT, STREAK_7, PR_BROKEN, VOLUME_1000, etc.

    private String name;

    private String description;

    private String icon; // Emoji or icon name

    private String category; // MILESTONE, STREAK, STRENGTH, CONSISTENCY, VOLUME

    private Integer tier; // 1 = Bronze, 2 = Silver, 3 = Gold, 4 = Platinum

    private Integer xpReward; // XP points earned

    private LocalDateTime earnedAt;

    @CreatedDate
    private LocalDateTime createdAt;

    // Reference to the workout/record that triggered this achievement
    private String triggerWorkoutId;
    private String triggerRecordId;
}
