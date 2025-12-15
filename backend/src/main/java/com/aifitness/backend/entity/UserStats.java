package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

/**
 * UserStats entity for tracking gamification stats.
 * Stores XP, level, streaks, and aggregate stats.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "user_stats")
public class UserStats {

    @Id
    private String id;

    @Indexed(unique = true)
    private String userId;

    // XP and Level
    private Integer totalXp;
    private Integer level;

    // Streaks
    private Integer currentStreak;
    private Integer longestStreak;
    private LocalDateTime lastWorkoutDate;

    // Aggregate Stats
    private Integer totalWorkouts;
    private Double totalVolume; // Total weight lifted (kg)
    private Integer totalSets;
    private Integer totalReps;
    private Integer totalMinutes;

    // PRs count
    private Integer totalPRs;

    // Achievements count
    private Integer achievementCount;

    // Leaderboard rank (updated periodically)
    private Integer globalRank;
    private Integer weeklyRank;

    @LastModifiedDate
    private LocalDateTime updatedAt;

    // Helper method to calculate level from XP
    public static int calculateLevel(int xp) {
        // XP thresholds: 100, 250, 500, 1000, 2000, 4000, 8000, etc.
        int level = 1;
        int threshold = 100;
        int xpRemaining = xp;

        while (xpRemaining >= threshold) {
            xpRemaining -= threshold;
            level++;
            threshold = (int) (threshold * 1.5);
        }

        return level;
    }

    public static int xpForNextLevel(int currentLevel) {
        int threshold = 100;
        for (int i = 1; i < currentLevel; i++) {
            threshold = (int) (threshold * 1.5);
        }
        return threshold;
    }
}
