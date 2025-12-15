package com.aifitness.backend.service;

import com.aifitness.backend.entity.Achievement;
import com.aifitness.backend.entity.UserStats;
import com.aifitness.backend.entity.WorkoutLog;
import com.aifitness.backend.repository.AchievementRepository;
import com.aifitness.backend.repository.UserStatsRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;

/**
 * Service for gamification features - achievements, XP, levels, streaks.
 */
@Service
@RequiredArgsConstructor
public class GamificationService {

    private final AchievementRepository achievementRepository;
    private final UserStatsRepository userStatsRepository;

    // XP rewards for different actions
    private static final int XP_WORKOUT_COMPLETE = 50;
    private static final int XP_PR_BROKEN = 100;
    private static final int XP_STREAK_DAY = 10;
    private static final int XP_ACHIEVEMENT = 25;

    // Achievement definitions
    private static final Map<String, AchievementDef> ACHIEVEMENT_DEFS = new HashMap<>();

    static {
        // Milestone achievements
        ACHIEVEMENT_DEFS.put("FIRST_WORKOUT",
                new AchievementDef("First Workout", "Complete your first workout", "💪", "MILESTONE", 1, 50));
        ACHIEVEMENT_DEFS.put("WORKOUT_10",
                new AchievementDef("Getting Started", "Complete 10 workouts", "🏋️", "MILESTONE", 1, 100));
        ACHIEVEMENT_DEFS.put("WORKOUT_50",
                new AchievementDef("Dedicated", "Complete 50 workouts", "🌟", "MILESTONE", 2, 250));
        ACHIEVEMENT_DEFS.put("WORKOUT_100",
                new AchievementDef("Century Club", "Complete 100 workouts", "🏆", "MILESTONE", 3, 500));
        ACHIEVEMENT_DEFS.put("WORKOUT_500",
                new AchievementDef("Legendary", "Complete 500 workouts", "👑", "MILESTONE", 4, 1000));

        // Streak achievements
        ACHIEVEMENT_DEFS.put("STREAK_7",
                new AchievementDef("Week Warrior", "7 day workout streak", "🔥", "STREAK", 1, 100));
        ACHIEVEMENT_DEFS.put("STREAK_14",
                new AchievementDef("Two Week Titan", "14 day workout streak", "🔥🔥", "STREAK", 2, 200));
        ACHIEVEMENT_DEFS.put("STREAK_30",
                new AchievementDef("Monthly Monster", "30 day workout streak", "🔥🔥🔥", "STREAK", 3, 500));
        ACHIEVEMENT_DEFS.put("STREAK_100",
                new AchievementDef("Unstoppable", "100 day workout streak", "💎", "STREAK", 4, 1000));

        // Strength achievements
        ACHIEVEMENT_DEFS.put("FIRST_PR",
                new AchievementDef("Personal Best", "Break your first PR", "⭐", "STRENGTH", 1, 75));
        ACHIEVEMENT_DEFS.put("PR_10", new AchievementDef("PR Hunter", "Break 10 PRs", "🎯", "STRENGTH", 2, 200));
        ACHIEVEMENT_DEFS.put("PR_50", new AchievementDef("Strength Legend", "Break 50 PRs", "💪", "STRENGTH", 3, 500));

        // Volume achievements
        ACHIEVEMENT_DEFS.put("VOLUME_10K",
                new AchievementDef("10K Club", "Lift 10,000 kg total", "🏗️", "VOLUME", 1, 100));
        ACHIEVEMENT_DEFS.put("VOLUME_100K",
                new AchievementDef("100K Club", "Lift 100,000 kg total", "🏗️🏗️", "VOLUME", 2, 300));
        ACHIEVEMENT_DEFS.put("VOLUME_1M",
                new AchievementDef("Million Pound Club", "Lift 1,000,000 kg total", "🏗️🏗️🏗️", "VOLUME", 4, 1000));
    }

    /**
     * Process completed workout and update stats/achievements
     */
    public List<Achievement> processWorkoutCompletion(String userId, WorkoutLog workoutLog) {
        List<Achievement> newAchievements = new ArrayList<>();

        // Get or create user stats
        UserStats stats = getOrCreateUserStats(userId);

        // Update workout count
        stats.setTotalWorkouts(stats.getTotalWorkouts() + 1);
        stats.setTotalVolume(
                stats.getTotalVolume() + (workoutLog.getTotalVolume() != null ? workoutLog.getTotalVolume() : 0));
        stats.setTotalSets(stats.getTotalSets() + (workoutLog.getTotalSets() != null ? workoutLog.getTotalSets() : 0));
        stats.setTotalReps(stats.getTotalReps() + (workoutLog.getTotalReps() != null ? workoutLog.getTotalReps() : 0));

        // Update streak
        LocalDateTime lastWorkout = stats.getLastWorkoutDate();
        LocalDateTime now = LocalDateTime.now();

        if (lastWorkout != null) {
            long daysSince = ChronoUnit.DAYS.between(lastWorkout.toLocalDate(), now.toLocalDate());
            if (daysSince <= 1) {
                stats.setCurrentStreak(stats.getCurrentStreak() + 1);
            } else {
                stats.setCurrentStreak(1);
            }
        } else {
            stats.setCurrentStreak(1);
        }

        if (stats.getCurrentStreak() > stats.getLongestStreak()) {
            stats.setLongestStreak(stats.getCurrentStreak());
        }

        stats.setLastWorkoutDate(now);

        // Award XP
        int xpEarned = XP_WORKOUT_COMPLETE + (stats.getCurrentStreak() * XP_STREAK_DAY);
        stats.setTotalXp(stats.getTotalXp() + xpEarned);
        stats.setLevel(UserStats.calculateLevel(stats.getTotalXp()));

        // Check for achievements
        newAchievements.addAll(checkMilestoneAchievements(userId, stats, workoutLog.getId()));
        newAchievements.addAll(checkStreakAchievements(userId, stats, workoutLog.getId()));
        newAchievements.addAll(checkVolumeAchievements(userId, stats, workoutLog.getId()));

        // Update achievement count
        stats.setAchievementCount((int) achievementRepository.countByUserId(userId));

        userStatsRepository.save(stats);

        return newAchievements;
    }

    /**
     * Process PR broken and award achievements
     */
    public List<Achievement> processPRBroken(String userId, String workoutLogId) {
        List<Achievement> newAchievements = new ArrayList<>();

        UserStats stats = getOrCreateUserStats(userId);
        stats.setTotalPRs(stats.getTotalPRs() + 1);
        stats.setTotalXp(stats.getTotalXp() + XP_PR_BROKEN);
        stats.setLevel(UserStats.calculateLevel(stats.getTotalXp()));

        // Check PR achievements
        if (stats.getTotalPRs() == 1 && !hasAchievement(userId, "FIRST_PR")) {
            newAchievements.add(grantAchievement(userId, "FIRST_PR", workoutLogId));
        }
        if (stats.getTotalPRs() >= 10 && !hasAchievement(userId, "PR_10")) {
            newAchievements.add(grantAchievement(userId, "PR_10", workoutLogId));
        }
        if (stats.getTotalPRs() >= 50 && !hasAchievement(userId, "PR_50")) {
            newAchievements.add(grantAchievement(userId, "PR_50", workoutLogId));
        }

        userStatsRepository.save(stats);
        return newAchievements;
    }

    private List<Achievement> checkMilestoneAchievements(String userId, UserStats stats, String workoutId) {
        List<Achievement> achievements = new ArrayList<>();
        int count = stats.getTotalWorkouts();

        if (count == 1 && !hasAchievement(userId, "FIRST_WORKOUT")) {
            achievements.add(grantAchievement(userId, "FIRST_WORKOUT", workoutId));
        }
        if (count >= 10 && !hasAchievement(userId, "WORKOUT_10")) {
            achievements.add(grantAchievement(userId, "WORKOUT_10", workoutId));
        }
        if (count >= 50 && !hasAchievement(userId, "WORKOUT_50")) {
            achievements.add(grantAchievement(userId, "WORKOUT_50", workoutId));
        }
        if (count >= 100 && !hasAchievement(userId, "WORKOUT_100")) {
            achievements.add(grantAchievement(userId, "WORKOUT_100", workoutId));
        }
        if (count >= 500 && !hasAchievement(userId, "WORKOUT_500")) {
            achievements.add(grantAchievement(userId, "WORKOUT_500", workoutId));
        }

        return achievements;
    }

    private List<Achievement> checkStreakAchievements(String userId, UserStats stats, String workoutId) {
        List<Achievement> achievements = new ArrayList<>();
        int streak = stats.getCurrentStreak();

        if (streak >= 7 && !hasAchievement(userId, "STREAK_7")) {
            achievements.add(grantAchievement(userId, "STREAK_7", workoutId));
        }
        if (streak >= 14 && !hasAchievement(userId, "STREAK_14")) {
            achievements.add(grantAchievement(userId, "STREAK_14", workoutId));
        }
        if (streak >= 30 && !hasAchievement(userId, "STREAK_30")) {
            achievements.add(grantAchievement(userId, "STREAK_30", workoutId));
        }
        if (streak >= 100 && !hasAchievement(userId, "STREAK_100")) {
            achievements.add(grantAchievement(userId, "STREAK_100", workoutId));
        }

        return achievements;
    }

    private List<Achievement> checkVolumeAchievements(String userId, UserStats stats, String workoutId) {
        List<Achievement> achievements = new ArrayList<>();
        double volume = stats.getTotalVolume();

        if (volume >= 10000 && !hasAchievement(userId, "VOLUME_10K")) {
            achievements.add(grantAchievement(userId, "VOLUME_10K", workoutId));
        }
        if (volume >= 100000 && !hasAchievement(userId, "VOLUME_100K")) {
            achievements.add(grantAchievement(userId, "VOLUME_100K", workoutId));
        }
        if (volume >= 1000000 && !hasAchievement(userId, "VOLUME_1M")) {
            achievements.add(grantAchievement(userId, "VOLUME_1M", workoutId));
        }

        return achievements;
    }

    private boolean hasAchievement(String userId, String type) {
        return achievementRepository.existsByUserIdAndAchievementType(userId, type);
    }

    private Achievement grantAchievement(String userId, String type, String workoutId) {
        AchievementDef def = ACHIEVEMENT_DEFS.get(type);
        if (def == null)
            return null;

        Achievement achievement = Achievement.builder()
                .userId(userId)
                .achievementType(type)
                .name(def.name)
                .description(def.description)
                .icon(def.icon)
                .category(def.category)
                .tier(def.tier)
                .xpReward(def.xp)
                .earnedAt(LocalDateTime.now())
                .triggerWorkoutId(workoutId)
                .build();

        return achievementRepository.save(achievement);
    }

    public UserStats getOrCreateUserStats(String userId) {
        return userStatsRepository.findByUserId(userId)
                .orElseGet(() -> {
                    UserStats stats = UserStats.builder()
                            .userId(userId)
                            .totalXp(0)
                            .level(1)
                            .currentStreak(0)
                            .longestStreak(0)
                            .totalWorkouts(0)
                            .totalVolume(0.0)
                            .totalSets(0)
                            .totalReps(0)
                            .totalMinutes(0)
                            .totalPRs(0)
                            .achievementCount(0)
                            .build();
                    return userStatsRepository.save(stats);
                });
    }

    public List<Achievement> getUserAchievements(String userId) {
        return achievementRepository.findByUserIdOrderByEarnedAtDesc(userId);
    }

    public List<Achievement> getRecentAchievements(String userId) {
        return achievementRepository.findTop5ByUserIdOrderByEarnedAtDesc(userId);
    }

    public List<UserStats> getLeaderboard(String type, int limit) {
        switch (type.toLowerCase()) {
            case "xp":
                return userStatsRepository.findTop100ByOrderByTotalXpDesc().stream().limit(limit).toList();
            case "streak":
                return userStatsRepository.findTop100ByOrderByCurrentStreakDesc().stream().limit(limit).toList();
            case "volume":
                return userStatsRepository.findTop100ByOrderByTotalVolumeDesc().stream().limit(limit).toList();
            case "workouts":
                return userStatsRepository.findTop100ByOrderByTotalWorkoutsDesc().stream().limit(limit).toList();
            default:
                return userStatsRepository.findTop100ByOrderByTotalXpDesc().stream().limit(limit).toList();
        }
    }

    // Helper class for achievement definitions
    private record AchievementDef(String name, String description, String icon, String category, int tier, int xp) {
    }
}
