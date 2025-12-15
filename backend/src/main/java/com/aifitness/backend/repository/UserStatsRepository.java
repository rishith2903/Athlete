package com.aifitness.backend.repository;

import com.aifitness.backend.entity.UserStats;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository for UserStats entity.
 */
@Repository
public interface UserStatsRepository extends MongoRepository<UserStats, String> {

    // Get stats by user ID
    Optional<UserStats> findByUserId(String userId);

    // Get top users by XP (global leaderboard)
    List<UserStats> findTop100ByOrderByTotalXpDesc();

    // Get top users by streak
    List<UserStats> findTop100ByOrderByCurrentStreakDesc();

    // Get top users by total volume
    List<UserStats> findTop100ByOrderByTotalVolumeDesc();

    // Get top users by total workouts
    List<UserStats> findTop100ByOrderByTotalWorkoutsDesc();
}
