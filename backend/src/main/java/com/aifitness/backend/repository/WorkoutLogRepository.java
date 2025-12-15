package com.aifitness.backend.repository;

import com.aifitness.backend.entity.WorkoutLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.Aggregation;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Repository for WorkoutLog entity.
 * Provides methods for querying workout history and statistics.
 */
@Repository
public interface WorkoutLogRepository extends MongoRepository<WorkoutLog, String> {

    // Find all workouts for a user
    List<WorkoutLog> findByUserIdOrderByStartTimeDesc(String userId);

    // Find workouts with pagination
    Page<WorkoutLog> findByUserId(String userId, Pageable pageable);

    // Find workouts in a date range
    List<WorkoutLog> findByUserIdAndStartTimeBetween(String userId, LocalDateTime start, LocalDateTime end);

    // Find workouts on a specific day
    @Query("{ 'userId': ?0, 'startTime': { $gte: ?1, $lt: ?2 } }")
    List<WorkoutLog> findByUserIdAndDate(String userId, LocalDateTime startOfDay, LocalDateTime endOfDay);

    // Count workouts for a user
    long countByUserId(String userId);

    // Count workouts in a date range
    long countByUserIdAndStartTimeBetween(String userId, LocalDateTime start, LocalDateTime end);

    // Find recent workouts (last N)
    List<WorkoutLog> findTop10ByUserIdOrderByStartTimeDesc(String userId);

    // Find workouts by template
    List<WorkoutLog> findByUserIdAndTemplateId(String userId, String templateId);

    // Get total volume in a date range
    @Aggregation(pipeline = {
            "{ $match: { 'userId': ?0, 'startTime': { $gte: ?1, $lte: ?2 } } }",
            "{ $group: { _id: null, totalVolume: { $sum: '$totalVolume' }, totalDuration: { $sum: '$durationMinutes' } } }"
    })
    WorkoutStats getWorkoutStats(String userId, LocalDateTime start, LocalDateTime end);

    interface WorkoutStats {
        Double getTotalVolume();

        Integer getTotalDuration();
    }
}
