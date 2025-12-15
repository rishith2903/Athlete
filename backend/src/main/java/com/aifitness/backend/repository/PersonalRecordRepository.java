package com.aifitness.backend.repository;

import com.aifitness.backend.entity.PersonalRecord;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository for PersonalRecord entity.
 * Provides methods for querying and updating personal records.
 */
@Repository
public interface PersonalRecordRepository extends MongoRepository<PersonalRecord, String> {

    // Find all PRs for a user
    List<PersonalRecord> findByUserIdOrderByAchievedAtDesc(String userId);

    // Find PRs for a specific exercise
    List<PersonalRecord> findByUserIdAndExerciseId(String userId, String exerciseId);

    // Find specific PR type for an exercise
    Optional<PersonalRecord> findByUserIdAndExerciseIdAndRecordType(String userId, String exerciseId,
            String recordType);

    // Find all PRs of a specific type
    List<PersonalRecord> findByUserIdAndRecordType(String userId, String recordType);

    // Find recent PRs
    List<PersonalRecord> findTop10ByUserIdOrderByAchievedAtDesc(String userId);

    // Find PRs achieved in a workout
    List<PersonalRecord> findByWorkoutLogId(String workoutLogId);

    // Count total PRs for a user
    long countByUserId(String userId);

    // Delete all PRs for an exercise (when exercise is deleted)
    void deleteByExerciseId(String exerciseId);

    // Check if a value would be a new PR
    @Query("{ 'userId': ?0, 'exerciseId': ?1, 'recordType': ?2, 'value': { $gte: ?3 } }")
    Optional<PersonalRecord> findExistingHigherRecord(String userId, String exerciseId, String recordType,
            Double value);
}
