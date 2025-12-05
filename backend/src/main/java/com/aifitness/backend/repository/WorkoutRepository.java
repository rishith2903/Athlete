package com.aifitness.backend.repository;

import com.aifitness.backend.entity.Workout;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface WorkoutRepository extends MongoRepository<Workout, String> {
    
    Page<Workout> findByUserId(String userId, Pageable pageable);
    
    List<Workout> findByUserIdAndStatus(String userId, String status);
    
    @Query("{ 'userId': ?0, 'scheduledFor': { $gte: ?1, $lte: ?2 } }")
    List<Workout> findByUserIdAndScheduledForBetween(String userId, LocalDateTime start, LocalDateTime end);
    
    @Query("{ 'userId': ?0, 'completedAt': { $gte: ?1, $lte: ?2 } }")
    List<Workout> findByUserIdAndCompletedAtBetween(String userId, LocalDateTime start, LocalDateTime end);
    
    @Query("{ 'userId': ?0, 'type': ?1 }")
    List<Workout> findByUserIdAndType(String userId, String type);
    
    @Query("{ 'userId': ?0, 'difficulty': ?1 }")
    List<Workout> findByUserIdAndDifficulty(String userId, String difficulty);
    
    @Query("{ 'userId': ?0, 'aiGenerated': true }")
    List<Workout> findAiGeneratedWorkoutsByUserId(String userId);
    
    @Query("{ 'userId': ?0, 'rating': { $gte: ?1 } }")
    List<Workout> findHighRatedWorkoutsByUserId(String userId, Integer minRating);
    
    @Query("{ 'userId': ?0, 'targetMuscleGroups': { $regex: ?1, $options: 'i' } }")
    List<Workout> findByUserIdAndTargetMuscleGroup(String userId, String muscleGroup);
    
    @Query("{ 'status': 'PLANNED', 'scheduledFor': { $lte: ?0 } }")
    List<Workout> findOverdueWorkouts(LocalDateTime currentTime);
    
    @Query("{ 'userId': ?0, 'status': 'IN_PROGRESS' }")
    List<Workout> findInProgressWorkoutsByUserId(String userId);
    
    Long countByUserIdAndStatus(String userId, String status);
    
    @Query("{ 'userId': ?0, 'completedAt': { $gte: ?1 } }")
    Long countRecentCompletedWorkouts(String userId, LocalDateTime since);
}