package com.aifitness.backend.repository;

import com.aifitness.backend.entity.Exercise;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository for Exercise entity.
 * Provides methods for querying the exercise library.
 */
@Repository
public interface ExerciseRepository extends MongoRepository<Exercise, String> {

    // Find by category
    List<Exercise> findByCategoryIgnoreCase(String category);

    // Find by equipment
    List<Exercise> findByEquipmentContaining(String equipment);

    // Find by primary muscle
    List<Exercise> findByPrimaryMusclesContaining(String muscle);

    // Find by difficulty
    List<Exercise> findByDifficultyIgnoreCase(String difficulty);

    // Find by exercise type
    List<Exercise> findByExerciseTypeIgnoreCase(String exerciseType);

    // Search by name (case insensitive)
    List<Exercise> findByNameContainingIgnoreCase(String name);

    // Find active exercises only
    List<Exercise> findByIsActiveTrue();

    // Complex query: Find by category and difficulty
    List<Exercise> findByCategoryIgnoreCaseAndDifficultyIgnoreCase(String category, String difficulty);

    // Complex query: Find by category and equipment available
    @Query("{ 'category': ?0, 'equipment': { $in: ?1 }, 'isActive': true }")
    List<Exercise> findByCategoryAndEquipmentIn(String category, List<String> equipment);

    // Find compound exercises
    List<Exercise> findByIsCompoundTrue();

    // Find by movement pattern
    List<Exercise> findByMovementPatternIgnoreCase(String movementPattern);

    // Count exercises by category
    long countByCategory(String category);
}
