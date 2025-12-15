package com.aifitness.backend.repository;

import com.aifitness.backend.entity.WorkoutTemplate;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository for WorkoutTemplate entity.
 */
@Repository
public interface WorkoutTemplateRepository extends MongoRepository<WorkoutTemplate, String> {

    // Find all templates for a user
    List<WorkoutTemplate> findByUserIdAndIsActiveTrueOrderByCreatedAtDesc(String userId);

    // Find by category
    List<WorkoutTemplate> findByUserIdAndCategoryAndIsActiveTrue(String userId, String category);

    // Find public templates
    List<WorkoutTemplate> findByIsPublicTrueAndIsActiveTrueOrderByTimesUsedDesc();

    // Count templates for a user
    long countByUserIdAndIsActiveTrue(String userId);

    // Find most used templates
    List<WorkoutTemplate> findTop5ByUserIdAndIsActiveTrueOrderByTimesUsedDesc(String userId);
}
