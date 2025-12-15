package com.aifitness.backend.repository;

import com.aifitness.backend.entity.Achievement;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository for Achievement entity.
 */
@Repository
public interface AchievementRepository extends MongoRepository<Achievement, String> {

    // Get all achievements for a user
    List<Achievement> findByUserIdOrderByEarnedAtDesc(String userId);

    // Get achievements by category
    List<Achievement> findByUserIdAndCategoryOrderByEarnedAtDesc(String userId, String category);

    // Check if user has specific achievement
    boolean existsByUserIdAndAchievementType(String userId, String achievementType);

    // Get recent achievements
    List<Achievement> findTop5ByUserIdOrderByEarnedAtDesc(String userId);

    // Count achievements
    long countByUserId(String userId);

    // Count by category
    long countByUserIdAndCategory(String userId, String category);
}
