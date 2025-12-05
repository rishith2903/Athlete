package com.aifitness.backend.repository;

import com.aifitness.backend.entity.Meal;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.Aggregation;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@Repository
public interface MealRepository extends MongoRepository<Meal, String> {
    
    Page<Meal> findByUserId(String userId, Pageable pageable);
    
    List<Meal> findByUserIdAndDate(String userId, LocalDate date);
    
    List<Meal> findByUserIdAndDateBetween(String userId, LocalDate startDate, LocalDate endDate);
    
    List<Meal> findByUserIdAndType(String userId, String type);
    
    @Query("{ 'userId': ?0, 'consumed': true, 'date': { $gte: ?1, $lte: ?2 } }")
    List<Meal> findConsumedMealsByUserIdAndDateRange(String userId, LocalDate startDate, LocalDate endDate);
    
    @Query("{ 'userId': ?0, 'aiGenerated': true }")
    List<Meal> findAiGeneratedMealsByUserId(String userId);
    
    @Query("{ 'userId': ?0, 'dietaryTags': { $in: ?1 } }")
    List<Meal> findByUserIdAndDietaryTags(String userId, List<String> tags);
    
    @Query("{ 'userId': ?0, 'totalCalories': { $lte: ?1 } }")
    List<Meal> findLowCalorieMealsByUserId(String userId, Double maxCalories);
    
    @Query("{ 'userId': ?0, 'totalProtein': { $gte: ?1 } }")
    List<Meal> findHighProteinMealsByUserId(String userId, Double minProtein);
    
    @Aggregation(pipeline = {
        "{ $match: { 'userId': ?0, 'date': { $gte: ?1, $lte: ?2 } } }",
        "{ $group: { '_id': null, 'totalCalories': { $sum: '$totalCalories' }, 'totalProtein': { $sum: '$totalProtein' }, 'totalCarbs': { $sum: '$totalCarbs' }, 'totalFat': { $sum: '$totalFat' } } }"
    })
    Map<String, Double> calculateNutritionSummary(String userId, LocalDate startDate, LocalDate endDate);
    
    Long countByUserIdAndConsumed(String userId, boolean consumed);
    
    @Query("{ 'userId': ?0, 'date': ?1, 'consumed': false }")
    List<Meal> findPlannedMealsForToday(String userId, LocalDate date);
}