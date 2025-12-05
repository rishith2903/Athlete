package com.aifitness.backend.repository;

import com.aifitness.backend.entity.Progress;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

@Repository
public interface ProgressRepository extends MongoRepository<Progress, String> {
    
    Page<Progress> findByUserId(String userId, Pageable pageable);
    
    Optional<Progress> findByUserIdAndDate(String userId, LocalDate date);
    
    List<Progress> findByUserIdAndDateBetweenOrderByDateDesc(String userId, LocalDate startDate, LocalDate endDate);
    
    @Query("{ 'userId': ?0 }")
    Optional<Progress> findLatestProgressByUserId(String userId);
    
    @Query("{ 'userId': ?0, 'weight': { $exists: true } }")
    List<Progress> findWeightProgressByUserId(String userId);
    
    @Query("{ 'userId': ?0, 'personalRecords': { $exists: true, $ne: {} } }")
    List<Progress> findProgressWithPersonalRecords(String userId);
    
    @Query("{ 'userId': ?0, 'goalCompletionPercentage': { $gte: ?1 } }")
    List<Progress> findProgressAboveGoalPercentage(String userId, Double percentage);
    
    @Query("{ 'userId': ?0 }")
    List<Progress> findAllByUserIdOrderByDateDesc(String userId);
    
    boolean existsByUserIdAndDate(String userId, LocalDate date);
}