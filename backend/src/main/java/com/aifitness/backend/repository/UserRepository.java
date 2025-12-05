package com.aifitness.backend.repository;

import com.aifitness.backend.entity.User;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface UserRepository extends MongoRepository<User, String> {
    
    Optional<User> findByEmail(String email);
    
    Optional<User> findByUsername(String username);
    
    Optional<User> findByEmailOrUsername(String email, String username);
    
    boolean existsByEmail(String email);
    
    boolean existsByUsername(String username);
    
    Optional<User> findByRefreshToken(String refreshToken);
    
    List<User> findBySubscriptionPlan(String subscriptionPlan);
    
    @Query("{ 'subscriptionExpiryDate': { $lte: ?0 } }")
    List<User> findUsersWithExpiredSubscriptions(LocalDateTime date);
    
    @Query("{ 'roles': { $in: ?0 } }")
    List<User> findByRole(String role);
    
    @Query("{ 'enabled': true, 'accountNonLocked': true }")
    List<User> findActiveUsers();
    
    @Query("{ 'lastLoginAt': { $gte: ?0 } }")
    List<User> findRecentlyActiveUsers(LocalDateTime since);
    
    @Query("{ 'fitnessGoal': ?0 }")
    List<User> findByFitnessGoal(String fitnessGoal);
    
    @Query("{ 'medicalConditions': { $in: ?0 } }")
    List<User> findUsersWithMedicalConditions(List<String> conditions);
    
    @Query("{ 'workoutReminders': true }")
    List<User> findUsersWithWorkoutRemindersEnabled();
    
    @Query("{ 'mealReminders': true }")
    List<User> findUsersWithMealRemindersEnabled();
}