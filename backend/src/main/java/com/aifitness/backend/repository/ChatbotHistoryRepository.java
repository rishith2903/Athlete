package com.aifitness.backend.repository;

import com.aifitness.backend.entity.ChatbotHistory;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.data.mongodb.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface ChatbotHistoryRepository extends MongoRepository<ChatbotHistory, String> {
    
    Page<ChatbotHistory> findByUserId(String userId, Pageable pageable);
    
    Optional<ChatbotHistory> findBySessionId(String sessionId);
    
    List<ChatbotHistory> findByUserIdAndStatus(String userId, String status);
    
    @Query("{ 'userId': ?0, 'context': ?1 }")
    List<ChatbotHistory> findByUserIdAndContext(String userId, String context);
    
    @Query("{ 'userId': ?0, 'status': 'ACTIVE' }")
    Optional<ChatbotHistory> findActiveSessionByUserId(String userId);
    
    @Query("{ 'userId': ?0, 'createdAt': { $gte: ?1 } }")
    List<ChatbotHistory> findRecentSessionsByUserId(String userId, LocalDateTime since);
    
    Long countByUserIdAndStatus(String userId, String status);
}