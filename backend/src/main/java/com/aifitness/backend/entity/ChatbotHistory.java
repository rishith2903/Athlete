package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "chatbot_history")
public class ChatbotHistory {
    
    @Id
    private String id;
    
    @Indexed
    private String userId;
    
    private String sessionId;
    
    @Builder.Default
    private List<Message> messages = new ArrayList<>();
    
    private String context; // WORKOUT, NUTRITION, GENERAL, FORM_CHECK, PROGRESS
    private String status; // ACTIVE, COMPLETED, ARCHIVED
    
    private Map<String, Object> metadata;
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    private LocalDateTime lastMessageAt;
    
    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Message {
        private String role; // USER, ASSISTANT, SYSTEM
        private String content;
        private LocalDateTime timestamp;
        private String messageType; // TEXT, IMAGE, VIDEO, AUDIO
        private List<String> attachmentUrls;
        private Map<String, Object> metadata;
        private boolean edited;
        private String editedContent;
        private LocalDateTime editedAt;
    }
}