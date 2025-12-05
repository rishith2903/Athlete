package com.aifitness.backend.controller;

import com.aifitness.backend.entity.ChatbotHistory;
import com.aifitness.backend.entity.User;
import com.aifitness.backend.service.AiModelService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import java.time.LocalDateTime;
import java.util.*;

@Slf4j
@RestController
@RequestMapping("/api/chatbot")
@RequiredArgsConstructor
@SecurityRequirement(name = "bearerAuth")
@Tag(name = "Chatbot", description = "AI Chatbot interaction endpoints")
public class ChatbotController {
    
    private final AiModelService aiModelService;
    
    public static class ChatRequest {
        @NotBlank(message = "Message cannot be empty")
        public String message;
        public Optional<String> context = Optional.empty();
        public Optional<String> sessionId = Optional.empty();
    }
    
    public static class ChatResponse {
        public String response;
        public String intent;
        public double confidence;
        public String sessionId;
        public LocalDateTime timestamp;
        public Map<String, Object> metadata;
    }
    
    @PostMapping
    @Operation(summary = "Send message to chatbot", description = "Get AI-powered fitness advice and answers")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Chatbot response generated successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid message or request"),
        @ApiResponse(responseCode = "500", description = "AI service error"),
        @ApiResponse(responseCode = "503", description = "Chatbot service unavailable")
    })
    public Mono<ResponseEntity<Map<String, Object>>> sendMessage(
            @Valid @RequestBody ChatRequest request,
            @AuthenticationPrincipal User user) {
        
        log.info("Chatbot request from user {}: {}", user.getId(), request.message);
        
        // Prepare chatbot request
        Map<String, Object> chatbotRequest = new HashMap<>();
        chatbotRequest.put("userId", user.getId());
        chatbotRequest.put("message", request.message);
        chatbotRequest.put("context", request.context.orElse("general"));
        chatbotRequest.put("sessionId", request.sessionId.orElse(UUID.randomUUID().toString()));
        chatbotRequest.put("userProfile", Map.of(
            "fitnessGoal", user.getFitnessGoal(),
            "activityLevel", user.getActivityLevel(),
            "age", calculateAge(user.getDateOfBirth()),
            "weight", user.getWeight(),
            "height", user.getHeight()
        ));
        
        return aiModelService.getChatbotResponse(chatbotRequest)
                .map(response -> {
                    // Ensure response has required fields
                    response.putIfAbsent("response", "I'm here to help with your fitness journey!");
                    response.putIfAbsent("intent", "general");
                    response.putIfAbsent("confidence", 0.8);
                    response.put("sessionId", chatbotRequest.get("sessionId"));
                    response.put("timestamp", LocalDateTime.now().toString());
                    
                    log.info("Chatbot response generated for user {}", user.getId());
                    return ResponseEntity.ok(response);
                })
                .onErrorResume(throwable -> {
                    log.error("Error getting chatbot response", throwable);
                    
                    Map<String, Object> errorResponse = new HashMap<>();
                    errorResponse.put("error", "Failed to get chatbot response");
                    errorResponse.put("message", throwable.getMessage());
                    errorResponse.put("status", "SERVICE_ERROR");
                    
                    // Provide fallback response
                    errorResponse.put("fallbackResponse", generateFallbackResponse(request.message));
                    
                    if (throwable.getMessage() != null && throwable.getMessage().contains("timeout")) {
                        return Mono.just(ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT).body(errorResponse));
                    }
                    
                    return Mono.just(ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(errorResponse));
                });
    }
    
    @GetMapping("/suggestions")
    @Operation(summary = "Get conversation suggestions", description = "Get suggested questions or topics")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Suggestions retrieved successfully")
    })
    public ResponseEntity<Map<String, Object>> getSuggestions(@AuthenticationPrincipal User user) {
        List<String> suggestions = Arrays.asList(
            "How can I lose weight effectively?",
            "What's a good workout routine for beginners?",
            "How much protein should I eat daily?",
            "What exercises target core muscles?",
            "How can I improve my squat form?",
            "What should I eat before a workout?",
            "How many rest days do I need?",
            "What's the best way to build muscle?"
        );
        
        return ResponseEntity.ok(Map.of(
            "suggestions", suggestions,
            "categories", Arrays.asList("Workout", "Nutrition", "Form", "Recovery")
        ));
    }
    
    private String generateFallbackResponse(String message) {
        String lowerMessage = message.toLowerCase();
        
        if (lowerMessage.contains("workout")) {
            return "For workout advice, consider starting with compound exercises like squats, push-ups, and planks. Would you like a personalized workout plan?";
        } else if (lowerMessage.contains("diet") || lowerMessage.contains("nutrition")) {
            return "Nutrition is key to fitness success. Focus on balanced meals with adequate protein, complex carbs, and healthy fats. Would you like a meal plan?";
        } else if (lowerMessage.contains("weight") || lowerMessage.contains("lose")) {
            return "Weight loss requires a caloric deficit combined with regular exercise. Aim for 1-2 pounds per week for sustainable results.";
        } else if (lowerMessage.contains("muscle") || lowerMessage.contains("gain")) {
            return "Building muscle requires progressive overload, adequate protein intake (1.6-2.2g/kg body weight), and sufficient rest.";
        } else {
            return "I'm here to help with your fitness journey! You can ask me about workouts, nutrition, form checks, or general fitness advice.";
        }
    }
    
    private int calculateAge(java.time.LocalDate dateOfBirth) {
        if (dateOfBirth == null) return 25; // Default age
        return java.time.Period.between(dateOfBirth, java.time.LocalDate.now()).getYears();
    }
}