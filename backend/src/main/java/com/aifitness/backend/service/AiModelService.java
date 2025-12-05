package com.aifitness.backend.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class AiModelService {
    
    private final WebClient.Builder webClientBuilder;
    private final ObjectMapper objectMapper;
    
    @Value("${ai.services.workout-recommender.url}")
    private String workoutServiceUrl;
    
    @Value("${ai.services.workout-recommender.timeout}")
    private Long workoutServiceTimeout;
    
    @Value("${ai.services.nutrition-planner.url}")
    private String nutritionServiceUrl;
    
    @Value("${ai.services.nutrition-planner.timeout}")
    private Long nutritionServiceTimeout;
    
    @Value("${ai.services.pose-checker.url}")
    private String poseServiceUrl;
    
    @Value("${ai.services.pose-checker.timeout}")
    private Long poseServiceTimeout;
    
    @Value("${ai.services.chatbot.url}")
    private String chatbotServiceUrl;
    
    @Value("${ai.services.chatbot.timeout}")
    private Long chatbotServiceTimeout;
    
    /**
     * Get AI-generated workout recommendations
     */
    public Mono<Map<String, Object>> getWorkoutRecommendation(Map<String, Object> userProfile) {
        WebClient webClient = webClientBuilder
                .baseUrl(workoutServiceUrl)
                .build();
        
        return webClient.post()
                .uri("/recommend")
                .contentType(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(userProfile))
                .retrieve()
                .bodyToMono(Map.class)
                .map(response -> (Map<String, Object>) response)
                .timeout(Duration.ofMillis(workoutServiceTimeout))
                .doOnSuccess(response -> log.info("Workout recommendation received"))
                .doOnError(error -> log.error("Error getting workout recommendation", error));
    }
    
    /**
     * Get AI-generated nutrition plan
     */
    public Mono<Map<String, Object>> getNutritionPlan(Map<String, Object> userProfile) {
        WebClient webClient = webClientBuilder
                .baseUrl(nutritionServiceUrl)
                .build();
        
        return webClient.post()
                .uri("/plan")
                .contentType(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(userProfile))
                .retrieve()
                .bodyToMono(Map.class)
                .map(response -> (Map<String, Object>) response)
                .timeout(Duration.ofMillis(nutritionServiceTimeout))
                .doOnSuccess(response -> log.info("Nutrition plan received"))
                .doOnError(error -> log.error("Error getting nutrition plan", error));
    }
    
    /**
     * Check exercise form from video/image
     */
    public Mono<Map<String, Object>> checkExerciseForm(MultipartFile file, String exerciseType) {
        WebClient webClient = webClientBuilder
                .baseUrl(poseServiceUrl)
                .build();
        
        try {
            MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
            bodyBuilder.part("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            });
            bodyBuilder.part("exercise_type", exerciseType);
            
            return webClient.post()
                    .uri("/analyze")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                    .retrieve()
                    .bodyToMono(Map.class)
                    .map(response -> (Map<String, Object>) response)
                    .timeout(Duration.ofMillis(poseServiceTimeout))
                    .doOnSuccess(response -> log.info("Form check completed for exercise: {}", exerciseType))
                    .doOnError(error -> log.error("Error checking exercise form", error));
        } catch (Exception e) {
            log.error("Error processing file for form check", e);
            return Mono.error(e);
        }
    }
    
    /**
     * Get chatbot response
     */
    public Mono<Map<String, Object>> getChatbotResponse(Map<String, Object> chatRequest) {
        WebClient webClient = webClientBuilder
                .baseUrl(chatbotServiceUrl)
                .build();
        
        return webClient.post()
                .uri("/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(chatRequest))
                .retrieve()
                .bodyToMono(Map.class)
                .map(response -> (Map<String, Object>) response)
                .timeout(Duration.ofMillis(chatbotServiceTimeout))
                .doOnSuccess(response -> log.info("Chatbot response received"))
                .doOnError(error -> log.error("Error getting chatbot response", error));
    }
    
    /**
     * Analyze progress and get insights
     */
    public Mono<Map<String, Object>> analyzeProgress(Map<String, Object> progressData) {
        WebClient webClient = webClientBuilder
                .baseUrl(workoutServiceUrl)
                .build();
        
        return webClient.post()
                .uri("/analyze-progress")
                .contentType(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(progressData))
                .retrieve()
                .bodyToMono(Map.class)
                .map(response -> (Map<String, Object>) response)
                .timeout(Duration.ofMillis(workoutServiceTimeout))
                .doOnSuccess(response -> log.info("Progress analysis completed"))
                .doOnError(error -> log.error("Error analyzing progress", error));
    }
    
    /**
     * Get personalized exercise recommendations based on user data
     */
    public Mono<Map<String, Object>> getPersonalizedExercises(Map<String, Object> userData) {
        WebClient webClient = webClientBuilder
                .baseUrl(workoutServiceUrl)
                .build();
        
        return webClient.post()
                .uri("/personalized-exercises")
                .contentType(MediaType.APPLICATION_JSON)
                .body(BodyInserters.fromValue(userData))
                .retrieve()
                .bodyToMono(Map.class)
                .map(response -> (Map<String, Object>) response)
                .timeout(Duration.ofMillis(workoutServiceTimeout))
                .doOnSuccess(response -> log.info("Personalized exercises received"))
                .doOnError(error -> log.error("Error getting personalized exercises", error));
    }
}