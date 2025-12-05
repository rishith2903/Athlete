package com.aifitness.backend.controller;

import com.aifitness.backend.entity.Meal;
import com.aifitness.backend.entity.User;
import com.aifitness.backend.service.AiModelService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import jakarta.validation.Valid;
import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/nutrition")
@RequiredArgsConstructor
@SecurityRequirement(name = "bearerAuth")
@Tag(name = "Nutrition Management", description = "Endpoints for nutrition and meal planning")
public class NutritionController {
    
    private final AiModelService aiModelService;
    
    @PostMapping("/ai-plan")
    @Operation(summary = "Generate AI nutrition plan", description = "Generate personalized nutrition plan using AI")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Nutrition plan generated successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid request"),
        @ApiResponse(responseCode = "500", description = "AI service error"),
        @ApiResponse(responseCode = "503", description = "AI service unavailable")
    })
    public Mono<ResponseEntity<Map<String, Object>>> generateNutritionPlan(
            @AuthenticationPrincipal User user,
            @RequestBody Map<String, Object> preferences) {
        
        Map<String, Object> userProfile = new HashMap<>();
        userProfile.put("userId", user.getId());
        userProfile.put("weight", user.getWeight());
        userProfile.put("height", user.getHeight());
        userProfile.put("activityLevel", user.getActivityLevel());
        userProfile.put("fitnessGoal", user.getFitnessGoal());
        userProfile.put("targetWeight", user.getTargetWeight());
        userProfile.put("allergies", user.getAllergies());
        userProfile.put("dietaryRestrictions", user.getDietaryRestrictions());
        userProfile.put("preferences", preferences);
        
        return aiModelService.getNutritionPlan(userProfile)
                .map(response -> {
                    // Add metadata
                    response.put("generated", true);
                    response.put("userId", user.getId());
                    return ResponseEntity.ok(response);
                })
                .onErrorResume(throwable -> {
                    Map<String, Object> errorResponse = new HashMap<>();
                    errorResponse.put("error", "Failed to generate nutrition plan");
                    errorResponse.put("message", throwable.getMessage());
                    errorResponse.put("status", "SERVICE_ERROR");
                    
                    if (throwable.getMessage() != null && throwable.getMessage().contains("timeout")) {
                        return Mono.just(ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT).body(errorResponse));
                    }
                    return Mono.just(ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(errorResponse));
                });
    }
}