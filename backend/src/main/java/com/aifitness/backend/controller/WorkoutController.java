package com.aifitness.backend.controller;

import com.aifitness.backend.entity.User;
import com.aifitness.backend.entity.Workout;
import com.aifitness.backend.service.AiModelService;
import com.aifitness.backend.service.WorkoutService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import jakarta.validation.Valid;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/workout")
@RequiredArgsConstructor
@SecurityRequirement(name = "bearerAuth")
@Tag(name = "Workout Management", description = "Endpoints for managing workouts and exercise plans")
public class WorkoutController {
    
    private final WorkoutService workoutService;
    private final AiModelService aiModelService;
    
    @GetMapping
    @Operation(summary = "Get user workouts", description = "Retrieve paginated list of user's workouts")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Workouts retrieved successfully"),
        @ApiResponse(responseCode = "401", description = "User not authenticated")
    })
    public ResponseEntity<Page<Workout>> getUserWorkouts(
            @AuthenticationPrincipal User user,
            Pageable pageable) {
        Page<Workout> workouts = workoutService.getUserWorkouts(user.getId(), pageable);
        return ResponseEntity.ok(workouts);
    }
    
    @GetMapping("/{id}")
    @Operation(summary = "Get workout by ID", description = "Retrieve specific workout details")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Workout found"),
        @ApiResponse(responseCode = "404", description = "Workout not found"),
        @ApiResponse(responseCode = "403", description = "Access denied")
    })
    public ResponseEntity<Workout> getWorkout(
            @PathVariable String id,
            @AuthenticationPrincipal User user) {
        Workout workout = workoutService.getWorkout(id, user.getId());
        return ResponseEntity.ok(workout);
    }
    
    @PostMapping
    @Operation(summary = "Create workout", description = "Create a new workout plan")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "201", description = "Workout created successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid input"),
        @ApiResponse(responseCode = "401", description = "User not authenticated")
    })
    public ResponseEntity<Workout> createWorkout(
            @Valid @RequestBody Workout workout,
            @AuthenticationPrincipal User user) {
        workout.setUserId(user.getId());
        Workout created = workoutService.createWorkout(workout);
        return ResponseEntity.status(HttpStatus.CREATED).body(created);
    }
    
    @PutMapping("/{id}")
    @Operation(summary = "Update workout", description = "Update existing workout details")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Workout updated successfully"),
        @ApiResponse(responseCode = "404", description = "Workout not found"),
        @ApiResponse(responseCode = "403", description = "Access denied")
    })
    public ResponseEntity<Workout> updateWorkout(
            @PathVariable String id,
            @Valid @RequestBody Workout workout,
            @AuthenticationPrincipal User user) {
        Workout updated = workoutService.updateWorkout(id, workout, user.getId());
        return ResponseEntity.ok(updated);
    }
    
    @DeleteMapping("/{id}")
    @Operation(summary = "Delete workout", description = "Delete a workout")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "204", description = "Workout deleted successfully"),
        @ApiResponse(responseCode = "404", description = "Workout not found"),
        @ApiResponse(responseCode = "403", description = "Access denied")
    })
    public ResponseEntity<Void> deleteWorkout(
            @PathVariable String id,
            @AuthenticationPrincipal User user) {
        workoutService.deleteWorkout(id, user.getId());
        return ResponseEntity.noContent().build();
    }
    
    @PostMapping("/ai-generate")
    @Operation(summary = "Generate AI workout", description = "Generate personalized workout plan using AI")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "AI workout generated successfully"),
        @ApiResponse(responseCode = "400", description = "Invalid request"),
        @ApiResponse(responseCode = "500", description = "AI service error")
    })
    public Mono<ResponseEntity<Map<String, Object>>> generateAiWorkout(
            @AuthenticationPrincipal User user,
            @RequestBody Map<String, Object> preferences) {
        
        Map<String, Object> userProfile = Map.of(
            "userId", user.getId(),
            "fitnessGoal", user.getFitnessGoal(),
            "activityLevel", user.getActivityLevel(),
            "equipment", user.getEquipmentAvailable(),
            "preferredExercises", user.getPreferredExercises(),
            "workoutDuration", user.getWorkoutDuration(),
            "preferences", preferences
        );
        
        return aiModelService.getWorkoutRecommendation(userProfile)
                .map(ResponseEntity::ok);
    }
    
    @GetMapping("/scheduled")
    @Operation(summary = "Get scheduled workouts", description = "Get workouts scheduled for a specific date range")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Scheduled workouts retrieved"),
        @ApiResponse(responseCode = "401", description = "User not authenticated")
    })
    public ResponseEntity<List<Workout>> getScheduledWorkouts(
            @AuthenticationPrincipal User user,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        List<Workout> workouts = workoutService.getScheduledWorkouts(user.getId(), startDate, endDate);
        return ResponseEntity.ok(workouts);
    }
    
    @PostMapping("/{id}/start")
    @Operation(summary = "Start workout", description = "Mark workout as started")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Workout started"),
        @ApiResponse(responseCode = "404", description = "Workout not found"),
        @ApiResponse(responseCode = "403", description = "Access denied")
    })
    public ResponseEntity<Workout> startWorkout(
            @PathVariable String id,
            @AuthenticationPrincipal User user) {
        Workout workout = workoutService.startWorkout(id, user.getId());
        return ResponseEntity.ok(workout);
    }
    
    @PostMapping("/{id}/complete")
    @Operation(summary = "Complete workout", description = "Mark workout as completed")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Workout completed"),
        @ApiResponse(responseCode = "404", description = "Workout not found"),
        @ApiResponse(responseCode = "403", description = "Access denied")
    })
    public ResponseEntity<Workout> completeWorkout(
            @PathVariable String id,
            @RequestBody Map<String, Object> completionData,
            @AuthenticationPrincipal User user) {
        Workout workout = workoutService.completeWorkout(id, user.getId(), completionData);
        return ResponseEntity.ok(workout);
    }
    
    @GetMapping("/statistics")
    @Operation(summary = "Get workout statistics", description = "Get user's workout statistics and analytics")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Statistics retrieved"),
        @ApiResponse(responseCode = "401", description = "User not authenticated")
    })
    public ResponseEntity<Map<String, Object>> getWorkoutStatistics(
            @AuthenticationPrincipal User user,
            @RequestParam(defaultValue = "30") Integer days) {
        Map<String, Object> stats = workoutService.getWorkoutStatistics(user.getId(), days);
        return ResponseEntity.ok(stats);
    }
}