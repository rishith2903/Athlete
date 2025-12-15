package com.aifitness.backend.controller;

import com.aifitness.backend.entity.PersonalRecord;
import com.aifitness.backend.entity.WorkoutLog;
import com.aifitness.backend.service.WorkoutLogService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for Workout Logging operations.
 */
@RestController
@RequestMapping("/api/workout-logs")
@RequiredArgsConstructor
@Tag(name = "Workout Logging", description = "Log and track workout sessions")
public class WorkoutLogController {

    private final WorkoutLogService workoutLogService;

    @PostMapping
    @Operation(summary = "Log a completed workout")
    public ResponseEntity<Map<String, Object>> logWorkout(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestBody WorkoutLog workoutLog) {
        workoutLog.setUserId(getUserId(userDetails));
        if (workoutLog.getStartTime() == null) {
            workoutLog.setStartTime(LocalDateTime.now());
        }
        WorkoutLog saved = workoutLogService.logWorkout(workoutLog);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", saved,
                "message", "Workout logged successfully"));
    }

    @GetMapping
    @Operation(summary = "Get user's workout history")
    public ResponseEntity<Map<String, Object>> getWorkoutHistory(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        String userId = getUserId(userDetails);
        Page<WorkoutLog> workouts = workoutLogService.getUserWorkoutsPaged(
                userId,
                PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "startTime")));
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", workouts.getContent(),
                "totalPages", workouts.getTotalPages(),
                "totalElements", workouts.getTotalElements(),
                "currentPage", page));
    }

    @GetMapping("/recent")
    @Operation(summary = "Get recent workouts")
    public ResponseEntity<Map<String, Object>> getRecentWorkouts(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<WorkoutLog> workouts = workoutLogService.getRecentWorkouts(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", workouts));
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get workout by ID")
    public ResponseEntity<Map<String, Object>> getWorkoutById(@PathVariable String id) {
        return workoutLogService.getWorkoutById(id)
                .map(workout -> ResponseEntity.ok(Map.<String, Object>of(
                        "success", true,
                        "data", workout)))
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/range")
    @Operation(summary = "Get workouts in date range")
    public ResponseEntity<Map<String, Object>> getWorkoutsInRange(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestParam String startDate,
            @RequestParam String endDate) {
        String userId = getUserId(userDetails);
        LocalDateTime start = LocalDateTime.parse(startDate);
        LocalDateTime end = LocalDateTime.parse(endDate);
        List<WorkoutLog> workouts = workoutLogService.getWorkoutsInRange(userId, start, end);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", workouts,
                "count", workouts.size()));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update a workout log")
    public ResponseEntity<Map<String, Object>> updateWorkout(
            @PathVariable String id,
            @RequestBody WorkoutLog workoutLog) {
        WorkoutLog updated = workoutLogService.updateWorkout(id, workoutLog);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", updated));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a workout log")
    public ResponseEntity<Map<String, Object>> deleteWorkout(@PathVariable String id) {
        workoutLogService.deleteWorkout(id);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Workout deleted successfully"));
    }

    @GetMapping("/stats")
    @Operation(summary = "Get workout statistics")
    public ResponseEntity<Map<String, Object>> getWorkoutStats(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        long totalWorkouts = workoutLogService.getWorkoutCount(userId);

        // Get last 30 days stats
        LocalDateTime thirtyDaysAgo = LocalDateTime.now().minusDays(30);
        long recentWorkouts = workoutLogService.getWorkoutCountInRange(userId, thirtyDaysAgo, LocalDateTime.now());

        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", Map.of(
                        "totalWorkouts", totalWorkouts,
                        "workoutsLast30Days", recentWorkouts)));
    }

    // Personal Records endpoints
    @GetMapping("/prs")
    @Operation(summary = "Get all personal records")
    public ResponseEntity<Map<String, Object>> getPersonalRecords(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<PersonalRecord> prs = workoutLogService.getUserPRs(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", prs));
    }

    @GetMapping("/prs/recent")
    @Operation(summary = "Get recent personal records")
    public ResponseEntity<Map<String, Object>> getRecentPRs(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<PersonalRecord> prs = workoutLogService.getRecentPRs(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", prs));
    }

    @GetMapping("/prs/exercise/{exerciseId}")
    @Operation(summary = "Get PRs for a specific exercise")
    public ResponseEntity<Map<String, Object>> getExercisePRs(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable String exerciseId) {
        String userId = getUserId(userDetails);
        List<PersonalRecord> prs = workoutLogService.getExercisePRs(userId, exerciseId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", prs));
    }

    private String getUserId(UserDetails userDetails) {
        // Extract user ID from the authenticated user
        // This assumes userDetails has username as ID or you can cast to User entity
        return userDetails.getUsername();
    }
}
