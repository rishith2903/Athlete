package com.aifitness.backend.controller;

import com.aifitness.backend.service.StatsService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST Controller for workout statistics and progress analytics.
 */
@RestController
@RequestMapping("/api/stats")
@RequiredArgsConstructor
@Tag(name = "Statistics", description = "Workout statistics and progress analytics")
public class StatsController {

    private final StatsService statsService;

    @GetMapping("/dashboard")
    @Operation(summary = "Get dashboard summary stats")
    public ResponseEntity<Map<String, Object>> getDashboardStats(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        StatsService.DashboardStats stats = statsService.getDashboardStats(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", stats));
    }

    @GetMapping("/frequency")
    @Operation(summary = "Get workout frequency by week")
    public ResponseEntity<Map<String, Object>> getWorkoutFrequency(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestParam(defaultValue = "12") int weeks) {
        String userId = getUserId(userDetails);
        List<StatsService.WeeklyStats> stats = statsService.getWorkoutFrequency(userId, weeks);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", stats));
    }

    @GetMapping("/volume")
    @Operation(summary = "Get volume by muscle group over time")
    public ResponseEntity<Map<String, Object>> getVolumeByMuscle(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestParam(defaultValue = "30") int days) {
        String userId = getUserId(userDetails);
        Map<String, List<StatsService.VolumeDataPoint>> data = statsService.getVolumeByMuscle(userId, days);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", data));
    }

    @GetMapping("/strength/{exerciseId}")
    @Operation(summary = "Get strength progress for an exercise")
    public ResponseEntity<Map<String, Object>> getStrengthProgress(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable String exerciseId,
            @RequestParam(defaultValue = "90") int days) {
        String userId = getUserId(userDetails);
        List<StatsService.StrengthDataPoint> data = statsService.getStrengthProgress(userId, exerciseId, days);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", data));
    }

    @GetMapping("/calendar")
    @Operation(summary = "Get workout calendar for a month")
    public ResponseEntity<Map<String, Object>> getWorkoutCalendar(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestParam int year,
            @RequestParam int month) {
        String userId = getUserId(userDetails);
        List<StatsService.CalendarDay> calendar = statsService.getWorkoutCalendar(userId, year, month);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", calendar,
                "year", year,
                "month", month));
    }

    private String getUserId(UserDetails userDetails) {
        return userDetails.getUsername();
    }
}
