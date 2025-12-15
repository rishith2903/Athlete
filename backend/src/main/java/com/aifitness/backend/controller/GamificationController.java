package com.aifitness.backend.controller;

import com.aifitness.backend.entity.Achievement;
import com.aifitness.backend.entity.UserStats;
import com.aifitness.backend.service.GamificationService;
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
 * REST Controller for gamification features.
 */
@RestController
@RequestMapping("/api/gamification")
@RequiredArgsConstructor
@Tag(name = "Gamification", description = "Achievements, XP, levels, and leaderboards")
public class GamificationController {

    private final GamificationService gamificationService;

    @GetMapping("/stats")
    @Operation(summary = "Get user's gamification stats")
    public ResponseEntity<Map<String, Object>> getUserStats(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        UserStats stats = gamificationService.getOrCreateUserStats(userId);

        int xpForNext = UserStats.xpForNextLevel(stats.getLevel());
        int currentLevelXp = stats.getTotalXp() - calculateXpForLevel(stats.getLevel());

        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", stats,
                "xpProgress", Map.of(
                        "current", currentLevelXp,
                        "required", xpForNext,
                        "percentage", (currentLevelXp * 100) / xpForNext)));
    }

    @GetMapping("/achievements")
    @Operation(summary = "Get user's achievements")
    public ResponseEntity<Map<String, Object>> getAchievements(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<Achievement> achievements = gamificationService.getUserAchievements(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", achievements,
                "count", achievements.size()));
    }

    @GetMapping("/achievements/recent")
    @Operation(summary = "Get recent achievements")
    public ResponseEntity<Map<String, Object>> getRecentAchievements(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<Achievement> achievements = gamificationService.getRecentAchievements(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", achievements));
    }

    @GetMapping("/leaderboard")
    @Operation(summary = "Get leaderboard")
    public ResponseEntity<Map<String, Object>> getLeaderboard(
            @RequestParam(defaultValue = "xp") String type,
            @RequestParam(defaultValue = "50") int limit) {
        List<UserStats> leaderboard = gamificationService.getLeaderboard(type, limit);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", leaderboard,
                "type", type));
    }

    private String getUserId(UserDetails userDetails) {
        return userDetails.getUsername();
    }

    private int calculateXpForLevel(int level) {
        int total = 0;
        int threshold = 100;
        for (int i = 1; i < level; i++) {
            total += threshold;
            threshold = (int) (threshold * 1.5);
        }
        return total;
    }
}
