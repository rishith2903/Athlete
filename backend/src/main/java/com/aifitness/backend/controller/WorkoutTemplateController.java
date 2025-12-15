package com.aifitness.backend.controller;

import com.aifitness.backend.entity.WorkoutLog;
import com.aifitness.backend.entity.WorkoutTemplate;
import com.aifitness.backend.service.WorkoutTemplateService;
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
 * REST Controller for Workout Templates.
 */
@RestController
@RequestMapping("/api/templates")
@RequiredArgsConstructor
@Tag(name = "Workout Templates", description = "Create and manage workout templates")
public class WorkoutTemplateController {

    private final WorkoutTemplateService templateService;

    @PostMapping
    @Operation(summary = "Create a new template")
    public ResponseEntity<Map<String, Object>> createTemplate(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestBody WorkoutTemplate template) {
        template.setUserId(getUserId(userDetails));
        WorkoutTemplate created = templateService.createTemplate(template);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", created,
                "message", "Template created successfully"));
    }

    @GetMapping
    @Operation(summary = "Get user's templates")
    public ResponseEntity<Map<String, Object>> getTemplates(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<WorkoutTemplate> templates = templateService.getUserTemplates(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", templates,
                "count", templates.size()));
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get template by ID")
    public ResponseEntity<Map<String, Object>> getTemplate(@PathVariable String id) {
        return templateService.getTemplateById(id)
                .map(template -> ResponseEntity.ok(Map.<String, Object>of(
                        "success", true,
                        "data", template)))
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/category/{category}")
    @Operation(summary = "Get templates by category")
    public ResponseEntity<Map<String, Object>> getByCategory(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable String category) {
        String userId = getUserId(userDetails);
        List<WorkoutTemplate> templates = templateService.getTemplatesByCategory(userId, category);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", templates));
    }

    @GetMapping("/public")
    @Operation(summary = "Get popular public templates")
    public ResponseEntity<Map<String, Object>> getPublicTemplates() {
        List<WorkoutTemplate> templates = templateService.getPublicTemplates();
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", templates));
    }

    @GetMapping("/popular")
    @Operation(summary = "Get user's most used templates")
    public ResponseEntity<Map<String, Object>> getPopularTemplates(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<WorkoutTemplate> templates = templateService.getMostUsedTemplates(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", templates));
    }

    @PostMapping("/{id}/start")
    @Operation(summary = "Start a workout from a template")
    public ResponseEntity<Map<String, Object>> startWorkout(
            @AuthenticationPrincipal UserDetails userDetails,
            @PathVariable String id) {
        String userId = getUserId(userDetails);
        WorkoutLog workoutLog = templateService.startWorkoutFromTemplate(id, userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", workoutLog,
                "message", "Workout started from template"));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update a template")
    public ResponseEntity<Map<String, Object>> updateTemplate(
            @PathVariable String id,
            @RequestBody WorkoutTemplate template) {
        WorkoutTemplate updated = templateService.updateTemplate(id, template);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", updated));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a template")
    public ResponseEntity<Map<String, Object>> deleteTemplate(@PathVariable String id) {
        templateService.deleteTemplate(id);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Template deleted successfully"));
    }

    private String getUserId(UserDetails userDetails) {
        return userDetails.getUsername();
    }
}
