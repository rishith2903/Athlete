package com.aifitness.backend.controller;

import com.aifitness.backend.entity.Exercise;
import com.aifitness.backend.service.ExerciseService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST Controller for Exercise Library operations.
 */
@RestController
@RequestMapping("/api/exercises")
@RequiredArgsConstructor
@Tag(name = "Exercise Library", description = "Exercise database and search operations")
public class ExerciseController {

    private final ExerciseService exerciseService;

    @GetMapping
    @Operation(summary = "Get all exercises", description = "Returns all active exercises in the library")
    public ResponseEntity<Map<String, Object>> getAllExercises() {
        List<Exercise> exercises = exerciseService.getAllExercises();
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get exercise by ID")
    public ResponseEntity<Map<String, Object>> getExerciseById(@PathVariable String id) {
        return exerciseService.getExerciseById(id)
                .map(exercise -> ResponseEntity.ok(Map.<String, Object>of(
                        "success", true,
                        "data", exercise)))
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/search")
    @Operation(summary = "Search exercises by name")
    public ResponseEntity<Map<String, Object>> searchExercises(@RequestParam String q) {
        List<Exercise> exercises = exerciseService.searchExercises(q);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @GetMapping("/filter")
    @Operation(summary = "Filter exercises by criteria")
    public ResponseEntity<Map<String, Object>> filterExercises(
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String muscle,
            @RequestParam(required = false) String equipment,
            @RequestParam(required = false) String difficulty) {
        List<Exercise> exercises = exerciseService.filterExercises(category, muscle, equipment, difficulty);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @GetMapping("/category/{category}")
    @Operation(summary = "Get exercises by category")
    public ResponseEntity<Map<String, Object>> getByCategory(@PathVariable String category) {
        List<Exercise> exercises = exerciseService.getExercisesByCategory(category);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @GetMapping("/muscle/{muscle}")
    @Operation(summary = "Get exercises by muscle group")
    public ResponseEntity<Map<String, Object>> getByMuscle(@PathVariable String muscle) {
        List<Exercise> exercises = exerciseService.getExercisesByMuscle(muscle);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @GetMapping("/equipment/{equipment}")
    @Operation(summary = "Get exercises by equipment")
    public ResponseEntity<Map<String, Object>> getByEquipment(@PathVariable String equipment) {
        List<Exercise> exercises = exerciseService.getExercisesByEquipment(equipment);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @GetMapping("/compound")
    @Operation(summary = "Get compound exercises only")
    public ResponseEntity<Map<String, Object>> getCompoundExercises() {
        List<Exercise> exercises = exerciseService.getCompoundExercises();
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", exercises,
                "count", exercises.size()));
    }

    @PostMapping
    @Operation(summary = "Create a new exercise")
    public ResponseEntity<Map<String, Object>> createExercise(@RequestBody Exercise exercise) {
        Exercise created = exerciseService.createExercise(exercise);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", created));
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update an exercise")
    public ResponseEntity<Map<String, Object>> updateExercise(
            @PathVariable String id,
            @RequestBody Exercise exercise) {
        Exercise updated = exerciseService.updateExercise(id, exercise);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", updated));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete an exercise (soft delete)")
    public ResponseEntity<Map<String, Object>> deleteExercise(@PathVariable String id) {
        exerciseService.deleteExercise(id);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Exercise deleted successfully"));
    }
}
