package com.aifitness.backend.service;

import com.aifitness.backend.entity.Exercise;
import com.aifitness.backend.repository.ExerciseRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

/**
 * Service for managing exercises in the exercise library.
 */
@Service
@RequiredArgsConstructor
public class ExerciseService {

    private final ExerciseRepository exerciseRepository;

    /**
     * Get all active exercises
     */
    public List<Exercise> getAllExercises() {
        return exerciseRepository.findByIsActiveTrue();
    }

    /**
     * Get exercise by ID
     */
    public Optional<Exercise> getExerciseById(String id) {
        return exerciseRepository.findById(id);
    }

    /**
     * Search exercises by name
     */
    public List<Exercise> searchExercises(String query) {
        return exerciseRepository.findByNameContainingIgnoreCase(query);
    }

    /**
     * Get exercises by category
     */
    public List<Exercise> getExercisesByCategory(String category) {
        return exerciseRepository.findByCategoryIgnoreCase(category);
    }

    /**
     * Get exercises by primary muscle
     */
    public List<Exercise> getExercisesByMuscle(String muscle) {
        return exerciseRepository.findByPrimaryMusclesContaining(muscle.toLowerCase());
    }

    /**
     * Get exercises by equipment
     */
    public List<Exercise> getExercisesByEquipment(String equipment) {
        return exerciseRepository.findByEquipmentContaining(equipment.toLowerCase());
    }

    /**
     * Get exercises by difficulty
     */
    public List<Exercise> getExercisesByDifficulty(String difficulty) {
        return exerciseRepository.findByDifficultyIgnoreCase(difficulty);
    }

    /**
     * Get compound exercises
     */
    public List<Exercise> getCompoundExercises() {
        return exerciseRepository.findByIsCompoundTrue();
    }

    /**
     * Filter exercises by multiple criteria
     */
    public List<Exercise> filterExercises(String category, String muscle, String equipment, String difficulty) {
        List<Exercise> exercises = exerciseRepository.findByIsActiveTrue();

        if (category != null && !category.isEmpty()) {
            exercises = exercises.stream()
                    .filter(e -> e.getCategory().equalsIgnoreCase(category))
                    .toList();
        }

        if (muscle != null && !muscle.isEmpty()) {
            exercises = exercises.stream()
                    .filter(e -> e.getPrimaryMuscles().stream()
                            .anyMatch(m -> m.equalsIgnoreCase(muscle)))
                    .toList();
        }

        if (equipment != null && !equipment.isEmpty()) {
            exercises = exercises.stream()
                    .filter(e -> e.getEquipment().stream()
                            .anyMatch(eq -> eq.equalsIgnoreCase(equipment)))
                    .toList();
        }

        if (difficulty != null && !difficulty.isEmpty()) {
            exercises = exercises.stream()
                    .filter(e -> e.getDifficulty().equalsIgnoreCase(difficulty))
                    .toList();
        }

        return exercises;
    }

    /**
     * Create a new exercise
     */
    public Exercise createExercise(Exercise exercise) {
        exercise.setActive(true);
        return exerciseRepository.save(exercise);
    }

    /**
     * Update an exercise
     */
    public Exercise updateExercise(String id, Exercise exercise) {
        exercise.setId(id);
        return exerciseRepository.save(exercise);
    }

    /**
     * Delete an exercise (soft delete)
     */
    public void deleteExercise(String id) {
        exerciseRepository.findById(id).ifPresent(exercise -> {
            exercise.setActive(false);
            exerciseRepository.save(exercise);
        });
    }

    /**
     * Get exercise count by category
     */
    public long getExerciseCountByCategory(String category) {
        return exerciseRepository.countByCategory(category);
    }

    /**
     * Seed exercises from dataset
     */
    public void seedExercises(List<Exercise> exercises) {
        exerciseRepository.saveAll(exercises);
    }
}
