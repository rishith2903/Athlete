package com.aifitness.backend.service;

import com.aifitness.backend.entity.WorkoutLog;
import com.aifitness.backend.entity.WorkoutTemplate;
import com.aifitness.backend.repository.WorkoutTemplateRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Service for managing workout templates.
 */
@Service
@RequiredArgsConstructor
public class WorkoutTemplateService {

    private final WorkoutTemplateRepository templateRepository;

    /**
     * Create a new template
     */
    public WorkoutTemplate createTemplate(WorkoutTemplate template) {
        template.setActive(true);
        template.setTimesUsed(0);
        return templateRepository.save(template);
    }

    /**
     * Get template by ID
     */
    public Optional<WorkoutTemplate> getTemplateById(String id) {
        return templateRepository.findById(id);
    }

    /**
     * Get all templates for a user
     */
    public List<WorkoutTemplate> getUserTemplates(String userId) {
        return templateRepository.findByUserIdAndIsActiveTrueOrderByCreatedAtDesc(userId);
    }

    /**
     * Get templates by category
     */
    public List<WorkoutTemplate> getTemplatesByCategory(String userId, String category) {
        return templateRepository.findByUserIdAndCategoryAndIsActiveTrue(userId, category);
    }

    /**
     * Get popular public templates
     */
    public List<WorkoutTemplate> getPublicTemplates() {
        return templateRepository.findByIsPublicTrueAndIsActiveTrueOrderByTimesUsedDesc();
    }

    /**
     * Get most used templates
     */
    public List<WorkoutTemplate> getMostUsedTemplates(String userId) {
        return templateRepository.findTop5ByUserIdAndIsActiveTrueOrderByTimesUsedDesc(userId);
    }

    /**
     * Update a template
     */
    public WorkoutTemplate updateTemplate(String id, WorkoutTemplate template) {
        template.setId(id);
        return templateRepository.save(template);
    }

    /**
     * Delete a template (soft delete)
     */
    public void deleteTemplate(String id) {
        templateRepository.findById(id).ifPresent(template -> {
            template.setActive(false);
            templateRepository.save(template);
        });
    }

    /**
     * Start a workout from a template
     * Returns a WorkoutLog pre-populated with template exercises
     */
    public WorkoutLog startWorkoutFromTemplate(String templateId, String userId) {
        Optional<WorkoutTemplate> templateOpt = templateRepository.findById(templateId);
        if (templateOpt.isEmpty()) {
            throw new RuntimeException("Template not found");
        }

        WorkoutTemplate template = templateOpt.get();

        // Increment times used
        template.setTimesUsed(template.getTimesUsed() + 1);
        template.setLastUsedDate(LocalDateTime.now().toString());
        templateRepository.save(template);

        // Create workout log from template
        WorkoutLog workoutLog = WorkoutLog.builder()
                .userId(userId)
                .name(template.getName())
                .templateId(templateId)
                .startTime(LocalDateTime.now())
                .exercises(template.getExercises().stream()
                        .map(te -> WorkoutLog.ExerciseLog.builder()
                                .exerciseId(te.getExerciseId())
                                .exerciseName(te.getExerciseName())
                                .order(te.getOrder())
                                .supersetGroup(te.getSupersetGroup())
                                .notes(te.getNotes())
                                .sets(createEmptySets(te.getTargetSets()))
                                .build())
                        .collect(Collectors.toList()))
                .build();

        return workoutLog;
    }

    /**
     * Create a template from a completed workout
     */
    public WorkoutTemplate createTemplateFromWorkout(WorkoutLog workoutLog, String name, String userId) {
        WorkoutTemplate template = WorkoutTemplate.builder()
                .userId(userId)
                .name(name)
                .exercises(workoutLog.getExercises().stream()
                        .map(el -> WorkoutTemplate.TemplateExercise.builder()
                                .exerciseId(el.getExerciseId())
                                .exerciseName(el.getExerciseName())
                                .order(el.getOrder())
                                .targetSets(el.getSets().size())
                                .supersetGroup(el.getSupersetGroup())
                                .notes(el.getNotes())
                                .build())
                        .collect(Collectors.toList()))
                .estimatedDuration(workoutLog.getDurationMinutes())
                .isActive(true)
                .timesUsed(0)
                .build();

        return templateRepository.save(template);
    }

    private List<WorkoutLog.SetLog> createEmptySets(Integer count) {
        if (count == null || count <= 0)
            count = 3;
        return java.util.stream.IntStream.rangeClosed(1, count)
                .mapToObj(i -> WorkoutLog.SetLog.builder()
                        .setNumber(i)
                        .completed(false)
                        .build())
                .collect(Collectors.toList());
    }

    /**
     * Count templates for a user
     */
    public long getTemplateCount(String userId) {
        return templateRepository.countByUserIdAndIsActiveTrue(userId);
    }
}
