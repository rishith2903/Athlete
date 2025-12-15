package com.aifitness.backend.service;

import com.aifitness.backend.entity.PersonalRecord;
import com.aifitness.backend.entity.WorkoutLog;
import com.aifitness.backend.repository.PersonalRecordRepository;
import com.aifitness.backend.repository.WorkoutLogRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Service for managing workout logs and tracking progress.
 */
@Service
@RequiredArgsConstructor
public class WorkoutLogService {

    private final WorkoutLogRepository workoutLogRepository;
    private final PersonalRecordRepository personalRecordRepository;

    /**
     * Log a completed workout
     */
    public WorkoutLog logWorkout(WorkoutLog workoutLog) {
        // Calculate totals
        int totalSets = 0;
        int totalReps = 0;
        double totalVolume = 0;

        for (WorkoutLog.ExerciseLog exercise : workoutLog.getExercises()) {
            for (WorkoutLog.SetLog set : exercise.getSets()) {
                if (set.isCompleted() && !set.isWarmup()) {
                    totalSets++;
                    if (set.getReps() != null) {
                        totalReps += set.getReps();
                        if (set.getWeight() != null) {
                            totalVolume += set.getWeight() * set.getReps();
                        }
                    }
                }
            }
        }

        workoutLog.setTotalSets(totalSets);
        workoutLog.setTotalReps(totalReps);
        workoutLog.setTotalVolume(totalVolume);

        // Calculate duration if not set
        if (workoutLog.getDurationMinutes() == null && workoutLog.getStartTime() != null
                && workoutLog.getEndTime() != null) {
            long minutes = java.time.Duration.between(workoutLog.getStartTime(), workoutLog.getEndTime()).toMinutes();
            workoutLog.setDurationMinutes((int) minutes);
        }

        WorkoutLog savedLog = workoutLogRepository.save(workoutLog);

        // Check for PRs
        checkAndUpdatePRs(savedLog);

        return savedLog;
    }

    /**
     * Get workout by ID
     */
    public Optional<WorkoutLog> getWorkoutById(String id) {
        return workoutLogRepository.findById(id);
    }

    /**
     * Get user's workout history
     */
    public List<WorkoutLog> getUserWorkouts(String userId) {
        return workoutLogRepository.findByUserIdOrderByStartTimeDesc(userId);
    }

    /**
     * Get user's workout history with pagination
     */
    public Page<WorkoutLog> getUserWorkoutsPaged(String userId, Pageable pageable) {
        return workoutLogRepository.findByUserId(userId, pageable);
    }

    /**
     * Get recent workouts
     */
    public List<WorkoutLog> getRecentWorkouts(String userId) {
        return workoutLogRepository.findTop10ByUserIdOrderByStartTimeDesc(userId);
    }

    /**
     * Get workouts in date range
     */
    public List<WorkoutLog> getWorkoutsInRange(String userId, LocalDateTime start, LocalDateTime end) {
        return workoutLogRepository.findByUserIdAndStartTimeBetween(userId, start, end);
    }

    /**
     * Update a workout log
     */
    public WorkoutLog updateWorkout(String id, WorkoutLog workoutLog) {
        workoutLog.setId(id);
        return workoutLogRepository.save(workoutLog);
    }

    /**
     * Delete a workout log
     */
    public void deleteWorkout(String id) {
        workoutLogRepository.deleteById(id);
    }

    /**
     * Get workout count for user
     */
    public long getWorkoutCount(String userId) {
        return workoutLogRepository.countByUserId(userId);
    }

    /**
     * Get workout count in date range
     */
    public long getWorkoutCountInRange(String userId, LocalDateTime start, LocalDateTime end) {
        return workoutLogRepository.countByUserIdAndStartTimeBetween(userId, start, end);
    }

    /**
     * Check for personal records and update them
     */
    private void checkAndUpdatePRs(WorkoutLog workoutLog) {
        List<PersonalRecord> newPRs = new ArrayList<>();

        for (WorkoutLog.ExerciseLog exercise : workoutLog.getExercises()) {
            Double maxWeight = 0.0;
            Integer maxReps = 0;
            Double maxVolume = 0.0;

            for (WorkoutLog.SetLog set : exercise.getSets()) {
                if (!set.isCompleted() || set.isWarmup())
                    continue;

                // Track max weight
                if (set.getWeight() != null && set.getWeight() > maxWeight) {
                    maxWeight = set.getWeight();
                }

                // Track max reps at any weight
                if (set.getReps() != null && set.getReps() > maxReps) {
                    maxReps = set.getReps();
                }

                // Track max volume (weight × reps)
                if (set.getWeight() != null && set.getReps() != null) {
                    double volume = set.getWeight() * set.getReps();
                    if (volume > maxVolume) {
                        maxVolume = volume;
                    }
                }
            }

            // Check if max weight is a new PR
            if (maxWeight > 0) {
                checkAndCreatePR(workoutLog.getUserId(), exercise.getExerciseId(),
                        exercise.getExerciseName(), "MAX_WEIGHT", maxWeight, "kg",
                        workoutLog.getId(), newPRs);
            }

            // Check if max volume is a new PR
            if (maxVolume > 0) {
                checkAndCreatePR(workoutLog.getUserId(), exercise.getExerciseId(),
                        exercise.getExerciseName(), "MAX_VOLUME", maxVolume, "kg×reps",
                        workoutLog.getId(), newPRs);
            }
        }

        // Mark sets as PRs in the workout log
        if (!newPRs.isEmpty()) {
            personalRecordRepository.saveAll(newPRs);
        }
    }

    private void checkAndCreatePR(String userId, String exerciseId, String exerciseName,
            String recordType, Double value, String unit,
            String workoutLogId, List<PersonalRecord> newPRs) {
        Optional<PersonalRecord> existing = personalRecordRepository
                .findByUserIdAndExerciseIdAndRecordType(userId, exerciseId, recordType);

        if (existing.isEmpty() || existing.get().getValue() < value) {
            PersonalRecord pr = PersonalRecord.builder()
                    .userId(userId)
                    .exerciseId(exerciseId)
                    .exerciseName(exerciseName)
                    .recordType(recordType)
                    .value(value)
                    .unit(unit)
                    .workoutLogId(workoutLogId)
                    .achievedAt(LocalDateTime.now())
                    .previousValue(existing.map(PersonalRecord::getValue).orElse(null))
                    .previousAchievedAt(existing.map(PersonalRecord::getAchievedAt).orElse(null))
                    .build();

            if (existing.isPresent()) {
                pr.setId(existing.get().getId());
            }

            newPRs.add(pr);
        }
    }

    /**
     * Get all PRs for a user
     */
    public List<PersonalRecord> getUserPRs(String userId) {
        return personalRecordRepository.findByUserIdOrderByAchievedAtDesc(userId);
    }

    /**
     * Get PRs for a specific exercise
     */
    public List<PersonalRecord> getExercisePRs(String userId, String exerciseId) {
        return personalRecordRepository.findByUserIdAndExerciseId(userId, exerciseId);
    }

    /**
     * Get recent PRs
     */
    public List<PersonalRecord> getRecentPRs(String userId) {
        return personalRecordRepository.findTop10ByUserIdOrderByAchievedAtDesc(userId);
    }
}
