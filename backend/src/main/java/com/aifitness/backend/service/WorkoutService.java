package com.aifitness.backend.service;

import com.aifitness.backend.entity.Workout;
import com.aifitness.backend.exception.BadRequestException;
import com.aifitness.backend.exception.ResourceNotFoundException;
import com.aifitness.backend.repository.WorkoutRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class WorkoutService {
    
    private final WorkoutRepository workoutRepository;
    
    public Page<Workout> getUserWorkouts(String userId, Pageable pageable) {
        return workoutRepository.findByUserId(userId, pageable);
    }
    
    public Workout getWorkout(String workoutId, String userId) {
        Workout workout = workoutRepository.findById(workoutId)
                .orElseThrow(() -> new ResourceNotFoundException("Workout not found"));
        
        if (!workout.getUserId().equals(userId)) {
            throw new BadRequestException("Access denied to this workout");
        }
        
        return workout;
    }
    
    @Transactional
    public Workout createWorkout(Workout workout) {
        workout.setStatus("PLANNED");
        workout.setAiGenerated(false);
        return workoutRepository.save(workout);
    }
    
    @Transactional
    public Workout updateWorkout(String workoutId, Workout updatedWorkout, String userId) {
        Workout workout = getWorkout(workoutId, userId);
        
        // Update fields
        workout.setName(updatedWorkout.getName());
        workout.setDescription(updatedWorkout.getDescription());
        workout.setType(updatedWorkout.getType());
        workout.setDifficulty(updatedWorkout.getDifficulty());
        workout.setDuration(updatedWorkout.getDuration());
        workout.setExercises(updatedWorkout.getExercises());
        workout.setScheduledFor(updatedWorkout.getScheduledFor());
        
        return workoutRepository.save(workout);
    }
    
    @Transactional
    public void deleteWorkout(String workoutId, String userId) {
        Workout workout = getWorkout(workoutId, userId);
        workoutRepository.delete(workout);
    }
    
    public List<Workout> getScheduledWorkouts(String userId, LocalDateTime startDate, LocalDateTime endDate) {
        return workoutRepository.findByUserIdAndScheduledForBetween(userId, startDate, endDate);
    }
    
    @Transactional
    public Workout startWorkout(String workoutId, String userId) {
        Workout workout = getWorkout(workoutId, userId);
        workout.setStatus("IN_PROGRESS");
        workout.setStartedAt(LocalDateTime.now());
        return workoutRepository.save(workout);
    }
    
    @Transactional
    public Workout completeWorkout(String workoutId, String userId, Map<String, Object> completionData) {
        Workout workout = getWorkout(workoutId, userId);
        workout.setStatus("COMPLETED");
        workout.setCompletedAt(LocalDateTime.now());
        
        if (completionData.containsKey("rating")) {
            workout.setRating((Integer) completionData.get("rating"));
        }
        if (completionData.containsKey("perceivedExertion")) {
            workout.setPerceivedExertion((Integer) completionData.get("perceivedExertion"));
        }
        if (completionData.containsKey("feedback")) {
            workout.setFeedback((String) completionData.get("feedback"));
        }
        if (completionData.containsKey("notes")) {
            workout.setNotes((String) completionData.get("notes"));
        }
        
        return workoutRepository.save(workout);
    }
    
    public Map<String, Object> getWorkoutStatistics(String userId, Integer days) {
        LocalDateTime since = LocalDateTime.now().minusDays(days);
        
        List<Workout> completedWorkouts = workoutRepository
                .findByUserIdAndCompletedAtBetween(userId, since, LocalDateTime.now());
        
        Long totalWorkouts = (long) completedWorkouts.size();
        Integer totalMinutes = completedWorkouts.stream()
                .mapToInt(w -> w.getDuration() != null ? w.getDuration() : 0)
                .sum();
        Integer totalCalories = completedWorkouts.stream()
                .mapToInt(w -> w.getCaloriesBurned() != null ? w.getCaloriesBurned() : 0)
                .sum();
        
        Map<String, Object> stats = new HashMap<>();
        stats.put("totalWorkouts", totalWorkouts);
        stats.put("totalMinutes", totalMinutes);
        stats.put("totalCalories", totalCalories);
        stats.put("averageWorkoutsPerWeek", totalWorkouts / (days / 7.0));
        stats.put("completedWorkouts", completedWorkouts);
        
        return stats;
    }
}