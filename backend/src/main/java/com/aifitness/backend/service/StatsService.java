package com.aifitness.backend.service;

import com.aifitness.backend.entity.WorkoutLog;
import com.aifitness.backend.repository.WorkoutLogRepository;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Service for workout statistics and progress analytics.
 */
@Service
@RequiredArgsConstructor
public class StatsService {

    private final WorkoutLogRepository workoutLogRepository;

    /**
     * Get workout frequency (workouts per week for last N weeks)
     */
    public List<WeeklyStats> getWorkoutFrequency(String userId, int weeks) {
        LocalDateTime start = LocalDateTime.now().minusWeeks(weeks);
        List<WorkoutLog> workouts = workoutLogRepository.findByUserIdAndStartTimeBetween(
                userId, start, LocalDateTime.now());

        List<WeeklyStats> stats = new ArrayList<>();
        LocalDate now = LocalDate.now();

        for (int i = weeks - 1; i >= 0; i--) {
            LocalDate weekStart = now.minusWeeks(i).with(java.time.DayOfWeek.MONDAY);
            LocalDate weekEnd = weekStart.plusDays(6);

            long count = workouts.stream()
                    .filter(w -> {
                        LocalDate workoutDate = w.getStartTime().toLocalDate();
                        return !workoutDate.isBefore(weekStart) && !workoutDate.isAfter(weekEnd);
                    })
                    .count();

            double totalVolume = workouts.stream()
                    .filter(w -> {
                        LocalDate workoutDate = w.getStartTime().toLocalDate();
                        return !workoutDate.isBefore(weekStart) && !workoutDate.isAfter(weekEnd);
                    })
                    .mapToDouble(w -> w.getTotalVolume() != null ? w.getTotalVolume() : 0)
                    .sum();

            stats.add(WeeklyStats.builder()
                    .weekStart(weekStart.toString())
                    .weekEnd(weekEnd.toString())
                    .workoutCount((int) count)
                    .totalVolume(totalVolume)
                    .build());
        }

        return stats;
    }

    /**
     * Get volume per muscle group over time
     */
    public Map<String, List<VolumeDataPoint>> getVolumeByMuscle(String userId, int days) {
        LocalDateTime start = LocalDateTime.now().minusDays(days);
        List<WorkoutLog> workouts = workoutLogRepository.findByUserIdAndStartTimeBetween(
                userId, start, LocalDateTime.now());

        Map<String, Map<String, Double>> muscleVolumeByDate = new HashMap<>();

        for (WorkoutLog workout : workouts) {
            String date = workout.getStartTime().toLocalDate().toString();

            for (WorkoutLog.ExerciseLog exercise : workout.getExercises()) {
                // For now, use category as muscle group
                // In production, you'd look up the exercise to get primary muscles
                String muscle = "Unknown";

                for (WorkoutLog.SetLog set : exercise.getSets()) {
                    if (set.isCompleted() && !set.isWarmup()) {
                        double volume = (set.getWeight() != null ? set.getWeight() : 0)
                                * (set.getReps() != null ? set.getReps() : 0);

                        muscleVolumeByDate
                                .computeIfAbsent(muscle, k -> new HashMap<>())
                                .merge(date, volume, Double::sum);
                    }
                }
            }
        }

        // Convert to list format
        Map<String, List<VolumeDataPoint>> result = new HashMap<>();
        for (Map.Entry<String, Map<String, Double>> entry : muscleVolumeByDate.entrySet()) {
            result.put(entry.getKey(), entry.getValue().entrySet().stream()
                    .map(e -> VolumeDataPoint.builder()
                            .date(e.getKey())
                            .volume(e.getValue())
                            .build())
                    .sorted(Comparator.comparing(VolumeDataPoint::getDate))
                    .collect(Collectors.toList()));
        }

        return result;
    }

    /**
     * Get strength progress for a specific exercise
     */
    public List<StrengthDataPoint> getStrengthProgress(String userId, String exerciseId, int days) {
        LocalDateTime start = LocalDateTime.now().minusDays(days);
        List<WorkoutLog> workouts = workoutLogRepository.findByUserIdAndStartTimeBetween(
                userId, start, LocalDateTime.now());

        List<StrengthDataPoint> dataPoints = new ArrayList<>();

        for (WorkoutLog workout : workouts) {
            for (WorkoutLog.ExerciseLog exercise : workout.getExercises()) {
                if (exerciseId.equals(exercise.getExerciseId())) {
                    double maxWeight = 0;
                    int maxReps = 0;
                    double maxVolume = 0;

                    for (WorkoutLog.SetLog set : exercise.getSets()) {
                        if (set.isCompleted() && !set.isWarmup()) {
                            if (set.getWeight() != null && set.getWeight() > maxWeight) {
                                maxWeight = set.getWeight();
                                maxReps = set.getReps() != null ? set.getReps() : 0;
                            }
                            double volume = (set.getWeight() != null ? set.getWeight() : 0)
                                    * (set.getReps() != null ? set.getReps() : 0);
                            if (volume > maxVolume) {
                                maxVolume = volume;
                            }
                        }
                    }

                    if (maxWeight > 0) {
                        dataPoints.add(StrengthDataPoint.builder()
                                .date(workout.getStartTime().toLocalDate().toString())
                                .weight(maxWeight)
                                .reps(maxReps)
                                .volume(maxVolume)
                                .estimated1RM(calculate1RM(maxWeight, maxReps))
                                .build());
                    }
                }
            }
        }

        return dataPoints.stream()
                .sorted(Comparator.comparing(StrengthDataPoint::getDate))
                .collect(Collectors.toList());
    }

    /**
     * Get workout calendar data
     */
    public List<CalendarDay> getWorkoutCalendar(String userId, int year, int month) {
        LocalDateTime start = LocalDateTime.of(year, month, 1, 0, 0);
        LocalDateTime end = start.plusMonths(1).minusSeconds(1);

        List<WorkoutLog> workouts = workoutLogRepository.findByUserIdAndStartTimeBetween(userId, start, end);

        Map<Integer, List<WorkoutLog>> workoutsByDay = workouts.stream()
                .collect(Collectors.groupingBy(w -> w.getStartTime().getDayOfMonth()));

        List<CalendarDay> calendar = new ArrayList<>();
        int daysInMonth = start.toLocalDate().lengthOfMonth();

        for (int day = 1; day <= daysInMonth; day++) {
            List<WorkoutLog> dayWorkouts = workoutsByDay.getOrDefault(day, List.of());
            calendar.add(CalendarDay.builder()
                    .day(day)
                    .hasWorkout(!dayWorkouts.isEmpty())
                    .workoutCount(dayWorkouts.size())
                    .workoutNames(dayWorkouts.stream()
                            .map(WorkoutLog::getName)
                            .collect(Collectors.toList()))
                    .build());
        }

        return calendar;
    }

    /**
     * Get dashboard summary stats
     */
    public DashboardStats getDashboardStats(String userId) {
        LocalDateTime now = LocalDateTime.now();
        LocalDateTime weekAgo = now.minusDays(7);
        LocalDateTime monthAgo = now.minusDays(30);

        long totalWorkouts = workoutLogRepository.countByUserId(userId);
        long weekWorkouts = workoutLogRepository.countByUserIdAndStartTimeBetween(userId, weekAgo, now);
        long monthWorkouts = workoutLogRepository.countByUserIdAndStartTimeBetween(userId, monthAgo, now);

        List<WorkoutLog> recentWorkouts = workoutLogRepository.findTop10ByUserIdOrderByStartTimeDesc(userId);

        // Calculate streak
        int currentStreak = calculateStreak(userId);

        double monthVolume = recentWorkouts.stream()
                .filter(w -> w.getStartTime().isAfter(monthAgo))
                .mapToDouble(w -> w.getTotalVolume() != null ? w.getTotalVolume() : 0)
                .sum();

        return DashboardStats.builder()
                .totalWorkouts((int) totalWorkouts)
                .workoutsThisWeek((int) weekWorkouts)
                .workoutsThisMonth((int) monthWorkouts)
                .currentStreak(currentStreak)
                .totalVolumeThisMonth(monthVolume)
                .build();
    }

    private int calculateStreak(String userId) {
        List<WorkoutLog> workouts = workoutLogRepository.findByUserIdOrderByStartTimeDesc(userId);
        if (workouts.isEmpty())
            return 0;

        int streak = 0;
        LocalDate checkDate = LocalDate.now();

        // Check if there's a workout today or yesterday to start counting
        LocalDate lastWorkoutDate = workouts.get(0).getStartTime().toLocalDate();
        long daysSinceLastWorkout = ChronoUnit.DAYS.between(lastWorkoutDate, checkDate);
        if (daysSinceLastWorkout > 1)
            return 0;

        Set<LocalDate> workoutDates = workouts.stream()
                .map(w -> w.getStartTime().toLocalDate())
                .collect(Collectors.toSet());

        while (workoutDates.contains(checkDate) || workoutDates.contains(checkDate.minusDays(1))) {
            if (workoutDates.contains(checkDate)) {
                streak++;
            }
            checkDate = checkDate.minusDays(1);
            if (streak > 365)
                break; // Safety limit
        }

        return streak;
    }

    /**
     * Calculate estimated 1RM using Epley formula
     */
    private double calculate1RM(double weight, int reps) {
        if (reps == 1)
            return weight;
        if (reps == 0)
            return 0;
        return weight * (1 + reps / 30.0);
    }

    // DTOs
    @Data
    @Builder
    public static class WeeklyStats {
        private String weekStart;
        private String weekEnd;
        private int workoutCount;
        private double totalVolume;
    }

    @Data
    @Builder
    public static class VolumeDataPoint {
        private String date;
        private double volume;
    }

    @Data
    @Builder
    public static class StrengthDataPoint {
        private String date;
        private double weight;
        private int reps;
        private double volume;
        private double estimated1RM;
    }

    @Data
    @Builder
    public static class CalendarDay {
        private int day;
        private boolean hasWorkout;
        private int workoutCount;
        private List<String> workoutNames;
    }

    @Data
    @Builder
    public static class DashboardStats {
        private int totalWorkouts;
        private int workoutsThisWeek;
        private int workoutsThisMonth;
        private int currentStreak;
        private double totalVolumeThisMonth;
    }
}
