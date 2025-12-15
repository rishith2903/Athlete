package com.aifitness.backend.service;

import com.aifitness.backend.entity.BodyMeasurement;
import com.aifitness.backend.repository.BodyMeasurementRepository;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Service for managing body measurements.
 */
@Service
@RequiredArgsConstructor
public class BodyMeasurementService {

    private final BodyMeasurementRepository measurementRepository;

    /**
     * Add a new measurement
     */
    public BodyMeasurement addMeasurement(BodyMeasurement measurement) {
        if (measurement.getMeasurementDate() == null) {
            measurement.setMeasurementDate(LocalDateTime.now());
        }
        // Calculate BMI if weight and height available
        return measurementRepository.save(measurement);
    }

    /**
     * Get all measurements for a user
     */
    public List<BodyMeasurement> getUserMeasurements(String userId) {
        return measurementRepository.findByUserIdOrderByMeasurementDateDesc(userId);
    }

    /**
     * Get recent measurements
     */
    public List<BodyMeasurement> getRecentMeasurements(String userId) {
        return measurementRepository.findTop10ByUserIdOrderByMeasurementDateDesc(userId);
    }

    /**
     * Get latest measurement
     */
    public Optional<BodyMeasurement> getLatestMeasurement(String userId) {
        return measurementRepository.findFirstByUserIdOrderByMeasurementDateDesc(userId);
    }

    /**
     * Get measurements in date range
     */
    public List<BodyMeasurement> getMeasurementsInRange(String userId, LocalDateTime start, LocalDateTime end) {
        return measurementRepository.findByUserIdAndMeasurementDateBetweenOrderByMeasurementDateAsc(
                userId, start, end);
    }

    /**
     * Update a measurement
     */
    public BodyMeasurement updateMeasurement(String id, BodyMeasurement measurement) {
        measurement.setId(id);
        return measurementRepository.save(measurement);
    }

    /**
     * Delete a measurement
     */
    public void deleteMeasurement(String id) {
        measurementRepository.deleteById(id);
    }

    /**
     * Get measurement by ID
     */
    public Optional<BodyMeasurement> getMeasurementById(String id) {
        return measurementRepository.findById(id);
    }

    /**
     * Calculate progress between two measurements
     */
    public MeasurementProgress calculateProgress(String userId) {
        List<BodyMeasurement> measurements = measurementRepository
                .findTop10ByUserIdOrderByMeasurementDateDesc(userId);

        if (measurements.size() < 2) {
            return MeasurementProgress.builder().build();
        }

        BodyMeasurement latest = measurements.get(0);
        BodyMeasurement previous = measurements.get(measurements.size() - 1);

        return MeasurementProgress.builder()
                .weightChange(calculateChange(latest.getWeight(), previous.getWeight()))
                .bodyFatChange(calculateChange(latest.getBodyFatPercentage(), previous.getBodyFatPercentage()))
                .chestChange(calculateChange(latest.getChest(), previous.getChest()))
                .waistChange(calculateChange(latest.getWaist(), previous.getWaist()))
                .leftBicepChange(calculateChange(latest.getLeftBicep(), previous.getLeftBicep()))
                .rightBicepChange(calculateChange(latest.getRightBicep(), previous.getRightBicep()))
                .periodDays(java.time.temporal.ChronoUnit.DAYS.between(
                        previous.getMeasurementDate(), latest.getMeasurementDate()))
                .build();
    }

    private Double calculateChange(Double current, Double previous) {
        if (current == null || previous == null)
            return null;
        return current - previous;
    }

    @Data
    @Builder
    public static class MeasurementProgress {
        private Double weightChange;
        private Double bodyFatChange;
        private Double chestChange;
        private Double waistChange;
        private Double leftBicepChange;
        private Double rightBicepChange;
        private long periodDays;
    }
}
