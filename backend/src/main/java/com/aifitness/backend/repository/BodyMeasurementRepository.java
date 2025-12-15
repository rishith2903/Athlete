package com.aifitness.backend.repository;

import com.aifitness.backend.entity.BodyMeasurement;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository for BodyMeasurement entity.
 */
@Repository
public interface BodyMeasurementRepository extends MongoRepository<BodyMeasurement, String> {

    // Get all measurements for a user
    List<BodyMeasurement> findByUserIdOrderByMeasurementDateDesc(String userId);

    // Get measurements in date range
    List<BodyMeasurement> findByUserIdAndMeasurementDateBetweenOrderByMeasurementDateAsc(
            String userId, LocalDateTime start, LocalDateTime end);

    // Get latest measurement
    Optional<BodyMeasurement> findFirstByUserIdOrderByMeasurementDateDesc(String userId);

    // Get recent N measurements
    List<BodyMeasurement> findTop10ByUserIdOrderByMeasurementDateDesc(String userId);

    // Count measurements for a user
    long countByUserId(String userId);
}
