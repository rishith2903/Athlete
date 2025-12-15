package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;

import java.time.LocalDateTime;

/**
 * BodyMeasurement entity for tracking body stats over time.
 * Stores weight, body fat, and various body measurements.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "body_measurements")
public class BodyMeasurement {

    @Id
    private String id;

    @Indexed
    private String userId;

    @Indexed
    private LocalDateTime measurementDate;

    // Core measurements
    private Double weight; // kg or lbs
    private Double bodyFatPercentage;
    private Double bmi;

    // Body measurements (in cm or inches)
    private Double chest;
    private Double waist;
    private Double hips;
    private Double neck;

    // Arms
    private Double leftBicep;
    private Double rightBicep;
    private Double leftForearm;
    private Double rightForearm;

    // Legs
    private Double leftThigh;
    private Double rightThigh;
    private Double leftCalf;
    private Double rightCalf;

    // Other
    private Double shoulders;

    // Unit preference
    private String weightUnit; // KG or LBS
    private String measurementUnit; // CM or INCHES

    private String notes;

    @CreatedDate
    private LocalDateTime createdAt;
}
