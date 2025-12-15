package com.aifitness.backend.controller;

import com.aifitness.backend.entity.BodyMeasurement;
import com.aifitness.backend.service.BodyMeasurementService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for Body Measurements.
 */
@RestController
@RequestMapping("/api/measurements")
@RequiredArgsConstructor
@Tag(name = "Body Measurements", description = "Track body measurements and progress")
public class BodyMeasurementController {

    private final BodyMeasurementService measurementService;

    @PostMapping
    @Operation(summary = "Add a new measurement")
    public ResponseEntity<Map<String, Object>> addMeasurement(
            @AuthenticationPrincipal UserDetails userDetails,
            @RequestBody BodyMeasurement measurement) {
        measurement.setUserId(getUserId(userDetails));
        BodyMeasurement saved = measurementService.addMeasurement(measurement);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", saved,
                "message", "Measurement added successfully"));
    }

    @GetMapping
    @Operation(summary = "Get all measurements")
    public ResponseEntity<Map<String, Object>> getMeasurements(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<BodyMeasurement> measurements = measurementService.getUserMeasurements(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", measurements));
    }

    @GetMapping("/recent")
    @Operation(summary = "Get recent measurements")
    public ResponseEntity<Map<String, Object>> getRecentMeasurements(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        List<BodyMeasurement> measurements = measurementService.getRecentMeasurements(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", measurements));
    }

    @GetMapping("/latest")
    @Operation(summary = "Get latest measurement")
    public ResponseEntity<Map<String, Object>> getLatestMeasurement(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        return measurementService.getLatestMeasurement(userId)
                .map(m -> ResponseEntity.ok(Map.<String, Object>of(
                        "success", true,
                        "data", m)))
                .orElse(ResponseEntity.ok(Map.of(
                        "success", true,
                        "data", Map.of())));
    }

    @GetMapping("/progress")
    @Operation(summary = "Get measurement progress")
    public ResponseEntity<Map<String, Object>> getProgress(
            @AuthenticationPrincipal UserDetails userDetails) {
        String userId = getUserId(userDetails);
        BodyMeasurementService.MeasurementProgress progress = measurementService.calculateProgress(userId);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", progress));
    }

    @GetMapping("/{id}")
    @Operation(summary = "Get measurement by ID")
    public ResponseEntity<Map<String, Object>> getMeasurement(@PathVariable String id) {
        return measurementService.getMeasurementById(id)
                .map(m -> ResponseEntity.ok(Map.<String, Object>of(
                        "success", true,
                        "data", m)))
                .orElse(ResponseEntity.notFound().build());
    }

    @PutMapping("/{id}")
    @Operation(summary = "Update a measurement")
    public ResponseEntity<Map<String, Object>> updateMeasurement(
            @PathVariable String id,
            @RequestBody BodyMeasurement measurement) {
        BodyMeasurement updated = measurementService.updateMeasurement(id, measurement);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "data", updated));
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "Delete a measurement")
    public ResponseEntity<Map<String, Object>> deleteMeasurement(@PathVariable String id) {
        measurementService.deleteMeasurement(id);
        return ResponseEntity.ok(Map.of(
                "success", true,
                "message", "Measurement deleted successfully"));
    }

    private String getUserId(UserDetails userDetails) {
        return userDetails.getUsername();
    }
}
