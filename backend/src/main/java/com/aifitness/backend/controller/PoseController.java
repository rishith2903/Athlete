package com.aifitness.backend.controller;

import com.aifitness.backend.entity.User;
import com.aifitness.backend.service.AiModelService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/pose")
@RequiredArgsConstructor
@SecurityRequirement(name = "bearerAuth")
@Tag(name = "Pose Analysis", description = "Exercise form checking and pose analysis endpoints")
public class PoseController {
    
    private final AiModelService aiModelService;
    
    @PostMapping(value = "/check", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Check exercise form", description = "Analyze exercise form from uploaded video or image")
    @ApiResponses(value = {
        @ApiResponse(responseCode = "200", description = "Form analysis completed"),
        @ApiResponse(responseCode = "400", description = "Invalid file or parameters"),
        @ApiResponse(responseCode = "413", description = "File too large"),
        @ApiResponse(responseCode = "415", description = "Unsupported media type"),
        @ApiResponse(responseCode = "500", description = "Analysis service error"),
        @ApiResponse(responseCode = "503", description = "AI service unavailable")
    })
    public Mono<ResponseEntity<Map<String, Object>>> checkExerciseForm(
            @RequestParam("file") MultipartFile file,
            @RequestParam("exerciseType") String exerciseType,
            @AuthenticationPrincipal User user) {
        
        // Validate file
        if (file.isEmpty()) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("error", "File is empty");
            errorResponse.put("status", "INVALID_INPUT");
            return Mono.just(ResponseEntity.badRequest().body(errorResponse));
        }
        
        // Check file size (50MB limit)
        if (file.getSize() > 50 * 1024 * 1024) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("error", "File size exceeds 50MB limit");
            errorResponse.put("status", "FILE_TOO_LARGE");
            return Mono.just(ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE).body(errorResponse));
        }
        
        // Check file type
        String contentType = file.getContentType();
        if (contentType == null || (!contentType.startsWith("image/") && !contentType.startsWith("video/"))) {
            Map<String, Object> errorResponse = new HashMap<>();
            errorResponse.put("error", "Only image and video files are supported");
            errorResponse.put("status", "UNSUPPORTED_MEDIA_TYPE");
            return Mono.just(ResponseEntity.status(HttpStatus.UNSUPPORTED_MEDIA_TYPE).body(errorResponse));
        }
        
        log.info("Processing form check for user {} - exercise: {}, file: {}, size: {} bytes", 
                user.getId(), exerciseType, file.getOriginalFilename(), file.getSize());
        
        return aiModelService.checkExerciseForm(file, exerciseType)
                .map(response -> {
                    // Add metadata
                    response.put("userId", user.getId());
                    response.put("exerciseType", exerciseType);
                    response.put("fileName", file.getOriginalFilename());
                    response.put("fileSize", file.getSize());
                    response.put("analyzed", true);
                    
                    // Ensure required fields exist
                    response.putIfAbsent("formScore", 0.0);
                    response.putIfAbsent("feedback", "Analysis completed");
                    response.putIfAbsent("corrections", new HashMap<>());
                    
                    return ResponseEntity.ok(response);
                })
                .onErrorResume(throwable -> {
                    log.error("Error during form check analysis", throwable);
                    
                    Map<String, Object> errorResponse = new HashMap<>();
                    errorResponse.put("error", "Failed to analyze exercise form");
                    errorResponse.put("message", throwable.getMessage());
                    errorResponse.put("status", "ANALYSIS_ERROR");
                    errorResponse.put("exerciseType", exerciseType);
                    
                    if (throwable.getMessage() != null && throwable.getMessage().contains("timeout")) {
                        errorResponse.put("status", "TIMEOUT");
                        return Mono.just(ResponseEntity.status(HttpStatus.GATEWAY_TIMEOUT).body(errorResponse));
                    }
                    
                    return Mono.just(ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(errorResponse));
                });
    }
}