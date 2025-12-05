package com.aifitness.backend.dto.auth;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.util.Set;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RegisterRequest {
    
    @NotBlank(message = "Username is required")
    @Size(min = 3, max = 20, message = "Username must be between 3 and 20 characters")
    @Pattern(regexp = "^[a-zA-Z0-9_]+$", message = "Username can only contain letters, numbers, and underscores")
    private String username;
    
    @NotBlank(message = "Email is required")
    @Email(message = "Email should be valid")
    private String email;
    
    @NotBlank(message = "Password is required")
    @Size(min = 6, message = "Password must be at least 6 characters")
    private String password;
    
    @NotBlank(message = "First name is required")
    private String firstName;
    
    @NotBlank(message = "Last name is required")
    private String lastName;
    
    private String phoneNumber;
    private LocalDate dateOfBirth;
    
    // Physical attributes
    private Double height;
    private Double weight;
    private String gender;
    private String activityLevel;
    
    // Fitness goals
    private String fitnessGoal;
    private Double targetWeight;
    private LocalDate targetDate;
    
    // Health information
    private Set<String> medicalConditions;
    private Set<String> allergies;
    private Set<String> dietaryRestrictions;
    
    // Preferences
    private Set<String> preferredExercises;
    private Set<String> equipmentAvailable;
    private Integer workoutDuration;
    private Integer workoutsPerWeek;
}