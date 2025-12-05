package com.aifitness.backend.entity;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.LastModifiedDate;
import org.springframework.data.mongodb.core.index.Indexed;
import org.springframework.data.mongodb.core.mapping.Document;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Document(collection = "users")
public class User implements UserDetails {
    
    @Id
    private String id;
    
    @Indexed(unique = true)
    private String email;
    
    @Indexed(unique = true)
    private String username;
    
    private String password;
    
    private String firstName;
    private String lastName;
    private String phoneNumber;
    private LocalDate dateOfBirth;
    
    // Physical attributes
    private Double height; // in cm
    private Double weight; // in kg
    private String gender;
    private String activityLevel; // SEDENTARY, LIGHTLY_ACTIVE, MODERATELY_ACTIVE, VERY_ACTIVE, EXTREMELY_ACTIVE
    
    // Fitness goals
    private String fitnessGoal; // LOSE_WEIGHT, GAIN_MUSCLE, MAINTAIN, IMPROVE_ENDURANCE
    private Double targetWeight;
    private LocalDate targetDate;
    
    // Health information
    private Set<String> medicalConditions = new HashSet<>();
    private Set<String> allergies = new HashSet<>();
    private Set<String> dietaryRestrictions = new HashSet<>();
    
    // Preferences
    private Set<String> preferredExercises = new HashSet<>();
    private Set<String> equipmentAvailable = new HashSet<>();
    private Integer workoutDuration; // preferred duration in minutes
    private Integer workoutsPerWeek;
    
    // Account status
    private boolean enabled = true;
    private boolean accountNonExpired = true;
    private boolean accountNonLocked = true;
    private boolean credentialsNonExpired = true;
    
    @Builder.Default
    private Set<String> roles = new HashSet<>();
    
    private String profilePictureUrl;
    private String refreshToken;
    
    @CreatedDate
    private LocalDateTime createdAt;
    
    @LastModifiedDate
    private LocalDateTime updatedAt;
    
    private LocalDateTime lastLoginAt;
    
    // Subscription details
    private String subscriptionPlan; // FREE, BASIC, PREMIUM
    private LocalDate subscriptionExpiryDate;
    
    // Notification preferences
    private boolean emailNotifications = true;
    private boolean pushNotifications = true;
    private boolean workoutReminders = true;
    private boolean mealReminders = true;
    
    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return roles.stream()
                .map(role -> new SimpleGrantedAuthority("ROLE_" + role))
                .collect(Collectors.toList());
    }
    
    @Override
    public boolean isAccountNonExpired() {
        return accountNonExpired;
    }
    
    @Override
    public boolean isAccountNonLocked() {
        return accountNonLocked;
    }
    
    @Override
    public boolean isCredentialsNonExpired() {
        return credentialsNonExpired;
    }
    
    @Override
    public boolean isEnabled() {
        return enabled;
    }
}