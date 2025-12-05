package com.aifitness.backend.service;

import com.aifitness.backend.dto.auth.JwtResponse;
import com.aifitness.backend.dto.auth.LoginRequest;
import com.aifitness.backend.dto.auth.RegisterRequest;
import com.aifitness.backend.entity.User;
import com.aifitness.backend.exception.BadRequestException;
import com.aifitness.backend.exception.ResourceNotFoundException;
import com.aifitness.backend.repository.UserRepository;
import com.aifitness.backend.security.JwtTokenProvider;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

@Slf4j
@Service
@RequiredArgsConstructor
public class AuthService {
    
    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final JwtTokenProvider tokenProvider;
    private final AuthenticationManager authenticationManager;
    
    @Value("${jwt.expiration}")
    private Long jwtExpiration;
    
    @Transactional
    public JwtResponse register(RegisterRequest request) {
        // Check if user already exists
        if (userRepository.existsByEmail(request.getEmail())) {
            throw new BadRequestException("Email already registered");
        }
        
        if (userRepository.existsByUsername(request.getUsername())) {
            throw new BadRequestException("Username already taken");
        }
        
        // Create new user
        User user = User.builder()
                .username(request.getUsername())
                .email(request.getEmail())
                .password(passwordEncoder.encode(request.getPassword()))
                .firstName(request.getFirstName())
                .lastName(request.getLastName())
                .phoneNumber(request.getPhoneNumber())
                .dateOfBirth(request.getDateOfBirth())
                .height(request.getHeight())
                .weight(request.getWeight())
                .gender(request.getGender())
                .activityLevel(request.getActivityLevel())
                .fitnessGoal(request.getFitnessGoal())
                .targetWeight(request.getTargetWeight())
                .targetDate(request.getTargetDate())
                .medicalConditions(request.getMedicalConditions() != null ? request.getMedicalConditions() : new HashSet<>())
                .allergies(request.getAllergies() != null ? request.getAllergies() : new HashSet<>())
                .dietaryRestrictions(request.getDietaryRestrictions() != null ? request.getDietaryRestrictions() : new HashSet<>())
                .preferredExercises(request.getPreferredExercises() != null ? request.getPreferredExercises() : new HashSet<>())
                .equipmentAvailable(request.getEquipmentAvailable() != null ? request.getEquipmentAvailable() : new HashSet<>())
                .workoutDuration(request.getWorkoutDuration())
                .workoutsPerWeek(request.getWorkoutsPerWeek())
                .roles(Set.of("USER"))
                .subscriptionPlan("FREE")
                .enabled(true)
                .accountNonExpired(true)
                .accountNonLocked(true)
                .credentialsNonExpired(true)
                .build();
        
        user = userRepository.save(user);
        
        // Generate tokens
        String accessToken = tokenProvider.generateToken(user);
        String refreshToken = tokenProvider.generateRefreshToken(user);
        
        // Save refresh token
        user.setRefreshToken(refreshToken);
        user.setLastLoginAt(LocalDateTime.now());
        userRepository.save(user);
        
        return buildJwtResponse(user, accessToken, refreshToken);
    }
    
    @Transactional
    public JwtResponse login(LoginRequest request) {
        Authentication authentication = authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        request.getUsernameOrEmail(),
                        request.getPassword()
                )
        );
        
        User user = (User) authentication.getPrincipal();
        
        // Generate tokens
        String accessToken = tokenProvider.generateToken(user);
        String refreshToken = tokenProvider.generateRefreshToken(user);
        
        // Save refresh token and update last login
        user.setRefreshToken(refreshToken);
        user.setLastLoginAt(LocalDateTime.now());
        userRepository.save(user);
        
        return buildJwtResponse(user, accessToken, refreshToken);
    }
    
    @Transactional
    public JwtResponse refreshToken(String refreshToken) {
        if (!tokenProvider.validateToken(refreshToken)) {
            throw new BadRequestException("Invalid refresh token");
        }
        
        String username = tokenProvider.extractUsername(refreshToken);
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
        
        if (!refreshToken.equals(user.getRefreshToken())) {
            throw new BadRequestException("Invalid refresh token");
        }
        
        // Generate new tokens
        String newAccessToken = tokenProvider.generateToken(user);
        String newRefreshToken = tokenProvider.generateRefreshToken(user);
        
        // Save new refresh token
        user.setRefreshToken(newRefreshToken);
        userRepository.save(user);
        
        return buildJwtResponse(user, newAccessToken, newRefreshToken);
    }
    
    @Transactional
    public void logout(String token) {
        String jwt = token.substring(7); // Remove "Bearer " prefix
        String username = tokenProvider.extractUsername(jwt);
        
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new ResourceNotFoundException("User not found"));
        
        // Clear refresh token
        user.setRefreshToken(null);
        userRepository.save(user);
    }
    
    public void sendPasswordResetEmail(String email) {
        User user = userRepository.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with email: " + email));
        
        // TODO: Implement email service to send password reset email
        log.info("Password reset email would be sent to: {}", email);
    }
    
    @Transactional
    public void resetPassword(String token, String newPassword) {
        // TODO: Implement password reset token validation
        // For now, this is a placeholder
        log.info("Password reset functionality to be implemented");
    }
    
    @Transactional
    public void verifyEmail(String token) {
        // TODO: Implement email verification
        log.info("Email verification functionality to be implemented");
    }
    
    private JwtResponse buildJwtResponse(User user, String accessToken, String refreshToken) {
        return JwtResponse.builder()
                .accessToken(accessToken)
                .refreshToken(refreshToken)
                .tokenType("Bearer")
                .expiresIn(jwtExpiration)
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .firstName(user.getFirstName())
                .lastName(user.getLastName())
                .roles(user.getRoles())
                .subscriptionPlan(user.getSubscriptionPlan())
                .lastLoginAt(user.getLastLoginAt())
                .build();
    }
}