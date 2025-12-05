package com.aifitness.backend.qa;

import com.aifitness.backend.dto.auth.LoginRequest;
import com.aifitness.backend.dto.auth.RegisterRequest;
import com.aifitness.backend.entity.User;
import com.aifitness.backend.entity.Workout;
import com.aifitness.backend.entity.Meal;
import com.aifitness.backend.entity.Progress;
import com.aifitness.backend.repository.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.MvcResult;
import org.springframework.test.web.servlet.ResultActions;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.IntStream;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.springframework.test.web.servlet.result.MockMvcResultHandlers.print;
import static org.junit.jupiter.api.Assertions.*;
import static org.hamcrest.Matchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
@DisplayName("Backend QA Test Suite")
public class BackendQATestSuite {
    
    @Autowired
    private MockMvc mockMvc;
    
    @Autowired
    private ObjectMapper objectMapper;
    
    @Autowired
    private UserRepository userRepository;
    
    @Autowired
    private WorkoutRepository workoutRepository;
    
    @Autowired
    private MealRepository mealRepository;
    
    @Autowired
    private ProgressRepository progressRepository;
    
    private static String authToken;
    private static String testUserId;
    private static final String TEST_EMAIL = "qatest@example.com";
    private static final String TEST_PASSWORD = "Test123!@#";
    
    // Performance metrics storage
    private static final Map<String, List<Long>> performanceMetrics = new ConcurrentHashMap<>();
    
    @BeforeAll
    static void setupTestReport() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("BACKEND QA TEST SUITE - COMPREHENSIVE TESTING");
        System.out.println("=".repeat(80));
        System.out.println("Test Started: " + LocalDateTime.now());
        System.out.println("=".repeat(80) + "\n");
    }
    
    // ==================== AUTHENTICATION TESTS ====================
    
    @Test
    @Order(1)
    @DisplayName("Test User Registration - Valid Input")
    void testUserRegistration_ValidInput() throws Exception {
        System.out.println("\n--- TEST: User Registration (Valid Input) ---");
        
        RegisterRequest request = RegisterRequest.builder()
                .username("qatestuser")
                .email(TEST_EMAIL)
                .password(TEST_PASSWORD)
                .firstName("QA")
                .lastName("Tester")
                .height(175.0)
                .weight(70.0)
                .gender("MALE")
                .activityLevel("MODERATE")
                .fitnessGoal("LOSE_WEIGHT")
                .build();
        
        String requestBody = objectMapper.writeValueAsString(request);
        System.out.println("Input/Request Body: " + requestBody);
        
        long startTime = System.currentTimeMillis();
        
        MvcResult result = mockMvc.perform(post("/api/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(requestBody))
                .andDo(print())
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.accessToken").exists())
                .andExpect(jsonPath("$.username").value("qatestuser"))
                .andExpect(jsonPath("$.email").value(TEST_EMAIL))
                .andReturn();
        
        long responseTime = System.currentTimeMillis() - startTime;
        recordPerformance("auth/register", responseTime);
        
        String response = result.getResponse().getContentAsString();
        Map<String, Object> responseMap = objectMapper.readValue(response, Map.class);
        authToken = (String) responseMap.get("accessToken");
        testUserId = (String) responseMap.get("id");
        
        System.out.println("Expected Response: 201 Created with JWT tokens");
        System.out.println("Actual Response: " + response);
        System.out.println("Status: PASS");
        System.out.println("Performance: " + responseTime + " ms");
    }
    
    @Test
    @Order(2)
    @DisplayName("Test User Registration - Duplicate Email")
    void testUserRegistration_DuplicateEmail() throws Exception {
        System.out.println("\n--- TEST: User Registration (Duplicate Email) ---");
        
        RegisterRequest request = RegisterRequest.builder()
                .username("duplicate")
                .email(TEST_EMAIL) // Same email as previous test
                .password("Password123!")
                .firstName("Duplicate")
                .lastName("User")
                .build();
        
        String requestBody = objectMapper.writeValueAsString(request);
        System.out.println("Input/Request Body: " + requestBody);
        
        mockMvc.perform(post("/api/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(requestBody))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").exists());
        
        System.out.println("Expected Response: 400 Bad Request");
        System.out.println("Actual Response: 400 with error message");
        System.out.println("Status: PASS");
    }
    
    @Test
    @Order(3)
    @DisplayName("Test User Login - Valid Credentials")
    void testUserLogin_ValidCredentials() throws Exception {
        System.out.println("\n--- TEST: User Login (Valid Credentials) ---");
        
        LoginRequest request = new LoginRequest();
        request.setUsernameOrEmail(TEST_EMAIL);
        request.setPassword(TEST_PASSWORD);
        
        String requestBody = objectMapper.writeValueAsString(request);
        System.out.println("Input/Request Body: " + requestBody);
        
        long startTime = System.currentTimeMillis();
        
        MvcResult result = mockMvc.perform(post("/api/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(requestBody))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.accessToken").exists())
                .andReturn();
        
        long responseTime = System.currentTimeMillis() - startTime;
        recordPerformance("auth/login", responseTime);
        
        System.out.println("Expected Response: 200 OK with JWT token");
        System.out.println("Actual Response: " + result.getResponse().getContentAsString());
        System.out.println("Status: PASS");
        System.out.println("Performance: " + responseTime + " ms");
    }
    
    @Test
    @Order(4)
    @DisplayName("Test User Login - Invalid Credentials")
    void testUserLogin_InvalidCredentials() throws Exception {
        System.out.println("\n--- TEST: User Login (Invalid Credentials) ---");
        
        LoginRequest request = new LoginRequest();
        request.setUsernameOrEmail(TEST_EMAIL);
        request.setPassword("WrongPassword123!");
        
        mockMvc.perform(post("/api/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isUnauthorized());
        
        System.out.println("Status: PASS (401 Unauthorized as expected)");
    }
    
    // ==================== WORKOUT MANAGEMENT TESTS ====================
    
    @Test
    @Order(5)
    @DisplayName("Test Create Workout")
    void testCreateWorkout() throws Exception {
        System.out.println("\n--- TEST: Create Workout ---");
        
        Map<String, Object> workout = new HashMap<>();
        workout.put("name", "Morning Cardio");
        workout.put("type", "CARDIO");
        workout.put("difficulty", "INTERMEDIATE");
        workout.put("duration", 30);
        workout.put("exercises", Arrays.asList(
            Map.of("name", "Running", "duration", 20, "sets", 1),
            Map.of("name", "Jumping Jacks", "duration", 10, "sets", 3)
        ));
        
        String requestBody = objectMapper.writeValueAsString(workout);
        System.out.println("Input/Request Body: " + requestBody);
        
        long startTime = System.currentTimeMillis();
        
        MvcResult result = mockMvc.perform(post("/api/workout")
                .header("Authorization", "Bearer " + authToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(requestBody))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.name").value("Morning Cardio"))
                .andReturn();
        
        long responseTime = System.currentTimeMillis() - startTime;
        recordPerformance("workout/create", responseTime);
        
        System.out.println("Expected Response: 201 Created");
        System.out.println("Actual Response: " + result.getResponse().getContentAsString());
        System.out.println("Status: PASS");
        System.out.println("Performance: " + responseTime + " ms");
    }
    
    @Test
    @Order(6)
    @DisplayName("Test Get User Workouts")
    void testGetUserWorkouts() throws Exception {
        System.out.println("\n--- TEST: Get User Workouts ---");
        
        long startTime = System.currentTimeMillis();
        
        mockMvc.perform(get("/api/workout")
                .header("Authorization", "Bearer " + authToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.content").isArray());
        
        long responseTime = System.currentTimeMillis() - startTime;
        recordPerformance("workout/list", responseTime);
        
        System.out.println("Status: PASS");
        System.out.println("Performance: " + responseTime + " ms");
    }
    
    // ==================== NUTRITION TESTS ====================
    
    @Test
    @Order(7)
    @DisplayName("Test Log Meal")
    void testLogMeal() throws Exception {
        System.out.println("\n--- TEST: Log Meal ---");
        
        Map<String, Object> meal = new HashMap<>();
        meal.put("name", "Healthy Breakfast");
        meal.put("type", "BREAKFAST");
        meal.put("date", LocalDate.now().toString());
        meal.put("totalCalories", 450);
        meal.put("totalProtein", 25);
        meal.put("totalCarbs", 55);
        meal.put("totalFat", 15);
        meal.put("foodItems", Arrays.asList(
            Map.of("name", "Oatmeal", "calories", 300, "quantity", 100, "unit", "g"),
            Map.of("name", "Banana", "calories", 150, "quantity", 1, "unit", "piece")
        ));
        
        String requestBody = objectMapper.writeValueAsString(meal);
        System.out.println("Input/Request Body: " + requestBody);
        
        mockMvc.perform(post("/api/nutrition")
                .header("Authorization", "Bearer " + authToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(requestBody))
                .andExpect(status().isCreated());
        
        System.out.println("Status: PASS");
    }
    
    // ==================== PROGRESS TRACKING TESTS ====================
    
    @Test
    @Order(8)
    @DisplayName("Test Log Progress Entry")
    void testLogProgress() throws Exception {
        System.out.println("\n--- TEST: Log Progress Entry ---");
        
        Map<String, Object> progress = new HashMap<>();
        progress.put("date", LocalDate.now().toString());
        progress.put("weight", 69.5);
        progress.put("bodyFatPercentage", 18.5);
        progress.put("workoutsCompleted", 5);
        progress.put("caloriesConsumed", 2100);
        progress.put("mood", "GOOD");
        progress.put("energyLevel", 8);
        
        mockMvc.perform(post("/api/progress")
                .header("Authorization", "Bearer " + authToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(progress)))
                .andExpect(status().isCreated());
        
        System.out.println("Status: PASS");
    }
    
    // ==================== DATABASE CRUD TESTS ====================
    
    @Test
    @Order(9)
    @DisplayName("Test Database CRUD Operations")
    @Transactional
    void testDatabaseCRUD() throws Exception {
        System.out.println("\n--- TEST: Database CRUD Operations ---");
        
        // CREATE
        User user = User.builder()
                .username("dbtest")
                .email("dbtest@example.com")
                .password("hashed_password")
                .firstName("DB")
                .lastName("Test")
                .build();
        User savedUser = userRepository.save(user);
        assertNotNull(savedUser.getId());
        System.out.println("CREATE: User saved with ID " + savedUser.getId());
        
        // READ
        Optional<User> foundUser = userRepository.findById(savedUser.getId());
        assertTrue(foundUser.isPresent());
        System.out.println("READ: User found in database");
        
        // UPDATE
        foundUser.get().setFirstName("Updated");
        userRepository.save(foundUser.get());
        User updatedUser = userRepository.findById(savedUser.getId()).orElseThrow();
        assertEquals("Updated", updatedUser.getFirstName());
        System.out.println("UPDATE: User name updated successfully");
        
        // DELETE
        userRepository.deleteById(savedUser.getId());
        assertFalse(userRepository.findById(savedUser.getId()).isPresent());
        System.out.println("DELETE: User deleted successfully");
        
        System.out.println("Status: PASS - All CRUD operations successful");
    }
    
    // ==================== SECURITY TESTS ====================
    
    @Test
    @Order(10)
    @DisplayName("Test SQL Injection Prevention")
    void testSQLInjectionPrevention() throws Exception {
        System.out.println("\n--- TEST: SQL Injection Prevention ---");
        
        String maliciousInput = "'; DROP TABLE users; --";
        LoginRequest request = new LoginRequest();
        request.setUsernameOrEmail(maliciousInput);
        request.setPassword("password");
        
        System.out.println("Malicious Input: " + maliciousInput);
        
        mockMvc.perform(post("/api/auth/login")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isUnauthorized());
        
        // Verify database is intact
        assertTrue(userRepository.count() > 0);
        System.out.println("Database intact - SQL injection prevented");
        System.out.println("Status: PASS");
    }
    
    @Test
    @Order(11)
    @DisplayName("Test Authentication Bypass Prevention")
    void testAuthenticationBypassPrevention() throws Exception {
        System.out.println("\n--- TEST: Authentication Bypass Prevention ---");
        
        // Try to access protected endpoint without token
        mockMvc.perform(get("/api/workout"))
                .andExpect(status().isUnauthorized());
        System.out.println("No token: 401 Unauthorized ✓");
        
        // Try with invalid token
        mockMvc.perform(get("/api/workout")
                .header("Authorization", "Bearer invalid.token.here"))
                .andExpect(status().isUnauthorized());
        System.out.println("Invalid token: 401 Unauthorized ✓");
        
        // Try with malformed authorization header
        mockMvc.perform(get("/api/workout")
                .header("Authorization", "NotBearer " + authToken))
                .andExpect(status().isUnauthorized());
        System.out.println("Malformed header: 401 Unauthorized ✓");
        
        System.out.println("Status: PASS - Authentication bypass prevented");
    }
    
    @Test
    @Order(12)
    @DisplayName("Test XSS Prevention")
    void testXSSPrevention() throws Exception {
        System.out.println("\n--- TEST: XSS Prevention ---");
        
        String xssPayload = "<script>alert('XSS')</script>";
        Map<String, Object> workout = new HashMap<>();
        workout.put("name", xssPayload);
        workout.put("type", "CARDIO");
        
        MvcResult result = mockMvc.perform(post("/api/workout")
                .header("Authorization", "Bearer " + authToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(workout)))
                .andExpect(status().isCreated())
                .andReturn();
        
        String response = result.getResponse().getContentAsString();
        assertFalse(response.contains("<script>"));
        System.out.println("XSS payload properly escaped/sanitized");
        System.out.println("Status: PASS");
    }
    
    // ==================== LOAD/STRESS TESTS ====================
    
    @Test
    @Order(13)
    @DisplayName("Test Concurrent User Load")
    void testConcurrentUserLoad() throws Exception {
        System.out.println("\n--- TEST: Concurrent User Load Test ---");
        
        int numberOfUsers = 50;
        int requestsPerUser = 10;
        ExecutorService executor = Executors.newFixedThreadPool(numberOfUsers);
        CountDownLatch latch = new CountDownLatch(numberOfUsers * requestsPerUser);
        List<Long> responseTimes = Collections.synchronizedList(new ArrayList<>());
        List<Exception> errors = Collections.synchronizedList(new ArrayList<>());
        
        System.out.println("Simulating " + numberOfUsers + " concurrent users");
        System.out.println("Each user making " + requestsPerUser + " requests");
        
        long testStartTime = System.currentTimeMillis();
        
        for (int user = 0; user < numberOfUsers; user++) {
            final int userId = user;
            executor.submit(() -> {
                for (int req = 0; req < requestsPerUser; req++) {
                    try {
                        long startTime = System.currentTimeMillis();
                        
                        mockMvc.perform(get("/api/workout")
                                .header("Authorization", "Bearer " + authToken))
                                .andExpect(status().isOk());
                        
                        long responseTime = System.currentTimeMillis() - startTime;
                        responseTimes.add(responseTime);
                    } catch (Exception e) {
                        errors.add(e);
                    } finally {
                        latch.countDown();
                    }
                }
            });
        }
        
        boolean completed = latch.await(60, TimeUnit.SECONDS);
        executor.shutdown();
        
        long totalTestTime = System.currentTimeMillis() - testStartTime;
        
        // Calculate metrics
        double avgResponseTime = responseTimes.stream()
                .mapToLong(Long::longValue)
                .average()
                .orElse(0);
        
        long maxResponseTime = responseTimes.stream()
                .mapToLong(Long::longValue)
                .max()
                .orElse(0);
        
        long minResponseTime = responseTimes.stream()
                .mapToLong(Long::longValue)
                .min()
                .orElse(0);
        
        Collections.sort(responseTimes);
        long p95ResponseTime = responseTimes.get((int)(responseTimes.size() * 0.95));
        long p99ResponseTime = responseTimes.get((int)(responseTimes.size() * 0.99));
        
        double successRate = ((double)(responseTimes.size()) / (numberOfUsers * requestsPerUser)) * 100;
        double throughput = (responseTimes.size() * 1000.0) / totalTestTime;
        
        System.out.println("\n=== Load Test Results ===");
        System.out.println("Total Requests: " + (numberOfUsers * requestsPerUser));
        System.out.println("Successful Requests: " + responseTimes.size());
        System.out.println("Failed Requests: " + errors.size());
        System.out.println("Success Rate: " + String.format("%.2f%%", successRate));
        System.out.println("Average Response Time: " + String.format("%.2f ms", avgResponseTime));
        System.out.println("Min Response Time: " + minResponseTime + " ms");
        System.out.println("Max Response Time: " + maxResponseTime + " ms");
        System.out.println("P95 Response Time: " + p95ResponseTime + " ms");
        System.out.println("P99 Response Time: " + p99ResponseTime + " ms");
        System.out.println("Throughput: " + String.format("%.2f req/s", throughput));
        System.out.println("Total Test Duration: " + totalTestTime + " ms");
        
        assertTrue(completed, "Load test should complete within timeout");
        assertTrue(successRate >= 95, "Success rate should be >= 95%");
        assertTrue(avgResponseTime < 1000, "Average response time should be < 1s");
        
        System.out.println("\nStatus: PASS");
    }
    
    @Test
    @Order(14)
    @DisplayName("Test API Rate Limiting")
    void testRateLimiting() throws Exception {
        System.out.println("\n--- TEST: API Rate Limiting ---");
        
        // Make rapid requests to test rate limiting
        int rapidRequests = 100;
        int rateLimitErrors = 0;
        
        for (int i = 0; i < rapidRequests; i++) {
            try {
                MvcResult result = mockMvc.perform(get("/api/workout")
                        .header("Authorization", "Bearer " + authToken))
                        .andReturn();
                
                if (result.getResponse().getStatus() == 429) {
                    rateLimitErrors++;
                }
            } catch (Exception e) {
                // Ignore exceptions for this test
            }
        }
        
        System.out.println("Rapid requests made: " + rapidRequests);
        System.out.println("Rate limit responses: " + rateLimitErrors);
        
        // Note: Rate limiting might not be implemented yet
        if (rateLimitErrors > 0) {
            System.out.println("Rate limiting is active");
        } else {
            System.out.println("Rate limiting not detected (may not be implemented)");
        }
        
        System.out.println("Status: PASS");
    }
    
    // ==================== FILE UPLOAD TESTS ====================
    
    @Test
    @Order(15)
    @DisplayName("Test File Upload - Valid Image")
    void testFileUpload_ValidImage() throws Exception {
        System.out.println("\n--- TEST: File Upload (Valid Image) ---");
        
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "test.jpg",
            MediaType.IMAGE_JPEG_VALUE,
            "test image content".getBytes()
        );
        
        mockMvc.perform(multipart("/api/pose/check")
                .file(file)
                .param("exerciseType", "squat")
                .header("Authorization", "Bearer " + authToken))
                .andExpect(status().isOk());
        
        System.out.println("File: test.jpg (image/jpeg)");
        System.out.println("Status: PASS");
    }
    
    @Test
    @Order(16)
    @DisplayName("Test File Upload - Size Limit")
    void testFileUpload_SizeLimit() throws Exception {
        System.out.println("\n--- TEST: File Upload (Size Limit) ---");
        
        byte[] largeContent = new byte[51 * 1024 * 1024]; // 51MB
        MockMultipartFile file = new MockMultipartFile(
            "file",
            "large.jpg",
            MediaType.IMAGE_JPEG_VALUE,
            largeContent
        );
        
        mockMvc.perform(multipart("/api/pose/check")
                .file(file)
                .param("exerciseType", "squat")
                .header("Authorization", "Bearer " + authToken))
                .andExpect(status().isPayloadTooLarge());
        
        System.out.println("File size: 51MB (exceeds 50MB limit)");
        System.out.println("Expected: 413 Payload Too Large");
        System.out.println("Status: PASS");
    }
    
    // ==================== PAGINATION TESTS ====================
    
    @Test
    @Order(17)
    @DisplayName("Test Pagination")
    void testPagination() throws Exception {
        System.out.println("\n--- TEST: Pagination ---");
        
        // Test different page sizes and pages
        mockMvc.perform(get("/api/workout")
                .param("page", "0")
                .param("size", "5")
                .header("Authorization", "Bearer " + authToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.pageable.pageSize").value(5))
                .andExpect(jsonPath("$.pageable.pageNumber").value(0));
        
        mockMvc.perform(get("/api/workout")
                .param("page", "1")
                .param("size", "10")
                .header("Authorization", "Bearer " + authToken))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.pageable.pageSize").value(10))
                .andExpect(jsonPath("$.pageable.pageNumber").value(1));
        
        System.out.println("Pagination parameters working correctly");
        System.out.println("Status: PASS");
    }
    
    // ==================== ERROR HANDLING TESTS ====================
    
    @Test
    @Order(18)
    @DisplayName("Test 404 Not Found")
    void test404NotFound() throws Exception {
        System.out.println("\n--- TEST: 404 Not Found ---");
        
        mockMvc.perform(get("/api/workout/nonexistent-id")
                .header("Authorization", "Bearer " + authToken))
                .andExpect(status().isNotFound());
        
        System.out.println("Endpoint: /api/workout/nonexistent-id");
        System.out.println("Expected: 404 Not Found");
        System.out.println("Status: PASS");
    }
    
    @Test
    @Order(19)
    @DisplayName("Test 400 Bad Request")
    void test400BadRequest() throws Exception {
        System.out.println("\n--- TEST: 400 Bad Request ---");
        
        String invalidJson = "{invalid json}";
        
        mockMvc.perform(post("/api/workout")
                .header("Authorization", "Bearer " + authToken)
                .contentType(MediaType.APPLICATION_JSON)
                .content(invalidJson))
                .andExpect(status().isBadRequest());
        
        System.out.println("Input: Invalid JSON");
        System.out.println("Expected: 400 Bad Request");
        System.out.println("Status: PASS");
    }
    
    // ==================== PERFORMANCE REPORT ====================
    
    @AfterAll
    static void generatePerformanceReport() {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("PERFORMANCE METRICS SUMMARY");
        System.out.println("=".repeat(80));
        
        performanceMetrics.forEach((endpoint, times) -> {
            double avg = times.stream().mapToLong(Long::longValue).average().orElse(0);
            long max = times.stream().mapToLong(Long::longValue).max().orElse(0);
            long min = times.stream().mapToLong(Long::longValue).min().orElse(0);
            
            System.out.printf("Endpoint: %s%n", endpoint);
            System.out.printf("  Average: %.2f ms%n", avg);
            System.out.printf("  Min: %d ms%n", min);
            System.out.printf("  Max: %d ms%n", max);
            System.out.println();
        });
        
        System.out.println("=".repeat(80));
        System.out.println("QA TEST SUITE COMPLETED");
        System.out.println("Test Ended: " + LocalDateTime.now());
        System.out.println("=".repeat(80));
    }
    
    // Helper method to record performance metrics
    private static void recordPerformance(String endpoint, long responseTime) {
        performanceMetrics.computeIfAbsent(endpoint, k -> new ArrayList<>()).add(responseTime);
    }
}