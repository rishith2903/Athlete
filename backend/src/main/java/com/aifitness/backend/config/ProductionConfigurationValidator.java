package com.aifitness.backend.config;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.Profiles;

import java.util.Arrays;

public class ProductionConfigurationValidator implements EnvironmentPostProcessor {

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        // Check if we are running in production profile
        if (environment.acceptsProfiles(Profiles.of("production"))) {
            validateProductionConfig(environment);
        }
    }

    private void validateProductionConfig(ConfigurableEnvironment environment) {
        String mongoUri = environment.getProperty("spring.data.mongodb.uri");
        
        // Also check the alternative property if the main one is missing
        if (mongoUri == null || mongoUri.isEmpty()) {
            mongoUri = environment.getProperty("mongodb.uri");
        }

        if (mongoUri == null || mongoUri.isEmpty() || mongoUri.contains("localhost") || mongoUri.contains("127.0.0.1")) {
            String errorMessage = 
                "\n********************************************************************************\n" +
                "CRITICAL CONFIGURATION ERROR: Invalid MongoDB URI for production!\n" +
                "********************************************************************************\n" +
                "The application is running with 'production' profile but MONGODB_URI is missing or invalid.\n" +
                "Current URI: " + (mongoUri == null || mongoUri.isEmpty() ? "[EMPTY]" : mongoUri) + "\n\n" +
                "ACTION REQUIRED:\n" +
                "1. If deploying on Render, ensure you are using a Blueprint (render.yaml).\n" +
                "2. If deploying manually, you MUST add the following Environment Variable:\n" +
                "   Key: MONGODB_URI\n" +
                "   Value: <your-internal-mongodb-connection-string>\n" +
                "********************************************************************************\n";
            
            // Throwing a RuntimeException here will stop the application startup immediately
            throw new RuntimeException(errorMessage);
        }
    }
}
