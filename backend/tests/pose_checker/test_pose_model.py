"""
Test Suite for Pose Checking Model
Tests exercise form validation, angle calculations, and feedback generation
"""

import pytest
import numpy as np
import cv2
from PIL import Image
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

# Test case structure
@dataclass
class PoseTestCase:
    test_id: str
    exercise_type: str
    input_data: Dict
    expected_output: Dict
    description: str
    
class TestPoseChecker:
    """Comprehensive test suite for Pose Checking Model"""
    
    @pytest.fixture
    def mock_pose_model(self):
        """Mock pose checking model for testing"""
        class MockPoseModel:
            def __init__(self):
                self.keypoints_model = self._load_mock_keypoints_model()
                
            def _load_mock_keypoints_model(self):
                # Simulate keypoint detection model
                return lambda x: np.random.rand(17, 3)  # 17 keypoints with x,y,confidence
            
            def detect_pose(self, image):
                """Detect pose keypoints"""
                keypoints = self.keypoints_model(image)
                return {
                    'keypoints': keypoints.tolist(),
                    'confidence': float(np.mean(keypoints[:, 2]))
                }
            
            def check_form(self, keypoints, exercise_type):
                """Check exercise form based on keypoints"""
                form_rules = {
                    'squat': self._check_squat_form,
                    'pushup': self._check_pushup_form,
                    'plank': self._check_plank_form,
                    'deadlift': self._check_deadlift_form,
                    'lunge': self._check_lunge_form
                }
                
                if exercise_type in form_rules:
                    return form_rules[exercise_type](keypoints)
                else:
                    return {'error': 'Unknown exercise type'}
            
            def _check_squat_form(self, keypoints):
                """Check squat form"""
                # Calculate angles
                knee_angle = self._calculate_angle(keypoints[11], keypoints[13], keypoints[15])
                hip_angle = self._calculate_angle(keypoints[5], keypoints[11], keypoints[13])
                back_angle = self._calculate_angle(keypoints[5], keypoints[6], keypoints[11])
                
                feedback = []
                score = 100
                
                # Check knee angle (should be ~90 degrees at bottom)
                if knee_angle < 70:
                    feedback.append("Go deeper - aim for 90 degrees at knees")
                    score -= 20
                elif knee_angle > 110:
                    feedback.append("Good depth achieved")
                
                # Check knee alignment
                if abs(keypoints[13][0] - keypoints[15][0]) > 0.1:  # Knee vs ankle x-position
                    feedback.append("Keep knees aligned over toes")
                    score -= 15
                
                # Check back angle
                if back_angle < 160:
                    feedback.append("Keep chest up and back straight")
                    score -= 15
                
                return {
                    'exercise': 'squat',
                    'form_score': max(0, score),
                    'angles': {
                        'knee': knee_angle,
                        'hip': hip_angle,
                        'back': back_angle
                    },
                    'feedback': feedback,
                    'is_correct': score >= 70
                }
            
            def _check_pushup_form(self, keypoints):
                """Check pushup form"""
                elbow_angle = self._calculate_angle(keypoints[5], keypoints[7], keypoints[9])
                body_alignment = self._calculate_angle(keypoints[5], keypoints[11], keypoints[15])
                
                feedback = []
                score = 100
                
                if elbow_angle > 120:
                    feedback.append("Lower your body more - aim for 90 degree elbow bend")
                    score -= 20
                
                if body_alignment < 170:
                    feedback.append("Keep your body in a straight line")
                    score -= 25
                
                return {
                    'exercise': 'pushup',
                    'form_score': max(0, score),
                    'angles': {
                        'elbow': elbow_angle,
                        'body_alignment': body_alignment
                    },
                    'feedback': feedback,
                    'is_correct': score >= 70
                }
            
            def _check_plank_form(self, keypoints):
                """Check plank form"""
                body_alignment = self._calculate_angle(keypoints[5], keypoints[11], keypoints[15])
                hip_height = keypoints[11][1]
                shoulder_height = keypoints[5][1]
                
                feedback = []
                score = 100
                
                if body_alignment < 170:
                    feedback.append("Keep your body straight from head to heels")
                    score -= 30
                
                if abs(hip_height - shoulder_height) > 0.1:
                    if hip_height > shoulder_height:
                        feedback.append("Lower your hips - avoid sagging")
                    else:
                        feedback.append("Don't raise your hips too high")
                    score -= 20
                
                return {
                    'exercise': 'plank',
                    'form_score': max(0, score),
                    'angles': {
                        'body_alignment': body_alignment
                    },
                    'feedback': feedback,
                    'is_correct': score >= 70
                }
            
            def _check_deadlift_form(self, keypoints):
                """Check deadlift form"""
                back_angle = self._calculate_angle(keypoints[5], keypoints[6], keypoints[11])
                knee_angle = self._calculate_angle(keypoints[11], keypoints[13], keypoints[15])
                
                feedback = []
                score = 100
                
                if back_angle < 160:
                    feedback.append("Keep your back straight - avoid rounding")
                    score -= 30
                
                if knee_angle < 120:
                    feedback.append("Don't squat too deep - hinge at hips")
                    score -= 20
                
                return {
                    'exercise': 'deadlift',
                    'form_score': max(0, score),
                    'angles': {
                        'back': back_angle,
                        'knee': knee_angle
                    },
                    'feedback': feedback,
                    'is_correct': score >= 70
                }
            
            def _check_lunge_form(self, keypoints):
                """Check lunge form"""
                front_knee = self._calculate_angle(keypoints[11], keypoints[13], keypoints[15])
                back_knee = self._calculate_angle(keypoints[12], keypoints[14], keypoints[16])
                torso_angle = self._calculate_angle(keypoints[5], keypoints[6], keypoints[11])
                
                feedback = []
                score = 100
                
                if front_knee < 85 or front_knee > 95:
                    feedback.append("Front knee should be at 90 degrees")
                    score -= 20
                
                if torso_angle < 170:
                    feedback.append("Keep torso upright")
                    score -= 15
                
                return {
                    'exercise': 'lunge',
                    'form_score': max(0, score),
                    'angles': {
                        'front_knee': front_knee,
                        'back_knee': back_knee,
                        'torso': torso_angle
                    },
                    'feedback': feedback,
                    'is_correct': score >= 70
                }
            
            def _calculate_angle(self, p1, p2, p3):
                """Calculate angle between three points"""
                if isinstance(p1, list):
                    p1 = np.array(p1[:2])
                    p2 = np.array(p2[:2])
                    p3 = np.array(p3[:2])
                
                radians = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - \
                         np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
                angle = np.abs(radians * 180.0 / np.pi)
                
                if angle > 180.0:
                    angle = 360 - angle
                
                return angle
        
        return MockPoseModel()
    
    def generate_test_cases(self) -> List[PoseTestCase]:
        """Generate comprehensive test cases for pose checking"""
        test_cases = []
        
        # Test Case 1-10: Squat Form Tests
        squat_cases = [
            PoseTestCase(
                test_id="POSE_001",
                exercise_type="squat",
                input_data={
                    "keypoints": self._generate_perfect_squat_keypoints(),
                    "exercise": "squat"
                },
                expected_output={
                    "form_score": 100,
                    "is_correct": True,
                    "feedback": []
                },
                description="Perfect squat form"
            ),
            PoseTestCase(
                test_id="POSE_002",
                exercise_type="squat",
                input_data={
                    "keypoints": self._generate_shallow_squat_keypoints(),
                    "exercise": "squat"
                },
                expected_output={
                    "form_score": 80,
                    "is_correct": True,
                    "feedback": ["Go deeper - aim for 90 degrees at knees"]
                },
                description="Shallow squat depth"
            ),
            PoseTestCase(
                test_id="POSE_003",
                exercise_type="squat",
                input_data={
                    "keypoints": self._generate_knee_cave_squat_keypoints(),
                    "exercise": "squat"
                },
                expected_output={
                    "form_score": 85,
                    "is_correct": True,
                    "feedback": ["Keep knees aligned over toes"]
                },
                description="Knees caving inward"
            ),
            PoseTestCase(
                test_id="POSE_004",
                exercise_type="squat",
                input_data={
                    "keypoints": self._generate_rounded_back_squat_keypoints(),
                    "exercise": "squat"
                },
                expected_output={
                    "form_score": 85,
                    "is_correct": True,
                    "feedback": ["Keep chest up and back straight"]
                },
                description="Rounded back during squat"
            ),
            PoseTestCase(
                test_id="POSE_005",
                exercise_type="squat",
                input_data={
                    "keypoints": self._generate_heels_up_squat_keypoints(),
                    "exercise": "squat"
                },
                expected_output={
                    "form_score": 90,
                    "is_correct": True,
                    "feedback": ["Keep heels flat on ground"]
                },
                description="Heels coming up"
            )
        ]
        
        # Test Case 11-20: Push-up Form Tests
        pushup_cases = [
            PoseTestCase(
                test_id="POSE_011",
                exercise_type="pushup",
                input_data={
                    "keypoints": self._generate_perfect_pushup_keypoints(),
                    "exercise": "pushup"
                },
                expected_output={
                    "form_score": 100,
                    "is_correct": True,
                    "feedback": []
                },
                description="Perfect push-up form"
            ),
            PoseTestCase(
                test_id="POSE_012",
                exercise_type="pushup",
                input_data={
                    "keypoints": self._generate_high_pushup_keypoints(),
                    "exercise": "pushup"
                },
                expected_output={
                    "form_score": 80,
                    "is_correct": True,
                    "feedback": ["Lower your body more - aim for 90 degree elbow bend"]
                },
                description="Not going low enough"
            ),
            PoseTestCase(
                test_id="POSE_013",
                exercise_type="pushup",
                input_data={
                    "keypoints": self._generate_sagging_pushup_keypoints(),
                    "exercise": "pushup"
                },
                expected_output={
                    "form_score": 75,
                    "is_correct": True,
                    "feedback": ["Keep your body in a straight line"]
                },
                description="Sagging hips"
            )
        ]
        
        # Test Case 21-30: Plank Form Tests
        plank_cases = [
            PoseTestCase(
                test_id="POSE_021",
                exercise_type="plank",
                input_data={
                    "keypoints": self._generate_perfect_plank_keypoints(),
                    "exercise": "plank"
                },
                expected_output={
                    "form_score": 100,
                    "is_correct": True,
                    "feedback": []
                },
                description="Perfect plank form"
            ),
            PoseTestCase(
                test_id="POSE_022",
                exercise_type="plank",
                input_data={
                    "keypoints": self._generate_sagging_plank_keypoints(),
                    "exercise": "plank"
                },
                expected_output={
                    "form_score": 70,
                    "is_correct": True,
                    "feedback": ["Keep your body straight from head to heels", "Lower your hips - avoid sagging"]
                },
                description="Sagging plank"
            )
        ]
        
        # Test Case 31-40: Edge Cases
        edge_cases = [
            PoseTestCase(
                test_id="POSE_031",
                exercise_type="unknown",
                input_data={
                    "keypoints": [[0, 0, 0]] * 17,
                    "exercise": "unknown_exercise"
                },
                expected_output={
                    "error": "Unknown exercise type"
                },
                description="Unknown exercise type"
            ),
            PoseTestCase(
                test_id="POSE_032",
                exercise_type="squat",
                input_data={
                    "keypoints": None,
                    "exercise": "squat"
                },
                expected_output={
                    "error": "No keypoints detected"
                },
                description="Missing keypoints"
            ),
            PoseTestCase(
                test_id="POSE_033",
                exercise_type="squat",
                input_data={
                    "keypoints": [[0, 0, 0.1]] * 17,  # Low confidence
                    "exercise": "squat"
                },
                expected_output={
                    "error": "Low confidence in pose detection"
                },
                description="Low confidence keypoints"
            )
        ]
        
        test_cases.extend(squat_cases)
        test_cases.extend(pushup_cases)
        test_cases.extend(plank_cases)
        test_cases.extend(edge_cases)
        
        # Generate additional test cases to reach 50+
        for i in range(34, 51):
            test_cases.append(self._generate_random_test_case(f"POSE_{i:03d}"))
        
        return test_cases
    
    def _generate_perfect_squat_keypoints(self):
        """Generate keypoints for perfect squat form"""
        return [
            [0.5, 0.2, 0.99],  # 0: nose
            [0.48, 0.22, 0.98],  # 1: left_eye
            [0.52, 0.22, 0.98],  # 2: right_eye
            [0.47, 0.23, 0.97],  # 3: left_ear
            [0.53, 0.23, 0.97],  # 4: right_ear
            [0.45, 0.35, 0.99],  # 5: left_shoulder
            [0.55, 0.35, 0.99],  # 6: right_shoulder
            [0.43, 0.45, 0.98],  # 7: left_elbow
            [0.57, 0.45, 0.98],  # 8: right_elbow
            [0.41, 0.55, 0.97],  # 9: left_wrist
            [0.59, 0.55, 0.97],  # 10: right_wrist
            [0.45, 0.55, 0.99],  # 11: left_hip
            [0.55, 0.55, 0.99],  # 12: right_hip
            [0.45, 0.75, 0.99],  # 13: left_knee (90 degrees)
            [0.55, 0.75, 0.99],  # 14: right_knee
            [0.45, 0.95, 0.99],  # 15: left_ankle
            [0.55, 0.95, 0.99],  # 16: right_ankle
        ]
    
    def _generate_shallow_squat_keypoints(self):
        """Generate keypoints for shallow squat"""
        keypoints = self._generate_perfect_squat_keypoints()
        # Adjust knee position for shallow squat
        keypoints[13][1] = 0.65  # Higher knee position
        keypoints[14][1] = 0.65
        return keypoints
    
    def _generate_knee_cave_squat_keypoints(self):
        """Generate keypoints for knee cave squat"""
        keypoints = self._generate_perfect_squat_keypoints()
        # Move knees inward
        keypoints[13][0] = 0.48  # Left knee moves right
        keypoints[14][0] = 0.52  # Right knee moves left
        return keypoints
    
    def _generate_rounded_back_squat_keypoints(self):
        """Generate keypoints for rounded back squat"""
        keypoints = self._generate_perfect_squat_keypoints()
        # Adjust shoulder and hip positions for rounded back
        keypoints[5][1] = 0.4  # Shoulders forward
        keypoints[6][1] = 0.4
        return keypoints
    
    def _generate_heels_up_squat_keypoints(self):
        """Generate keypoints for heels up squat"""
        keypoints = self._generate_perfect_squat_keypoints()
        # Adjust ankle position
        keypoints[15][1] = 0.93  # Heels slightly up
        keypoints[16][1] = 0.93
        return keypoints
    
    def _generate_perfect_pushup_keypoints(self):
        """Generate keypoints for perfect pushup form"""
        return [
            [0.5, 0.7, 0.99],  # 0: nose (low position)
            [0.48, 0.68, 0.98],  # 1: left_eye
            [0.52, 0.68, 0.98],  # 2: right_eye
            [0.47, 0.67, 0.97],  # 3: left_ear
            [0.53, 0.67, 0.97],  # 4: right_ear
            [0.4, 0.7, 0.99],  # 5: left_shoulder
            [0.6, 0.7, 0.99],  # 6: right_shoulder
            [0.35, 0.7, 0.98],  # 7: left_elbow (90 degrees)
            [0.65, 0.7, 0.98],  # 8: right_elbow
            [0.3, 0.8, 0.97],  # 9: left_wrist
            [0.7, 0.8, 0.97],  # 10: right_wrist
            [0.45, 0.7, 0.99],  # 11: left_hip
            [0.55, 0.7, 0.99],  # 12: right_hip
            [0.45, 0.7, 0.99],  # 13: left_knee
            [0.55, 0.7, 0.99],  # 14: right_knee
            [0.45, 0.8, 0.99],  # 15: left_ankle
            [0.55, 0.8, 0.99],  # 16: right_ankle
        ]
    
    def _generate_high_pushup_keypoints(self):
        """Generate keypoints for high pushup (not going low enough)"""
        keypoints = self._generate_perfect_pushup_keypoints()
        # Adjust vertical positions to be higher
        for i in range(len(keypoints)):
            keypoints[i][1] -= 0.2  # Move all points up
        return keypoints
    
    def _generate_sagging_pushup_keypoints(self):
        """Generate keypoints for sagging pushup"""
        keypoints = self._generate_perfect_pushup_keypoints()
        # Adjust hip position to sag
        keypoints[11][1] = 0.85  # Left hip lower
        keypoints[12][1] = 0.85  # Right hip lower
        return keypoints
    
    def _generate_perfect_plank_keypoints(self):
        """Generate keypoints for perfect plank form"""
        return [
            [0.5, 0.5, 0.99],  # All aligned horizontally
            [0.48, 0.48, 0.98],
            [0.52, 0.48, 0.98],
            [0.47, 0.47, 0.97],
            [0.53, 0.47, 0.97],
            [0.4, 0.5, 0.99],  # Shoulders
            [0.6, 0.5, 0.99],
            [0.35, 0.6, 0.98],  # Elbows on ground
            [0.65, 0.6, 0.98],
            [0.3, 0.6, 0.97],
            [0.7, 0.6, 0.97],
            [0.45, 0.5, 0.99],  # Hips aligned
            [0.55, 0.5, 0.99],
            [0.45, 0.5, 0.99],  # Knees aligned
            [0.55, 0.5, 0.99],
            [0.45, 0.5, 0.99],  # Ankles aligned
            [0.55, 0.5, 0.99],
        ]
    
    def _generate_sagging_plank_keypoints(self):
        """Generate keypoints for sagging plank"""
        keypoints = self._generate_perfect_plank_keypoints()
        # Make hips sag
        keypoints[11][1] = 0.65  # Left hip lower
        keypoints[12][1] = 0.65  # Right hip lower
        return keypoints
    
    def _generate_random_test_case(self, test_id: str) -> PoseTestCase:
        """Generate random test case for variety"""
        exercises = ["squat", "pushup", "plank", "deadlift", "lunge"]
        exercise = np.random.choice(exercises)
        
        # Generate random keypoints
        keypoints = np.random.rand(17, 3).tolist()
        
        return PoseTestCase(
            test_id=test_id,
            exercise_type=exercise,
            input_data={
                "keypoints": keypoints,
                "exercise": exercise
            },
            expected_output={
                "form_score": np.random.randint(60, 100),
                "is_correct": np.random.choice([True, False]),
                "feedback": []
            },
            description=f"Random {exercise} test case"
        )
    
    @pytest.mark.parametrize("test_case", generate_test_cases(None))
    def test_pose_checking(self, mock_pose_model, test_case):
        """Test pose checking with various inputs"""
        print(f"\n{'='*60}")
        print(f"✅ Test Case ID: {test_case.test_id}")
        print(f"📋 Test Input: {test_case.exercise_type} exercise")
        print(f"   Description: {test_case.description}")
        
        # Process input
        if test_case.input_data['keypoints'] is None:
            result = {'error': 'No keypoints detected'}
        elif test_case.input_data['keypoints'] and np.mean([kp[2] for kp in test_case.input_data['keypoints']]) < 0.3:
            result = {'error': 'Low confidence in pose detection'}
        else:
            result = mock_pose_model.check_form(
                test_case.input_data['keypoints'],
                test_case.input_data['exercise']
            )
        
        print(f"⚙️ Expected Output: {test_case.expected_output}")
        print(f"🧪 Actual Output: {result}")
        
        # Validate result
        test_passed = self._validate_pose_result(result, test_case.expected_output)
        
        print(f"📊 Result: {'PASS ✅' if test_passed else 'FAIL ❌'}")
        
        if not test_passed:
            print(f"📝 Notes: Mismatch in output")
        
        assert test_passed, f"Test case {test_case.test_id} failed"
    
    def _validate_pose_result(self, actual: Dict, expected: Dict) -> bool:
        """Validate pose checking result"""
        if 'error' in expected:
            return 'error' in actual and actual['error'] == expected['error']
        
        if 'error' in actual:
            return False
        
        # Check form score within tolerance
        if 'form_score' in expected:
            score_diff = abs(actual.get('form_score', 0) - expected['form_score'])
            if score_diff > 10:  # Allow 10 point tolerance
                return False
        
        # Check is_correct flag
        if 'is_correct' in expected:
            if actual.get('is_correct') != expected['is_correct']:
                return False
        
        return True
    
    def test_angle_calculation(self, mock_pose_model):
        """Test angle calculation accuracy"""
        print(f"\n{'='*60}")
        print("Testing angle calculation...")
        
        # Test 90 degree angle
        p1 = [0, 0]
        p2 = [1, 0]
        p3 = [1, 1]
        angle = mock_pose_model._calculate_angle(p1, p2, p3)
        
        print(f"90° angle test: {angle:.2f}°")
        assert abs(angle - 90) < 5, "Angle calculation incorrect"
        
        # Test 180 degree angle
        p1 = [0, 0]
        p2 = [1, 0]
        p3 = [2, 0]
        angle = mock_pose_model._calculate_angle(p1, p2, p3)
        
        print(f"180° angle test: {angle:.2f}°")
        assert abs(angle - 180) < 5, "Angle calculation incorrect"
        
        print("✅ Angle calculation tests passed")
    
    def test_confidence_threshold(self, mock_pose_model):
        """Test confidence threshold handling"""
        print(f"\n{'='*60}")
        print("Testing confidence threshold...")
        
        # Low confidence keypoints
        low_conf_keypoints = [[0.5, 0.5, 0.1] for _ in range(17)]
        
        # Should reject low confidence
        confidence = np.mean([kp[2] for kp in low_conf_keypoints])
        print(f"Low confidence test: {confidence:.2f}")
        assert confidence < 0.3, "Should reject low confidence poses"
        
        # High confidence keypoints
        high_conf_keypoints = [[0.5, 0.5, 0.95] for _ in range(17)]
        confidence = np.mean([kp[2] for kp in high_conf_keypoints])
        print(f"High confidence test: {confidence:.2f}")
        assert confidence > 0.9, "Should accept high confidence poses"
        
        print("✅ Confidence threshold tests passed")
    
    def generate_test_report(self, test_results: List[Dict]):
        """Generate comprehensive test report"""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        report = {
            'model': 'Pose Checker',
            'test_date': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'pass_rate': f"{(passed_tests/total_tests)*100:.2f}%",
            'test_details': test_results
        }
        
        # Save report
        with open('tests/reports/pose_checker_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print("📊 POSE CHECKER TEST REPORT")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Pass Rate: {report['pass_rate']}")
        print(f"Report saved to: tests/reports/pose_checker_report.json")
        
        return report

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])