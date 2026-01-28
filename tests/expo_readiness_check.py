"""
MedGuard AI - Expo Readiness Validation Script
Checks all requirements before expo demonstration
"""

import os
import sys
import pickle
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_mark(passed):
    return f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"

def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}   MedGuard AI - Expo Readiness Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    total_tests = 0
    passed_tests = 0
    
    # ========================================
    # 1. FILE EXISTENCE CHECKS
    # ========================================
    print(f"{YELLOW}[FILE] Checking Required Files...{RESET}")
    
    required_files = {
        'models/primary_model.pkl': 'Primary AI model',
        'models/risk_predictor.pkl': 'Failure risk predictor',
        'models/shap_cache.pkl': 'SHAP explanations cache',
        'data/heart_disease.csv': 'UCI dataset',
        'src/streamlit_app.py': 'Main application',
        'src/cme_explainer.py': 'CME engine',
        'src/model_trainer.py': 'Training pipeline',
        'src/failure_risk_predictor.py': 'Risk predictor module',
        'src/demo_config.py': 'Demo configuration',
        'README.md': 'Documentation',
        'requirements.txt': 'Dependencies'
    }
    
    for filepath, description in required_files.items():
        total_tests += 1
        exists = os.path.exists(filepath)
        if exists:
            passed_tests += 1
        print(f"  {check_mark(exists)} {description:.<45} {filepath}")
    
    # ========================================
    # 2. FUNCTIONALITY CHECKS
    # ========================================
    print(f"\n{YELLOW}[FUNC] Checking Functionality...{RESET}")

    # Check model loading
    total_tests += 1
    try:
        import sys
        sys.path.insert(0, 'src')
        from model_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        trainer.load_model('models/primary_model.pkl')
        if trainer.is_trained:
            print(f"  {check_mark(True)} Primary model loads successfully")
            passed_tests += 1
        else:
            print(f"  {check_mark(False)} Primary model not trained after loading")
    except Exception as e:
        print(f"  {check_mark(False)} Primary model loading FAILED: {str(e)[:50]}")

    # Check risk predictor loading
    total_tests += 1
    try:
        risk_pred = ModelTrainer.load_risk_predictor()
        if risk_pred is not None:
            print(f"  {check_mark(True)} Risk predictor loads successfully")
            passed_tests += 1
        else:
            print(f"  {check_mark(False)} Risk predictor returned None")
    except Exception as e:
        print(f"  {check_mark(False)} Risk predictor loading FAILED: {str(e)[:50]}")

    # Check SHAP cache (skip detailed validation)
    total_tests += 1
    shap_exists = os.path.exists('models/shap_cache.pkl')
    if shap_exists:
        print(f"  {check_mark(True)} SHAP cache file exists")
        passed_tests += 1
    else:
        print(f"  {check_mark(False)} SHAP cache file missing")
    
    # Check demo config imports
    total_tests += 1
    try:
        sys.path.insert(0, 'src')
        from demo_config import DEMO_NARRATIVE, get_demo_patient_index
        assert len(DEMO_NARRATIVE) == 8, "Demo should have 8 steps"
        print(f"  {check_mark(True)} Demo configuration valid (8 steps)")
        passed_tests += 1
    except Exception as e:
        print(f"  {check_mark(False)} Demo configuration FAILED: {str(e)}")
    
    # ========================================
    # 3. UI/UX CHECKS
    # ========================================
    print(f"\n{YELLOW}[UI]   Checking UI Elements...{RESET}")
    
    # Check for MedGuard branding (not CME)
    total_tests += 1
    try:
        with open('src/streamlit_app.py', 'r', encoding='utf-8') as f:
            app_content = f.read()
        has_medguard = 'MedGuard AI' in app_content or 'MedGuard' in app_content
        no_cme_title = 'Contrastive Mistake Explainer' not in app_content or 'CME' in app_content
        # CME is okay internally, just not as user-facing title
        
        if has_medguard:
            print(f"  {check_mark(True)} MedGuard AI branding present")
            passed_tests += 1
        else:
            print(f"  {check_mark(False)} MedGuard AI branding missing")
    except Exception as e:
        print(f"  {check_mark(False)} Could not read app file: {str(e)}")
    
    # Check for demo mode toggle
    total_tests += 1
    has_demo = 'demo_active' in app_content and 'Enable Demo Mode' in app_content
    print(f"  {check_mark(has_demo)} Demo mode toggle exists")
    if has_demo:
        passed_tests += 1
    
    # Check for risk warning display
    total_tests += 1
    has_risk = 'HIGH FAILURE RISK' in app_content or 'Failure Risk' in app_content
    print(f"  {check_mark(has_risk)} Risk warning display present")
    if has_risk:
        passed_tests += 1
    
    # ========================================
    # 4. DOCUMENTATION CHECKS
    # ========================================
    print(f"\n{YELLOW}[DOC]  Checking Documentation...{RESET}")
    
    # Check README has expo section
    total_tests += 1
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            readme = f.read()
        has_expo = 'Expo' in readme or '60-Second Demo' in readme
        print(f"  {check_mark(has_expo)} README has expo section")
        if has_expo:
            passed_tests += 1
    except Exception as e:
        print(f"  {check_mark(False)} Could not read README: {str(e)}")
    
    # Check for poster file
    total_tests += 1
    has_poster = os.path.exists('docs/EXPO_POSTER.md')
    print(f"  {check_mark(has_poster)} Expo poster content exists")
    if has_poster:
        passed_tests += 1
    
    # ========================================
    # 5. PERFORMANCE CHECKS
    # ========================================
    print(f"\n{YELLOW}[PERF] Checking Performance...{RESET}")
    
    # Check file sizes (should be reasonable)
    total_tests += 1
    try:
        model_size = os.path.getsize('models/primary_model.pkl') / (1024 * 1024)  # MB
        reasonable_size = model_size < 100  # Should be under 100MB
        print(f"  {check_mark(reasonable_size)} Model size reasonable ({model_size:.1f} MB)")
        if reasonable_size:
            passed_tests += 1
    except FileNotFoundError:
        print(f"  {check_mark(False)} Model file not found for size check")
    
    # Check no external network dependencies in app
    total_tests += 1
    has_requests = 'requests.get' in app_content or 'urllib.request' in app_content
    offline_ready = not has_requests
    print(f"  {check_mark(offline_ready)} App works offline (no external API calls)")
    if offline_ready:
        passed_tests += 1
    
    # ========================================
    # FINAL REPORT
    # ========================================
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}   FINAL SCORE: {passed_tests}/{total_tests} Tests Passed{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    if passed_tests == total_tests:
        print(f"{GREEN}[SUCCESS] EXPO READY! All systems go!{RESET}\n")
        return 0
    elif passed_tests >= total_tests * 0.8:
        print(f"{YELLOW}[WARNING] MOSTLY READY: {total_tests - passed_tests} issues to address{RESET}\n")
        return 1
    else:
        print(f"{RED}[ERROR] NOT READY: Critical issues need fixing{RESET}\n")
        return 2

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
