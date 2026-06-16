import os
import sys

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.honeypot_detector import detect_trap, classify_title, check_boilerplate_description

def test_classify_title():
    assert classify_title("Senior AI Engineer") == "AI/ML"
    assert classify_title("HR Manager") == "HR/Recruiting"
    assert classify_title("Mechanical Designer") == "Mechanical Engineering"
    assert classify_title("Accountant") == "Accounting"
    assert classify_title("Unknown Title") == "Other"

def test_check_boilerplate_description():
    desc = "Customer support team lead at a SaaS product. Managed a team..."
    assert check_boilerplate_description(desc) == "Support/Operations"
    
    desc_clean = "Mechanical engineering design role at a hardware-product company."
    assert check_boilerplate_description(desc_clean) == "Mechanical Engineering"
    
    assert check_boilerplate_description("Completely custom role description") is None

def test_detect_trap_genuine():
    genuine_candidate = {
        "candidate_id": "CAND_GENUINE",
        "profile": {
            "current_title": "Senior AI Engineer",
            "years_of_experience": 6.5,
            "summary": "AI researcher specializing in retrieval systems and machine learning."
        },
        "skills": [
            {"name": "Python"},
            {"name": "Vector Search"},
            {"name": "NLP"}
        ],
        "career_history": [
            {
                "title": "AI Engineer",
                "description": "Designed search ranking system using embeddings."
            }
        ]
    }
    
    trap_score, trap_reason = detect_trap(genuine_candidate)
    assert trap_score == 0.0
    assert "Consistent profile" in trap_reason

def test_detect_trap_decoy_boilerplate():
    # Marketing manager summary decoy boilerplate
    decoy_candidate = {
        "candidate_id": "CAND_DECOY_1",
        "profile": {
            "current_title": "HR Manager",
            "years_of_experience": 8.0,
            "summary": "Lately I've been curious about how AI tools could augment my work \u2014 I've experimented with ChatGPT..."
        },
        "skills": [
            {"name": "HR Operations"},
            {"name": "NLP"},
            {"name": "Vector Search"},
            {"name": "Embeddings"}
        ],
        "career_history": [
            {
                "title": "HR Lead",
                "description": "Managed employee onboarding and payroll."
            }
        ]
    }
    
    trap_score, trap_reason = detect_trap(decoy_candidate)
    assert trap_score >= 0.4
    assert "Generic summary template detected" in trap_reason

def test_detect_trap_decoy_stuffing():
    # Non-AI title stuffed with AI skills
    decoy_candidate = {
        "candidate_id": "CAND_DECOY_2",
        "profile": {
            "current_title": "Accountant",
            "years_of_experience": 5.0,
            "summary": "Experienced accountant specializing in corporate finance."
        },
        "skills": [
            {"name": "Accounting"},
            {"name": "NLP"},
            {"name": "Vector Search"},
            {"name": "Embeddings"},
            {"name": "MLOps"}
        ],
        "career_history": [
            {
                "title": "Senior Accountant",
                "description": "Handled monthly bookkeeping and tax filings."
            }
        ]
    }
    
    trap_score, trap_reason = detect_trap(decoy_candidate)
    assert trap_score >= 0.4
    assert "Non-AI Title" in trap_reason
