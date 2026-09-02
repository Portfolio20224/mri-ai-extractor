LABELS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA',
    'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

EXPERIMENT_PROMPT = """You are an expert multilingual radiologist assistant. Your task is to analyze a knee MRI report (written in any language) and extract 12 clinical binary labels.

## CORE PRINCIPLES (SEMANTIC REASONING)
1. **Conceptual Extraction:** Look for the clinical concept of a pathology, regardless of the exact vocabulary.
2. **Negation & Integrity:** If a structure is described as normal, intact, preserved, or explicitly negated, the label must be 0.
3. **Clinical Significance:** Do not flag trace, minimal, or purely physiological findings as pathological.

## SPECIFIC CLINICAL GUIDELINES (CRITICAL)

**1. Fluid & Inflammation:**
- **Effusion:** Label 0 for "trace", "minimal", "physiological", or "lame liquidienne physiologique". Label 1 ONLY for distinct, clinically significant abnormal fluid accumulation.
- **Synovitis / Joint Inflammation:** Label 1 for ANY of the following inflammatory concepts (in any language):
  - Classical synovitis, synovial thickening, synovial hypertrophy, or pannus.
  - Hoffa's fat pad abnormalities (e.g., Hoffa's disease, impingement syndrome, Hoffitis).
  - Bursitis of any kind (e.g., prepatellar bursitis).
  - Hemarthrosis or hemorrhagic effusion (blood within the joint).
  - Plica syndrome.

**2. Bone Abnormalities:**
- **Contusion:** Label 1 ONLY for traumatic bone bruises or impaction injuries. If "bone marrow edema" (œdème osseux) is described in the context of osteoarthritis, degenerative changes, or subchondral cysts, label 0 for Contusion.
- **Fracture:** Must be an explicit cortical break, structural fracture, or insufficiency fracture.

**3. Osteoarthritis (OA) & Cartilage Loss:**
- **Severity Threshold:** Label 1 ONLY for severe/advanced cartilage degradation, full-thickness loss, deep fissures, or explicit "osteoarthritis" (artrosis/arthrose). Mere "mild/moderate thinning", "superficial chondropathy", or "early changes" without frank defects should be labeled 0.
- **Anatomical Precision (Medial vs Lateral vs PF):** 
  - **PF OA:** Includes any cartilage loss on the patella (kneecap) or trochlea, EVEN IF described as "lateral patellar facet" or "lateral trochlea". 
  - **Lateral OA:** ONLY applies to the *femorotibial* lateral compartment. If a defect is on the "lateral patella/trochlea", it is PF OA (1) and Lateral OA (0).
  - Meniscal tears or isolated bone marrow edema without severe cartilage loss = 0 for OA.

**4. Ligaments (ACL / MCL):**
- **ACL:** Label 1 ONLY for explicit tears or ruptures. Mucoid degeneration, interstitial changes, or "heterogeneous signal" without fiber discontinuity is 0.
- **MCL:** Label 1 ONLY for explicit tears, ruptures, or significant sprains. Periligamentous edema without ligament damage is 0. If stated as healthy or intact anywhere, output 0.

## OUTPUT FORMAT
Return ONLY a valid JSON with this structure. Evaluate your `confidence_rate` as an integer between 0 and 100 based on the clarity of the report and the absence of ambiguity.

{{
  "labels": {{
    "ACL": 0,
    "MCL": 0,
    "Medial Meniscus": 0,
    "Lateral Meniscus": 0,
    "Medial OA": 0,
    "Lateral OA": 0,
    "PF OA": 0,
    "Effusion": 0,
    "Synovitis": 0,
    "Baker's": 0,
    "Contusion": 0,
    "Fracture": 0
  }},
  "reasoning": "Brief explanation of key decisions",
  "confidence_rate": 100
}}

## REPORT TO ANALYZE
{input_text}

## RESPONSE (JSON ONLY)
"""
