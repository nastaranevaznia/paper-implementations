"""Synthetic patient-complaint dataset generator (English texts).

The manuscript states its data were *synthetically generated to simulate
real-world patient complaint patterns* (no real patient data).  This module
provides an equivalent generator: template-based complaint sentences for the
four Table-1 categories (30 base templates each), sampled with the natural
class priors, enriched with optional closing sentences and realistic noise
(cross-category distractor sentences, as real complaints often mention
several aspects at once).
"""

import numpy as np
import pandas as pd

from .config import CATEGORIES, CLASS_PRIORS

# Category 0: Communication problems (staff behavior / attitude)
_T0 = [
    "The nurse was rude and dismissive when I asked about my medication.",
    "The doctor did not listen to my concerns and interrupted me constantly.",
    "Staff attitude was unacceptable; nobody explained the waiting time.",
    "The receptionist shouted at my elderly mother in front of everyone.",
    "I felt humiliated by the way the physician spoke to me.",
    "Nobody answered my questions and the staff seemed annoyed.",
    "The surgeon was arrogant and refused to explain the risks.",
    "Communication was terrible; I never knew what would happen next.",
    "The triage nurse ignored my pain and told me to wait silently.",
    "Staff were gossiping instead of attending to patients in the ward.",
    "The doctor spent less than a minute with me and walked away.",
    "I asked for an interpreter but the staff mocked my request.",
    "The nurse rolled her eyes when I described my symptoms.",
    "The physician used offensive language during my examination.",
    "No one introduced themselves or told me their role in my care.",
    "The staff spoke about me as if I was not in the room.",
    "My call button was ignored for over an hour last night.",
    "The doctor laughed when I mentioned my fear of the procedure.",
    "Nurses scolded me for asking for pain relief twice.",
    "The receptionist hung up on me three times when I called.",
    "I was scolded for crying during a painful dressing change.",
    "The consultant spoke only to my husband and never to me.",
    "Staff refused to speak slowly even though I am hard of hearing.",
    "The night nurse threatened to remove my visitors for no reason.",
    "I was blamed for being sick when I asked for help walking.",
    "The doctor discussed my private condition loudly in the corridor.",
    "Nobody told me the test results; I heard them by accident.",
    "The midwife was rough and showed no empathy during labor.",
    "Security guards treated patients like criminals at the entrance.",
    "The pharmacist refused to explain how to take my pills.",
]

# Category 1: Diagnosis / Treatment issues (misdiagnosis / treatment errors)
_T1 = [
    "I was misdiagnosed twice before they found the real infection.",
    "The prescribed dosage was wrong and caused severe side effects.",
    "My fracture was missed on the X-ray and discovered weeks later.",
    "The surgery was performed on the wrong site and had to be repeated.",
    "They confused my lab results with another patient and mistreated me.",
    "My allergy to penicillin was ignored and I went into shock.",
    "The treatment did not follow the guidelines and my condition worsened.",
    "A delayed cancer diagnosis cost me months of valuable treatment time.",
    "The anesthesiologist made an error and I woke up during surgery.",
    "I received another patient's medication because labels were swapped.",
    "The follow-up scan was never ordered and the tumor kept growing.",
    "Wrong insulin dose sent me to the emergency room the same night.",
    "My appendicitis was dismissed as stomach flu until it ruptured.",
    "The dentist pulled the wrong tooth and damaged the healthy one.",
    "I was given blood from the wrong group and nearly died.",
    "The radiologist missed the blood clot clearly visible on the scan.",
    "My baby received an adult dose of vaccine by mistake.",
    "The doctor prescribed a drug that interacts badly with my heart pills.",
    "A sponge was left inside me after the operation.",
    "My stroke symptoms were sent home as migraine twice.",
    "The biopsy sample was lost and my treatment was delayed months.",
    "I developed an infection because instruments were not sterilized.",
    "The eye injection was done incorrectly and damaged my vision.",
    "Nobody noticed my broken ribs on three separate visits.",
    "The contraceptive implant was placed wrongly and failed.",
    "My thyroid condition went untreated for years despite clear tests.",
    "The cast was too tight and cut off circulation to my foot.",
    "I was discharged with untreated pneumonia and collapsed at home.",
    "The allergy test was misread and I was exposed to the allergen.",
    "Chemotherapy dosage was miscalculated and poisoned my kidneys.",
]

# Category 2: Management problems (facilities / appointments / medication)
_T2 = [
    "I waited four months for a specialist appointment that was cancelled.",
    "The ward was dirty, overcrowded and the bathroom had no water.",
    "My appointment was postponed three times without any explanation.",
    "The pharmacy had no stock of my prescribed medicine for weeks.",
    "There were no available beds and I slept on a chair overnight.",
    "The emergency room wait took nine hours with no triage update.",
    "Hospital parking and admission queues are completely unmanaged.",
    "My medical records were lost and I had to redo all the tests.",
    "The air conditioning was broken and the ward was unbearable.",
    "Discharge papers took two days while I kept paying for the bed.",
    "Online booking never works and the phone line is always busy.",
    "They overcharged me for services I never received.",
    "The elevator was broken and my wheelchair could not reach the clinic.",
    "Visiting hours change daily and nobody informs the families.",
    "The hospital food was expired and several patients got sick.",
    "My surgery was cancelled twice because the room was double-booked.",
    "There is only one toilet for forty patients on this floor.",
    "Ambulance took two hours to arrive after my father's heart attack.",
    "The billing office sent my invoice to a stranger's address.",
    "Wheelchairs are all broken and no porter is ever available.",
    "The clinic opens late every morning with a huge crowd outside.",
    "My referral letter expired while waiting for an authorization code.",
    "The intensive care unit had no free ventilator when I needed one.",
    "Hospital corridors are filthy and smell of smoke.",
    "I paid for a private room but was placed in a shared ward.",
    "The laboratory lost my blood sample twice in one week.",
    "No doctor was present during the entire night shift.",
    "The hospital bus never follows the announced timetable.",
    "Discharge medicines were missing half the prescribed items.",
    "The complaint office itself never answers its own phone.",
]

# Category 3: Responsibility concerns (staff unaccountability)
_T3 = [
    "Nobody took responsibility for the mistake in my treatment plan.",
    "After the incident, no manager would meet me or apologize.",
    "My formal complaint was ignored for six months without a reply.",
    "Each department blamed the other and nothing was ever fixed.",
    "The hospital refused to share the incident report with my family.",
    "Staff denied everything even though witnesses confirmed my story.",
    "No one followed up after my surgery complications were reported.",
    "The supervisor promised an investigation that never happened.",
    "They pressured me to withdraw my complaint against the clinic.",
    "Accountability is zero; the same error happened to my neighbor.",
    "Risk management closed my case without interviewing me.",
    "I was discharged with complications and nobody accepted the blame.",
    "The head nurse said mistakes happen and walked away.",
    "My request for a second opinion was blocked by the same doctor.",
    "The hospital lawyer threatened me for posting an honest review.",
    "No incident number was given so my case officially never existed.",
    "The ethics committee never replied to any of my three letters.",
    "Doctors changed my file after I complained about the error.",
    "The clinic deleted my negative feedback from their website.",
    "I was transferred to another hospital just to silence my complaint.",
    "Management admitted the fault verbally but denied it in writing.",
    "The same faulty machine injured patients for months with no recall.",
    "Nobody supervised the trainee who performed my procedure alone.",
    "The hospital blamed my age instead of their missed diagnosis.",
    "My compensation claim has been pending for over two years.",
    "The director refused to meet families of affected patients.",
    "Staff were never retrained after the serious safety incident.",
    "The pharmacy denied the dispensing error despite the receipt proof.",
    "Internal review cleared everyone without speaking to a single patient.",
    "They offered me money quietly instead of fixing the real problem.",
]

_TEMPLATES = [_T0, _T1, _T2, _T3]
DISTRACTORS = [t for cat in _TEMPLATES for t in cat]

# Generic closing sentences appended to ~half of the complaints for variety.
_CLOSERS = [
    "I expect a formal written response.",
    "This happened during my visit last week.",
    "I have witnesses who can confirm this.",
    "Please investigate this matter urgently.",
    "This is not the first time this has happened to me.",
    "I am filing this complaint on behalf of my father.",
    "I have photos and documents as evidence.",
    "I want an apology and corrective action.",
    "The whole experience was extremely distressing.",
    "I trusted this hospital and I feel betrayed.",
    "Someone must be held accountable for this.",
    "I will escalate this to the health ministry if needed.",
    "My family is very upset about this incident.",
    "This should never happen to any patient again.",
]

DEPARTMENTS = ["Emergency", "Surgery", "Internal", "Pediatrics", "Radiology"]


def generate_dataset(n, seed=0, n_centers=5, n_weeks=12, noise=0.15,
                     closer_prob=0.5):
    """Generate ``n`` synthetic complaints with metadata.

    Returns a DataFrame with columns: text, true_label, label_name,
    center_id, week, department.
    """
    rng = np.random.default_rng(seed)
    labels = rng.choice(4, size=n, p=CLASS_PRIORS)
    texts = []
    for y in labels:
        sent = str(rng.choice(_TEMPLATES[y]))
        if rng.random() < noise:
            # realistic cross-aspect distractor from another category
            others = [i for i in range(4) if i != y]
            sent += " " + str(rng.choice(_TEMPLATES[rng.choice(others)]))
        if rng.random() < closer_prob:
            sent += " " + str(rng.choice(_CLOSERS))
        texts.append(sent)
    df = pd.DataFrame({
        "text": texts,
        "true_label": labels,
        "label_name": [CATEGORIES[y] for y in labels],
        "center_id": rng.integers(1, n_centers + 1, size=n),
        "week": rng.integers(1, n_weeks + 1, size=n),
        "department": rng.choice(DEPARTMENTS, size=n),
    })
    return df
