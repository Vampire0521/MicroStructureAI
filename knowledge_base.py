"""
Metallurgical Knowledge Base
=============================
Maps microstructure class names to their metallurgical properties,
heat treatments, compositions, and applications.

This is a domain-knowledge lookup system — no ML involved.
All data sourced from standard metallurgy textbooks:
  - Callister: Materials Science and Engineering
  - ASM Handbook Vol. 4 (Heat Treating) & Vol. 9 (Metallography)
  - Avner: Introduction to Physical Metallurgy
"""


# ── Mapping from UHCS dataset folder names to knowledge base keys ───────────
# The UHCS dataset uses specific folder names; we map them to our richer KB.

UHCS_TO_KB_MAP = {
    "network":                    "Network_Carbides",
    "pearlite":                   "Pearlite",
    "pearlite+spheroidite":       "Pearlite_Spheroidite",
    "pearlite+widmanstatten":     "Pearlite_Widmanstatten",
    "spheroidite":                "Spheroidite",
    "spheroidite+widmanstatten":  "Spheroidite_Widmanstatten",
}

# Class names in alphabetical folder order (matches ImageFolder sorting)
CLASS_NAMES = [
    "Network_Carbides",
    "Pearlite",
    "Pearlite_Spheroidite",
    "Pearlite_Widmanstatten",
    "Spheroidite",
    "Spheroidite_Widmanstatten",
]


# ── The Knowledge Base ──────────────────────────────────────────────────────

KNOWLEDGE_BASE = {

    "Pearlite": {
        "description": "Lamellar structure of alternating ferrite (alpha) and cementite (Fe3C) plates",
        "heat_treatment": [
            "Slow furnace cooling (annealing) from austenite region (~850 deg C)",
            "Isothermal transformation at 500-700 deg C",
            "Fine pearlite forms at lower temperatures (~550 deg C), coarse pearlite at higher (~700 deg C)"
        ],
        "composition_range": {
            "Carbon": "0.2 - 0.8 wt% C",
            "Typical_grades": ["AISI 1040", "AISI 1060", "AISI 1080"],
            "Alloying": "Plain carbon or low-alloy steels"
        },
        "mechanical_properties": {
            "Hardness": "20-30 HRC (fine pearlite up to 40 HRC)",
            "UTS": "600-900 MPa",
            "Yield_Strength": "350-550 MPa",
            "Elongation": "10-25%",
            "Toughness": "Moderate"
        },
        "applications": [
            "Railway rails",
            "Wire ropes",
            "Springs (fine pearlite)",
            "Structural steels",
            "Piano wire (heavily drawn pearlite)"
        ],
        "interesting_fact": "Pearlite is named after its pearl-like iridescent appearance under optical microscope"
    },

    "Martensite": {
        "description": "Supersaturated solid solution of carbon in BCT iron - needle/lath morphology",
        "heat_treatment": [
            "Rapid quenching from austenite region (water, oil, or polymer quench)",
            "Austenitizing at 30-50 deg C above Ac3 line",
            "Lath martensite in low-carbon steels, plate martensite in high-carbon steels"
        ],
        "composition_range": {
            "Carbon": "0.1 - 1.5 wt% C",
            "Typical_grades": ["AISI 1045", "AISI 4140", "AISI 4340", "AISI 1095"],
            "Alloying": "Cr, Mo, Ni, Mn improve hardenability"
        },
        "mechanical_properties": {
            "Hardness": "50-67 HRC (as-quenched)",
            "UTS": "1200-2500 MPa",
            "Yield_Strength": "1000-2000 MPa",
            "Elongation": "1-5% (very brittle as-quenched)",
            "Toughness": "Very low - must be tempered before use"
        },
        "applications": [
            "Cutting tools",
            "Knife blades",
            "Bearings",
            "Gears (after tempering)",
            "Automotive shafts"
        ],
        "interesting_fact": "Named after Adolf Martens, a German metallurgist who first described it in 1890"
    },

    "Tempered_Martensite": {
        "description": "Martensite reheated to precipitate fine carbides, restoring ductility while retaining strength",
        "heat_treatment": [
            "Quench + Temper (Q&T) - the most common industrial heat treatment",
            "Tempering temperature 150-650 deg C depending on desired properties",
            "Low temp temper (~200 deg C) = high hardness retained",
            "High temp temper (~600 deg C) = more toughness, lower hardness"
        ],
        "composition_range": {
            "Carbon": "0.3 - 0.6 wt% C",
            "Typical_grades": ["AISI 4140", "AISI 4340", "EN24", "EN19"],
            "Alloying": "Cr-Mo and Ni-Cr-Mo steels most common"
        },
        "mechanical_properties": {
            "Hardness": "30-55 HRC (depends on tempering temperature)",
            "UTS": "900-1800 MPa",
            "Yield_Strength": "700-1500 MPa",
            "Elongation": "8-18%",
            "Toughness": "Good - best combination of strength + toughness"
        },
        "applications": [
            "Crankshafts",
            "Connecting rods",
            "High-strength bolts (Grade 10.9, 12.9)",
            "Axle shafts",
            "Landing gear components"
        ],
        "interesting_fact": "The Q&T process was known empirically by sword-makers for centuries before the science was understood"
    },

    "Bainite": {
        "description": "Acicular (needle-like) structure formed between pearlite and martensite transformation ranges",
        "heat_treatment": [
            "Isothermal transformation (austempering) at 250-550 deg C",
            "Upper bainite: 400-550 deg C (feathery appearance)",
            "Lower bainite: 250-400 deg C (acicular, similar to tempered martensite)",
            "Continuous cooling at intermediate rates"
        ],
        "composition_range": {
            "Carbon": "0.2 - 0.5 wt% C",
            "Typical_grades": ["AISI 4150", "AISI 9260", "Bainitic rail steels"],
            "Alloying": "Si retards cementite precipitation, enables carbide-free bainite"
        },
        "mechanical_properties": {
            "Hardness": "35-50 HRC",
            "UTS": "1000-1500 MPa",
            "Yield_Strength": "800-1200 MPa",
            "Elongation": "10-20%",
            "Toughness": "Excellent - especially lower bainite"
        },
        "applications": [
            "Automotive leaf springs (austempered)",
            "Modern rail steels",
            "Wear-resistant plates",
            "Fasteners",
            "Agricultural equipment"
        ],
        "interesting_fact": "Named after Edgar Bain, who first described the transformation in 1933"
    },

    "Spheroidite": {
        "description": "Globular/spheroidal cementite (Fe3C) particles dispersed in a continuous ferrite matrix",
        "heat_treatment": [
            "Spheroidizing anneal: prolonged heating just below Ac1 (~700 deg C) for 15-25 hours",
            "Or cyclic heating above and below Ac1",
            "Converts lamellar (pearlitic) cementite into spheroidal form"
        ],
        "composition_range": {
            "Carbon": "0.6 - 1.2 wt% C",
            "Typical_grades": ["AISI 1080", "AISI 52100", "Tool steels before hardening"],
            "Alloying": "High-carbon steels and bearing steels"
        },
        "mechanical_properties": {
            "Hardness": "15-22 HRC (softest microstructure for a given composition)",
            "UTS": "400-600 MPa",
            "Yield_Strength": "250-400 MPa",
            "Elongation": "20-35%",
            "Toughness": "Very good - excellent machinability"
        },
        "applications": [
            "Pre-machining condition for tool steels",
            "Bearing steel blanks (AISI 52100) before final heat treatment",
            "Cold forming and deep drawing stock",
            "Wire drawing stock"
        ],
        "interesting_fact": "Spheroidite has the lowest hardness of any microstructure for a given carbon content"
    },

    "Widmanstatten": {
        "description": "Large ferrite/cementite plates growing along specific crystallographic planes of prior austenite",
        "heat_treatment": [
            "Forms during moderately fast cooling from high austenite temperatures",
            "Coarse prior austenite grain size promotes formation",
            "Typically undesirable — indicates improper heat treatment (overheating)"
        ],
        "composition_range": {
            "Carbon": "0.1 - 0.4 wt% C (ferrite type) or 0.8-1.5 wt% C (cementite type)",
            "Typical_grades": ["Low-carbon structural steels", "Weld HAZ zones"],
            "Alloying": "Plain carbon and low-alloy steels; also seen in cast irons"
        },
        "mechanical_properties": {
            "Hardness": "15-25 HRC",
            "UTS": "450-600 MPa",
            "Yield_Strength": "300-400 MPa",
            "Elongation": "10-20%",
            "Toughness": "Poor - plates act as crack initiation paths"
        },
        "applications": [
            "Generally UNDESIRABLE - indicates need for normalizing treatment",
            "Found in as-cast structures and overheated weld HAZ",
            "Corrected by normalizing at proper temperature",
            "Historically seen in meteorites (Widmanstatten patterns)"
        ],
        "interesting_fact": "First discovered in iron meteorites by Alois von Widmanstatten in 1808"
    },

    "Network_Carbides": {
        "description": "Continuous cementite (Fe3C) network along prior austenite grain boundaries",
        "heat_treatment": [
            "Slow cooling of hypereutectoid steel (>0.76% C) from above Acm line",
            "Pro-eutectoid cementite precipitates preferentially along grain boundaries",
            "Undesirable — creates continuous brittle network"
        ],
        "composition_range": {
            "Carbon": "0.8 - 1.5 wt% C",
            "Typical_grades": ["AISI 1095", "AISI 52100", "W1 tool steel"],
            "Alloying": "Hypereutectoid plain carbon and alloy steels"
        },
        "mechanical_properties": {
            "Hardness": "25-35 HRC",
            "UTS": "Variable - depends on network continuity",
            "Yield_Strength": "Variable",
            "Elongation": "Very low - grain boundary embrittlement",
            "Toughness": "Very poor - catastrophic brittle fracture risk"
        },
        "applications": [
            "UNDESIRABLE microstructure - must be eliminated before use",
            "Must be broken up by spheroidizing anneal or normalizing",
            "Indicates improper slow cooling of high-carbon steel",
            "If found in service parts: reject / re-heat-treat"
        ],
        "interesting_fact": "Network carbides are one of the most common causes of premature failure in high-carbon steel components"
    },

    "Pearlite_Spheroidite": {
        "description": "Mixed microstructure: partially spheroidized pearlite — some lamellar regions remain alongside spheroidal cementite",
        "heat_treatment": [
            "Incomplete spheroidizing anneal (insufficient time or temperature)",
            "Or early stages of spheroidization from a pearlitic starting structure",
            "Intermediate between full pearlite and full spheroidite"
        ],
        "composition_range": {
            "Carbon": "0.4 - 1.0 wt% C",
            "Typical_grades": ["AISI 1060", "AISI 1080", "Medium to high carbon steels"],
            "Alloying": "Plain carbon steels undergoing spheroidizing"
        },
        "mechanical_properties": {
            "Hardness": "18-28 HRC (between pearlite and spheroidite)",
            "UTS": "500-750 MPa",
            "Yield_Strength": "300-500 MPa",
            "Elongation": "15-28%",
            "Toughness": "Moderate to good"
        },
        "applications": [
            "Transitional microstructure during spheroidizing heat treatment",
            "Can be acceptable for some cold-forming operations",
            "Often indicates the spheroidizing process needs more time"
        ],
        "interesting_fact": "The degree of spheroidization can be quantified by measuring the aspect ratio of cementite particles"
    },

    "Spheroidite_Widmanstatten": {
        "description": "Mixed microstructure: spheroidal cementite coexisting with Widmanstatten ferrite/cementite plates",
        "heat_treatment": [
            "Complex thermal history — possibly overheated then slowly cooled and partially spheroidized",
            "Or Widmanstatten structure that was subsequently given a partial spheroidizing treatment",
            "Unusual combination — indicates non-standard processing"
        ],
        "composition_range": {
            "Carbon": "0.3 - 1.0 wt% C",
            "Typical_grades": ["Various carbon and low-alloy steels"],
            "Alloying": "Plain carbon or low-alloy steels with complex thermal history"
        },
        "mechanical_properties": {
            "Hardness": "18-28 HRC",
            "UTS": "450-650 MPa",
            "Yield_Strength": "280-420 MPa",
            "Elongation": "12-22%",
            "Toughness": "Poor to moderate - Widmanstatten plates degrade toughness"
        },
        "applications": [
            "Generally indicates non-optimized processing",
            "Requires normalizing + proper spheroidizing to correct",
            "May be found in forgings with uncontrolled cooling"
        ],
        "interesting_fact": "This mixed microstructure is rarely intentionally produced — it typically results from processing errors"
    },

    "Pearlite_Widmanstatten": {
        "description": "Mixed microstructure: pearlitic regions coexisting with Widmanstatten ferrite/cementite plates",
        "heat_treatment": [
            "Cooling from high austenitizing temperature with intermediate rate",
            "Partial transformation in the pearlite range followed by Widmanstatten formation",
            "Common in as-cast or as-forged steel with poor temperature control"
        ],
        "composition_range": {
            "Carbon": "0.30 - 0.80 wt%",
            "Manganese": "0.50 - 1.00 wt%",
            "Silicon": "0.15 - 0.35 wt%"
        },
        "mechanical_properties": {
            "Hardness": "200 - 350 HV (varies with ratio)",
            "Tensile Strength": "550 - 850 MPa",
            "Ductility": "Reduced compared to fully pearlitic steel",
            "Impact Toughness": "Poor - Widmanstatten plates act as crack initiators"
        },
        "applications": [
            "Not intentionally produced - considered a process defect",
            "Indicates need for normalizing heat treatment",
            "Found in weld heat-affected zones (HAZ)"
        ],
        "interesting_fact": "The presence of Widmanstatten alongside pearlite indicates cooling from an excessively high austenitizing temperature"
    },
}


def get_knowledge(class_name: str) -> dict:
    """
    Retrieve metallurgical knowledge for a given microstructure class.

    Args:
        class_name: Microstructure name (e.g., 'Pearlite', 'Martensite')

    Returns:
        Dictionary with description, heat_treatment, composition_range,
        mechanical_properties, applications, and interesting_fact.
    """
    if class_name in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[class_name]

    # Try mapping from UHCS folder name
    mapped = UHCS_TO_KB_MAP.get(class_name.lower())
    if mapped and mapped in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[mapped]

    return {"error": f"Unknown microstructure class: {class_name}"}


def get_class_name_from_index(idx: int) -> str:
    """Convert class index to human-readable class name."""
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"Unknown (index {idx})"


def format_report(class_name: str, confidence: float, all_probs: dict = None) -> str:
    """
    Generate a formatted metallurgical report card.

    Args:
        class_name: Predicted microstructure class
        confidence: Prediction confidence (0-1)
        all_probs: Dict of class_name → probability (optional)

    Returns:
        Formatted report string.
    """
    kb = get_knowledge(class_name)
    if "error" in kb:
        return f"Error: {kb['error']}"

    lines = []
    lines.append("=" * 62)
    lines.append("       MICROSTRUCTURE ANALYSIS REPORT")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"  Identified Microstructure:  {class_name.replace('_', ' ').upper()}")
    lines.append(f"  Confidence:                 {confidence * 100:.1f}%")
    lines.append(f"  Description:                {kb['description']}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  PROBABLE HEAT TREATMENT:")
    for ht in kb["heat_treatment"]:
        lines.append(f"    > {ht}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  PROBABLE COMPOSITION:")
    comp = kb["composition_range"]
    lines.append(f"    Carbon:         {comp['Carbon']}")
    lines.append(f"    Typical Grades: {', '.join(comp['Typical_grades'])}")
    lines.append(f"    Alloying:       {comp['Alloying']}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  EXPECTED MECHANICAL PROPERTIES:")
    props = kb["mechanical_properties"]
    for key, val in props.items():
        lines.append(f"    {key.replace('_', ' '):<18} {val}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  TYPICAL APPLICATIONS:")
    for app in kb["applications"]:
        lines.append(f"    * {app}")
    lines.append("")

    if "interesting_fact" in kb:
        lines.append("-" * 62)
        lines.append(f"  [TIP] Fun Fact: {kb['interesting_fact']}")
        lines.append("")

    if all_probs:
        lines.append("-" * 62)
        lines.append("  ALL CLASS PROBABILITIES:")
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_probs:
            bar = "#" * int(prob * 30)
            lines.append(f"    {name:<28} {prob * 100:5.1f}% {bar}")
        lines.append("")

    lines.append("=" * 62)
    return "\n".join(lines)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Metallurgical Knowledge Base")
    print(f"Total classes: {len(KNOWLEDGE_BASE)}")
    print(f"Classes: {', '.join(KNOWLEDGE_BASE.keys())}")
    print()

    # Demo report
    report = format_report("Pearlite", 0.942, {
        "Pearlite": 0.942,
        "Spheroidite": 0.031,
        "Martensite": 0.012,
        "Network_Carbides": 0.005,
        "Widmanstatten": 0.004,
        "Pearlite_Spheroidite": 0.004,
        "Spheroidite_Widmanstatten": 0.002,
    })
    print(report)
