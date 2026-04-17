"""
Defect Knowledge Bases
=======================
Two knowledge bases for the Enhanced Steel Diagnostic Tool:

1. SURFACE_DEFECTS — 6 classes from the NEU Surface Defect Dataset
   (Rolled-in Scale, Patches, Crazing, Pitted Surface, Inclusion, Scratches)
   Each entry: description, severity, causes, remedies, affected_properties

2. MICROSTRUCTURAL_DEFECTS — defect flags triggered when the microstructure
   classifier identifies a problematic structure (Network Carbides,
   Widmanstatten, Decarburization, Grain Coarsening)
   Each entry: is_defect, problem, causes, remedies

Sources: ASM Handbook Vol. 1 & 9, Callister, plant metallurgy practice
"""


# ── NEU Dataset folder → KB key mapping ────────────────────────────────────

NEU_TO_KB_MAP = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted_Surface",
    "rolled-in_scale": "Rolled_in_Scale",
    "scratches": "Scratches",
}

SURFACE_CLASS_NAMES = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted_Surface",
    "Rolled_in_Scale",
    "Scratches",
]


# ── Surface Defect Knowledge Base ──────────────────────────────────────────

SURFACE_DEFECTS = {

    "Rolled_in_Scale": {
        "description": (
            "Oxide scale from high-temperature oxidation gets pressed into "
            "the steel surface during hot rolling"
        ),
        "severity": "Moderate to High",
        "causes": [
            "Excessive oxidation in the reheating furnace",
            "Furnace temperature too high or soak time too long",
            "Poor descaling - high-pressure water jets not functioning properly",
            "Scale breaker rolls worn out or improperly set"
        ],
        "remedies": [
            "Optimize reheating furnace atmosphere (reduce excess air)",
            "Control furnace temperature within 1150-1250 deg C range",
            "Maintain descaling nozzle pressure above 150 bar",
            "Regular inspection of scale breaker rolls",
            "Use controlled atmosphere or induction heating"
        ],
        "affected_properties": [
            "Surface finish - rough, pitted appearance",
            "Fatigue life reduced due to surface stress concentrators",
            "Poor paintability and coating adhesion",
            "Rejection in automotive exposed panels"
        ],
        "industry_standards": "Unacceptable per ASTM A568 surface quality requirements"
    },

    "Patches": {
        "description": (
            "Irregular dark/light patches on the surface caused by non-uniform "
            "cooling or finishing temperature variation"
        ),
        "severity": "Low to Moderate",
        "causes": [
            "Non-uniform finishing temperature across strip width",
            "Uneven cooling on the run-out table",
            "Variable roll gap causing thickness variation",
            "Improper laminar cooling header settings"
        ],
        "remedies": [
            "Calibrate run-out table cooling headers uniformly",
            "Ensure uniform finishing mill temperature (within +/-15 deg C across width)",
            "Profile control of work rolls to ensure even gap",
            "Edge masking in cooling section for thermal crown control"
        ],
        "affected_properties": [
            "Non-uniform mechanical properties across the sheet",
            "Inconsistent hardness - causes problems in forming",
            "Visual defect - rejected for exposed applications"
        ],
        "industry_standards": "May violate ASTM A568 or customer-specific surface specs"
    },

    "Crazing": {
        "description": (
            "Network of fine interconnected cracks on the surface, "
            "resembling dried mud or cracked paint"
        ),
        "severity": "High",
        "causes": [
            "Thermal fatigue of work rolls transferred to strip surface",
            "Excessive rolling force causing surface shear stress",
            "Work roll surface degradation (fire cracks on roll)",
            "High friction between roll and strip surface"
        ],
        "remedies": [
            "Regular redressing of work rolls (CNC grinding)",
            "Monitor roll surface condition with eddy current testing",
            "Optimize roll cooling to prevent thermal fatigue",
            "Use high-speed steel (HSS) or carbide-enhanced rolls",
            "Apply proper roll lubrication"
        ],
        "affected_properties": [
            "Severe surface integrity loss",
            "Crack propagation under fatigue loading",
            "Corrosion initiation sites",
            "Rejected for any structural or aesthetic application"
        ],
        "industry_standards": "Rejected per all surface quality standards"
    },

    "Pitted_Surface": {
        "description": (
            "Small pits or cavities scattered across the surface, "
            "from chemical attack or mechanical indentation"
        ),
        "severity": "Moderate",
        "causes": [
            "Over-pickling in HCl or H2SO4 acid bath",
            "Pickling acid concentration too high or temperature too high",
            "Pickling time too long (strip speed too slow)",
            "Residual acid not properly rinsed - continued attack",
            "Pitting corrosion during storage in humid conditions"
        ],
        "remedies": [
            "Control acid bath concentration (HCl: 5-15%, temp: 60-85 deg C)",
            "Optimize strip speed through pickling line",
            "Ensure thorough rinsing and drying after pickling",
            "Apply rust preventive oil immediately after processing",
            "Control warehouse humidity below 60% RH",
            "Add inhibitors to pickling bath to protect base metal"
        ],
        "affected_properties": [
            "Reduced fatigue strength",
            "Poor surface finish for painting or plating",
            "Corrosion initiation sites in service"
        ],
        "industry_standards": "Unacceptable per ASTM A568; customer reject criteria apply"
    },

    "Inclusion": {
        "description": (
            "Non-metallic inclusions (oxides, sulfides, silicates) "
            "exposed on the surface or near-surface"
        ),
        "severity": "High",
        "causes": [
            "Poor steelmaking practice - inadequate deoxidation",
            "Slag entrainment during casting",
            "Reoxidation in tundish or mold due to air exposure",
            "Refractory erosion releasing particles into the melt",
            "Insufficient calcium treatment for inclusion modification"
        ],
        "remedies": [
            "Optimize secondary steelmaking (LF/RH degassing)",
            "Calcium-silicon injection for inclusion shape control",
            "Argon shrouding of tundish streams to prevent reoxidation",
            "Use submerged entry nozzle (SEN) with proper immersion depth",
            "Electromagnetic stirring in mold for inclusion flotation",
            "Regular tundish lining maintenance"
        ],
        "affected_properties": [
            "Severe reduction in fatigue life",
            "Crack initiation sites under load",
            "Anisotropic mechanical properties (elongated sulfide inclusions)",
            "Lamellar tearing risk in thick plate welding"
        ],
        "industry_standards": "Controlled per ASTM E45 (inclusion rating) and customer specs"
    },

    "Scratches": {
        "description": (
            "Linear mechanical scratches or score marks on the surface "
            "from contact with equipment during processing or handling"
        ),
        "severity": "Low to Moderate",
        "causes": [
            "Damaged or worn guide rolls and deflector rolls",
            "Foreign particles on roll surfaces",
            "Improper coil handling during transportation",
            "Strip contact with sharp edges of equipment",
            "Tension bridle roll surface degradation"
        ],
        "remedies": [
            "Regular inspection and replacement of guide rolls",
            "Install strip cleaning brushes before critical roll contacts",
            "Use soft interleaving paper during coil storage",
            "Maintain smooth transitions at all strip contact points",
            "Train operators in proper coil handling procedures"
        ],
        "affected_properties": [
            "Cosmetic defect - rejected for exposed automotive panels",
            "Minor stress concentration effect",
            "Poor coating adhesion along scratch lines"
        ],
        "industry_standards": "Severity-dependent; light scratches may be acceptable per ASTM A568"
    },
}


# ── Microstructural Defect Flags ───────────────────────────────────────────
# Triggered when the microstructure CNN identifies specific problematic phases.

MICROSTRUCTURAL_DEFECTS = {

    "Network_Carbides": {
        "is_defect": True,
        "problem": (
            "Continuous cementite along grain boundaries creates "
            "easy crack propagation paths - extreme brittleness"
        ),
        "causes": [
            "Slow cooling of hypereutectoid steel from above Acm",
            "Insufficient normalizing before final heat treatment",
            "Improper forging temperature control"
        ],
        "remedies": [
            "Normalize at 50 deg C above Acm then air cool to break network",
            "Follow with spheroidizing anneal for best results",
            "Multiple normalizing cycles for severe cases",
            "Ensure forging finish temperature is properly controlled"
        ]
    },

    "Widmanstatten": {
        "is_defect": True,
        "problem": (
            "Large ferrite plates reduce toughness severely; "
            "act as preferred crack paths under impact loading"
        ),
        "causes": [
            "Cooling from excessively high austenitizing temperature",
            "Coarse prior austenite grains (overheating during forging/welding)",
            "Found commonly in weld heat-affected zones (HAZ)"
        ],
        "remedies": [
            "Normalize at correct temperature (Ac3 + 30-50 deg C only)",
            "Avoid excessive austenitizing temperatures",
            "Grain refinement through controlled rolling (TMCP)",
            "Post-weld normalizing heat treatment for HAZ"
        ]
    },

    "Decarburization": {
        "is_defect": True,
        "problem": (
            "Loss of carbon from surface creates a soft ferrite "
            "layer - dramatically reduces surface hardness and fatigue life"
        ),
        "causes": [
            "Heating in oxidizing atmosphere without protection",
            "Prolonged holding at high temperature in open furnace",
            "Improper atmosphere control during heat treatment"
        ],
        "remedies": [
            "Use protective atmosphere (nitrogen, argon, endothermic gas)",
            "Vacuum heat treatment for critical components",
            "Apply anti-decarburization coatings before heating",
            "Minimize time at high temperature",
            "Machine off decarburized layer if already occurred"
        ]
    },

    "Grain_Coarsening": {
        "is_defect": True,
        "problem": (
            "Excessively large austenite grains reduce strength, "
            "toughness, and promote Widmanstatten formation on cooling"
        ),
        "causes": [
            "Austenitizing temperature too high",
            "Holding time at temperature too long",
            "Absence of grain-refining elements (Al, Nb, V, Ti)"
        ],
        "remedies": [
            "Strict temperature control during austenitizing",
            "Use grain-refining micro-alloying (0.02-0.05% Nb or V)",
            "Normalizing treatment to refine grains",
            "Double normalizing for severely coarsened grains"
        ]
    },

    "Pearlite_Widmanstatten": {
        "is_defect": True,
        "problem": (
            "Widmanstatten ferrite plates mixed with pearlite severely "
            "reduce impact toughness and act as preferred crack paths"
        ),
        "causes": [
            "Cooling from excessively high austenitizing temperature",
            "Coarse prior austenite grains from overheating",
            "Found in weld heat-affected zones (HAZ)"
        ],
        "remedies": [
            "Normalize at correct temperature (Ac3 + 30-50 deg C only)",
            "Avoid excessive austenitizing temperatures",
            "Grain refinement through controlled rolling (TMCP)",
            "Post-weld normalizing heat treatment for HAZ"
        ]
    },
}


# ── Helper Functions ───────────────────────────────────────────────────────

def get_surface_defect(class_name: str) -> dict:
    """Look up surface defect info by class name or NEU folder name."""
    # Direct match
    if class_name in SURFACE_DEFECTS:
        return SURFACE_DEFECTS[class_name]
    # Try NEU folder mapping
    mapped = NEU_TO_KB_MAP.get(class_name)
    if mapped and mapped in SURFACE_DEFECTS:
        return SURFACE_DEFECTS[mapped]
    return {"error": f"Unknown surface defect: {class_name}"}


def get_micro_defect(class_name: str) -> dict:
    """
    Check if a microstructure class is flagged as a defect.
    Returns defect info dict if it is, or None if it is not a defect.
    """
    if class_name in MICROSTRUCTURAL_DEFECTS:
        return MICROSTRUCTURAL_DEFECTS[class_name]
    return None


def format_surface_report(class_name: str, confidence: float) -> str:
    """Generate a formatted surface defect report."""
    info = get_surface_defect(class_name)
    if "error" in info:
        return f"Error: {info['error']}"

    display_name = class_name.replace("_", " ").upper()
    lines = []
    lines.append("=" * 62)
    lines.append("       SURFACE DEFECT ANALYSIS REPORT")
    lines.append("=" * 62)
    lines.append("")
    lines.append(f"  Detected Defect:   {display_name}")
    lines.append(f"  Confidence:        {confidence * 100:.1f}%")
    lines.append(f"  Severity:          {info['severity']}")
    lines.append(f"  Description:       {info['description']}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  ROOT CAUSES:")
    for cause in info["causes"]:
        lines.append(f"    > {cause}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  RECOMMENDED REMEDIES:")
    for remedy in info["remedies"]:
        lines.append(f"    > {remedy}")
    lines.append("")
    lines.append("-" * 62)
    lines.append("  AFFECTED PROPERTIES:")
    for prop in info["affected_properties"]:
        lines.append(f"    * {prop}")
    lines.append("")
    if "industry_standards" in info:
        lines.append("-" * 62)
        lines.append(f"  STANDARD: {info['industry_standards']}")
        lines.append("")
    lines.append("=" * 62)
    return "\n".join(lines)


def format_defect_flag(class_name: str) -> str:
    """Generate a defect flag alert for a microstructural defect."""
    info = get_micro_defect(class_name)
    if info is None:
        return ""

    lines = []
    lines.append("")
    lines.append("!" * 62)
    lines.append("  [!] DEFECT FLAG - THIS MICROSTRUCTURE IS PROBLEMATIC")
    lines.append("!" * 62)
    lines.append(f"  Problem: {info['problem']}")
    lines.append("")
    lines.append("  Probable Causes:")
    for cause in info["causes"]:
        lines.append(f"    > {cause}")
    lines.append("")
    lines.append("  Corrective Actions:")
    for remedy in info["remedies"]:
        lines.append(f"    > {remedy}")
    lines.append("!" * 62)
    return "\n".join(lines)


# ── Quick Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Surface defect classes: {len(SURFACE_DEFECTS)}")
    print(f"Microstructural defect flags: {len(MICROSTRUCTURAL_DEFECTS)}")
    print()

    # Demo surface report
    print(format_surface_report("Crazing", 0.923))
    print()

    # Demo defect flag
    print(format_defect_flag("Network_Carbides"))
