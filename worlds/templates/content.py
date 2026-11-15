"""Static content banks for the policy-memo world.

Everything is fictional. Planted markers are generated separately by the
generator (decimal numbers and invented head tokens) so they cannot collide
with anything here. Nothing in this file may contain a decimal number.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Name generation (invented words, avoids parametric-knowledge leakage)
# --------------------------------------------------------------------------
ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "r", "s", "st", "t", "tr", "v", "z", "th", "c", "w"]
VOWELS = ["a", "e", "i", "o", "u", "ae", "ia", "ou", "y"]
CODAS = ["", "n", "r", "l", "s", "th", "x", "nd", "rn", "sk", "m", "lt"]
ENDINGS = ["ane", "ith", "orra", "elle", "ost", "ard", "wyn", "ick", "eth", "an", "ira", "un", "ar", "ess", "ow"]

AUTHORITIES = ["Velmora Basin Water Authority", "Tarsk Valley Water Authority", "Orrin Delta Water Board", "Kaldor Uplands Water Authority"]

# --------------------------------------------------------------------------
# Questions (the brief names only these, never the tracked issues)
# --------------------------------------------------------------------------
QUESTIONS = [
    ("Q1", "Can the Authority's supply system meet projected 2028 demand?"),
    ("Q2", "Which capital projects should the Authority fund in the 2027 to 2030 plan, and at what cost?"),
    ("Q3", "Which contractors are reliable enough to award upcoming works to?"),
    ("Q4", "What are the main risks to drinking water quality, and how severe are they?"),
    ("Q5", "How should the Authority prioritize districts for leakage and mains repair?"),
    ("Q6", "What should the Authority tell the Provincial Council about drought contingency?"),
]

# Report series used for resolve_by_rule pairs (two editions of the same series).
SERIES = {
    "Q1": "Quarterly Supply Outlook",
    "Q2": "Capital Works Register",
    "Q3": "Contractor Performance Return",
    "Q4": "Water Quality Monitoring Report",
    "Q5": "Network Performance Bulletin",
    "Q6": "Drought Readiness Update",
}

HIGH_CRED_SOURCES = ["Authority Audit Office", "Provincial Statistics Bureau", "Office of the Chief Engineer"]
LOW_CRED_SOURCES = [
    ("anonymous post on the Basin Residents forum", "Forum post"),
    ("leaflet from the Clearwater Now campaign", "Campaign leaflet"),
    ("unsourced social media thread", "Social media thread"),
    ("newsletter of the Pipe Trades Lobby", "Lobby newsletter"),
]
CONSULTANCIES = ["Marrow and Teal Consulting", "Ostrand Analytics", "Pellam Hydrology Group", "Quint Delta Advisory", "Harrow Lane Economics", "Seddon Fisk Partners"]
CONTROL_SOURCES = ["Operations Division", "Regional Planning Office", "Asset Management Unit", "Customer Services Directorate", "Engineering Standards Unit"]

# --------------------------------------------------------------------------
# Issue frames
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    question_id: str
    magnitude: str  # numeric | direct | qualifier
    subject: str  # contains {e}: an invented entity name
    # numeric and qualifier
    unit: str = ""
    lo: int = 0
    hi: int = 0
    # direct: realization of the value; contains {h}: the marker head token
    value_form: str = ""
    # qualifier: verb phrase and condition (condition contains {c})
    verb: str = ""
    condition: str = ""


FRAMES: list[Frame] = [
    # Q1 supply and demand
    Frame("Q1", "numeric", "projected 2028 storage level of the {e} Reservoir", unit="%", lo=30, hi=75),
    Frame("Q1", "numeric", "safe yield of the {e} Reservoir", unit="ML/day", lo=40, hi=180),
    Frame("Q1", "direct", "backup source designated for the {e} district", value_form="the {h} aquifer"),
    Frame("Q1", "direct", "operator contracted to run the {e} desalination plant", value_form="{h} Water Services"),
    Frame("Q1", "qualifier", "daily output of the {e} desalination plant", unit="ML/day", lo=30, hi=140, verb="exceed", condition="the {c} intake upgrade"),
    Frame("Q1", "qualifier", "2028 peak demand in the {e} district", unit="ML/day", lo=60, hi=220, verb="stay below", condition="the {c} metering rollout"),
    # Q2 capital projects
    Frame("Q2", "numeric", "capital cost of the {e} pipeline", unit="million credits", lo=80, hi=420),
    Frame("Q2", "numeric", "construction duration of the {e} treatment upgrade", unit="months", lo=14, hi=52),
    Frame("Q2", "direct", "route selected for the {e} pipeline", value_form="the {h} Pass route"),
    Frame("Q2", "direct", "site chosen for the {e} pumping station", value_form="the {h} Flats site"),
    Frame("Q2", "qualifier", "total cost of the {e} interconnector", unit="million credits", lo=60, hi=300, verb="stay below", condition="the {c} land purchase"),
    Frame("Q2", "qualifier", "capacity added by the {e} treatment upgrade", unit="ML/day", lo=20, hi=110, verb="exceed", condition="the {c} membrane order"),
    # Q3 contractors
    Frame("Q3", "numeric", "on-time completion rate of {e} Civil over the last five years", unit="%", lo=55, hi=96),
    Frame("Q3", "numeric", "average cost overrun on {e} Engineering projects", unit="%", lo=4, hi=38),
    Frame("Q3", "direct", "contractor awarded the {e} treatment upgrade contract", value_form="{h} Civil"),
    Frame("Q3", "direct", "firm appointed to audit the {e} reservoir dam", value_form="{h} Structural"),
    Frame("Q3", "qualifier", "crew size {e} Works can field for the 2027 season", unit="full-time equivalent staff", lo=40, hi=260, verb="exceed", condition="the {c} subcontract"),
    Frame("Q3", "qualifier", "defect rate on {e} Construction mains work", unit="defects per km", lo=2, hi=19, verb="stay below", condition="the {c} inspection regime"),
    # Q4 water quality
    Frame("Q4", "numeric", "average nitrate concentration at the {e} intake", unit="mg/L", lo=6, hi=48),
    Frame("Q4", "numeric", "turbidity exceedance rate at the {e} treatment plant", unit="%", lo=2, hi=27),
    Frame("Q4", "direct", "primary contamination source identified for the {e} catchment", value_form="the {h} tannery"),
    Frame("Q4", "direct", "laboratory accredited to test samples from the {e} plant", value_form="{h} Laboratories"),
    Frame("Q4", "qualifier", "peak nitrate level at the {e} intake next summer", unit="mg/L", lo=10, hi=60, verb="exceed", condition="the {c} fertilizer ban"),
    Frame("Q4", "qualifier", "lead levels in the {e} district supply", unit="micrograms per litre", lo=3, hi=25, verb="stay below", condition="the {c} pipe replacement"),
    # Q5 leakage and repair
    Frame("Q5", "numeric", "annual leakage rate in the {e} district", unit="%", lo=8, hi=36),
    Frame("Q5", "numeric", "mains burst rate in the {e} district last year", unit="bursts per 100 km", lo=6, hi=64),
    Frame("Q5", "direct", "district designated as the smart-metering pilot for the {e} zone", value_form="the {h} district"),
    Frame("Q5", "direct", "supplier chosen for replacement mains in the {e} zone", value_form="{h} Pipe Systems"),
    Frame("Q5", "qualifier", "leakage in the {e} district by 2028", unit="%", lo=6, hi=24, verb="fall below", condition="the {c} pressure-management scheme"),
    Frame("Q5", "qualifier", "repair backlog in the {e} zone by year end", unit="thousand labour hours", lo=9, hi=90, verb="stay below", condition="the {c} crew expansion"),
    # Q6 drought contingency
    Frame("Q6", "numeric", "days of supply remaining for the {e} district under the severe-drought scenario", unit="days", lo=40, hi=260),
    Frame("Q6", "numeric", "per-capita consumption target for the {e} district", unit="litres per person per day", lo=95, hi=190),
    Frame("Q6", "direct", "agency designated to lead drought response in the {e} region", value_form="the {h} Directorate"),
    Frame("Q6", "direct", "basin designated as the emergency transfer source for the {e} district", value_form="the {h} basin"),
    Frame("Q6", "qualifier", "emergency transfers available from the {e} basin", unit="ML/day", lo=15, hi=120, verb="exceed", condition="the {c} transfer agreement"),
    Frame("Q6", "qualifier", "restriction savings in the {e} district at stage two", unit="%", lo=6, hi=30, verb="exceed", condition="the {c} enforcement funding"),
]

# --------------------------------------------------------------------------
# Sentence templates. {src} is the source phrase, {subj} the issue subject.
# Qualifier certain templates must not contain any hedge word.
# --------------------------------------------------------------------------
NUMERIC_TEMPLATES = [
    "The {src} puts the {subj} at {val}.",
    "According to the {src}, the {subj} is {val}.",
    "The {src} reports the {subj} as {val}.",
]
DIRECT_TEMPLATES = [
    "The {src} reports that the {subj} is {val}.",
    "According to the {src}, the {subj} is {val}.",
]
QUALIFIER_HEDGED = [
    "According to the {src}, the {subj} is expected to {verb} {val}, but only if {cond} is completed on schedule.",
    "The {src} states that the {subj} is likely to {verb} {val}, provided that {cond} goes ahead.",
]
QUALIFIER_CERTAIN = [
    "According to the {src}, the {subj} will {verb} {val}; this figure is confirmed and carries no conditions.",
    "The {src} states that the {subj} will {verb} {val}. This is a firm figure.",
]

CONTROL_SECOND_NUMERIC = ["The {src} also gives the {subj} as {val}.", "The {src} lists the {subj} at {val}."]
CONTROL_SECOND_DIRECT = ["The {src} also records that the {subj} is {val}.", "The {src} confirms that the {subj} is {val}."]

FILLER = [
    "The figures were prepared for the Authority's planning cycle.",
    "Staff should direct questions about this item to the planning team.",
    "The item was circulated to the executive group for information.",
    "No further action is requested from the analysis team at this stage.",
    "Background tables are available from the issuing office on request.",
    "This summary was compiled from routine monitoring returns.",
    "The issuing office notes that methodology is unchanged from prior practice.",
    "A briefing session on this item is planned for later in the quarter.",
]

DISTRACTORS = [
    ("Facilities notice", "Building services", "The third-floor meeting rooms will be closed for maintenance on Thursday. Staff should book rooms on the second floor instead."),
    ("Staff announcement", "Human Resources", "The annual staff survey opens next week. Responses are anonymous and take about 15 minutes."),
    ("IT update", "Information Services", "The document management system will be offline overnight for an upgrade. Save work before 6 pm."),
    ("Training reminder", "Learning and Development", "Mandatory safety refresher training must be completed by the end of the month."),
    ("Board calendar", "Corporate Secretariat", "The next board meeting has moved to the second Tuesday of the month. Papers are due a week earlier."),
    ("Travel policy", "Finance Office", "Mileage claims must now be submitted through the new expenses portal within 30 days."),
    ("Media log", "Communications", "The communications team handled 12 press enquiries this week, mostly about summer bathing water."),
    ("Visitor notice", "Corporate Secretariat", "A delegation from a neighbouring authority will tour the head office on Friday morning."),
    ("Procurement update", "Procurement Office", "Stationery orders are now consolidated monthly. Submit requests by the 20th."),
    ("Records reminder", "Records Management", "Paper files older than seven years should be sent to the archive for review."),
    ("Wellbeing", "Human Resources", "Lunchtime walking groups meet at the main entrance on Mondays and Wednesdays."),
    ("Parking", "Building services", "Resurfacing of the staff car park will take place over two weekends next month."),
    ("Newsletter", "Communications", "Contributions for the internal newsletter are welcome, especially photos from field teams."),
    ("Security", "Information Services", "Phishing attempts have increased. Report suspicious emails using the report button."),
    ("Canteen", "Building services", "The canteen will trial a new menu with more vegetarian options from next week."),
    ("Diary note", "Executive Office", "The chief executive is on leave for a week from Monday. Urgent matters go to the deputy."),
]

BACKGROUND_PARAGRAPHS = [
    "The {auth} serves a mix of urban and rural districts, including {d1} and {d2}. Supply comes from upland reservoirs such as the {r1} Reservoir, supplemented by groundwater in dry years.",
    "Treatment capacity is concentrated at a small number of large works. The {p1} treatment plant handles most urban demand, while smaller plants serve outlying districts such as {d3}.",
    "Network condition varies widely. Older districts such as {d1} have cast-iron mains laid many decades ago, while newer developments in {d2} use modern plastic pipe.",
    "The Authority's capital programme runs on a four-year cycle. Projects are proposed by the Engineering Directorate, scored against a value framework, and approved by the board.",
    "Drought planning follows a staged approach. Stage one relies on voluntary restraint, stage two introduces hosepipe restrictions, and stage three allows non-essential use bans.",
    "Water quality monitoring covers raw water at intakes and treated water at customer taps. Results are reported to the provincial regulator each quarter.",
    "Contractors are procured through framework agreements. Performance is reviewed annually, and firms with repeated delays can be suspended from new awards.",
    "The {r2} Reservoir catchment includes farmland and light industry, which creates periodic pressure on raw water quality during wet periods.",
    "Leakage is estimated from night-flow measurements in metered zones. Zones without district meters rely on modelled estimates, which carry wider uncertainty.",
    "The Provincial Council receives an annual statement on supply resilience. Councillors have recently asked for clearer information on drought risk.",
    "Demand has grown with new housing in {d3} and {d2}. Per-capita use has fallen slightly over the same period as metering coverage has expanded.",
    "The {p2} pumping station lifts water from the valley floor to the eastern service reservoirs. It is one of the Authority's largest electricity users.",
    "Customer complaints focus on discoloured water after bursts, low pressure in hilly areas, and the time taken to repair visible leaks.",
    "The board has asked analysts to distinguish settled facts from contested estimates in all briefing material, and to cite sources for every figure.",
]
BACKGROUND_TITLES = [
    "Authority overview", "Supply system description", "Network condition summary", "Capital planning process",
    "Drought plan structure", "Water quality monitoring framework", "Contractor framework guide", "Catchment profile",
    "Leakage measurement methods", "Council reporting requirements", "Demand trends", "Pumping assets",
    "Customer service themes", "Briefing standards", "Glossary of planning terms", "Regulatory calendar",
    "Organisation chart notes", "History of the Authority",
]

HEDGE_WORDS = [
    "expected", "likely", "only if", "provided", "provided that", "conditional", "condition on", "conditioned",
    "may", "might", "could", "projected to", "provisional", "uncertain", "depends", "depending", "subject to",
    "assuming", "tentative", "if ", "unless", "hedged", "contingent",
]
