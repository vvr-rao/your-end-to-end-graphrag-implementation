"""Qwen3.8-27B-specific prompts.

Kept SEPARATE from `prompts.py` on purpose (user decision 2026-10-13): Qwen
gets its own wording, and edits here never change what OpenAI or any other
model is sent -- nor do edits to the shared prompts change Qwen's.

Selected per task with `prompt_set: qwen` in models.yaml (the deepinfra
preset sets it on class_proposal and relationship_extract). Started as a copy of the shared
`class_identification_and_expansion`, then given a compact output contract.
"""
from __future__ import annotations

import json
from typing import Any

from backend.app.services.prompts import GENERIC_PREDICATE_IRIS

# Shrinks the reply WITHOUT
# capping how many concepts are proposed. v2 (2026-10-14): v1's "be
# EXHAUSTIVE" line tripled new classes/relations per reply (19 -> 58 classes;
# 196 -> 667 in the final ontology) and two replies ran away enumerating
# DATA_CLASSES; v2 drops it and forbids listing unmentioned classes.
# Measured on 14 Qwen3.8-27B replies
# (2026-10-13): pretty-print whitespace was ~27% of each reply; 441 of 500
# MATCHES FOUND entries re-listed an IRI already in the same reply (only the
# IRI is used downstream); instances were 53% of the output; repeated labels
# drove the runaway replies that hit the 32k cap and were discarded.
QWEN_OUTPUT_RULES = (
    "\n\nOUTPUT RULES (these shorten the reply; they do NOT reduce "
    "coverage):\n"
    "  - These rules change only HOW you write each entry, not WHICH concepts "
    "you include: apply the coverage rules above unchanged.\n"
    "  - MATCHES FOUND: match an existing class ONLY when a specific phrase in "
    "the TEXT refers to it. NEVER walk through DATA_CLASSES listing classes "
    "the TEXT does not mention.\n"
    "  - Write the JSON on ONE line: no indentation, no line breaks, no spaces "
    "after ':' or ','.\n"
    "  - MATCHES FOUND: list each IRI AT MOST ONCE, even if the text mentions "
    "it many times. TEXT_SNIPPET = the shortest phrase showing the match "
    "(at most 8 words).\n"
    "  - Every LABEL appears AT MOST ONCE per section. Before adding an entry, "
    "check it is not already in your list; if it is, skip it. Never list "
    "spelling or plural variants of the same concept separately.\n"
    "  - MATCH NOT FOUND DESCRIPTION: exactly one sentence.\n"
    "  - MATCH NOT FOUND RELATIONS DESCRIPTION: at most 15 words.\n"
    "  - MATCH NOT FOUND INSTANCES: omit CANONICAL_FORM when it would equal "
    "LABEL; DESCRIPTION at most 12 words.\n"
    "  - Write the sections in the order shown: new classes FIRST, then "
    "relations, then instances, and MATCHES FOUND LAST.\n"
    "  - When every distinct concept has been listed once, close the JSON and "
    "stop.\n\n"
    "CLASS OR INSTANCE? Apply this test literally to every concept: does the "
    "name pick out ONE specific thing, or a KIND of thing that many things "
    "can be?\n"
    "  - KINDS are CLASSES (MATCH NOT FOUND) -- even when the text mentions "
    "only one example. Kinds include: conditions, states and event types "
    "(Recession, SupplyShortage, DataBreach, Hypoglycemia); properties, "
    "measures and metrics (InterestRate, OperatingMargin, Latency, "
    "BloodPressure); materials and substances (Lithium, Cholesterol, "
    "Helium); categories of product, technology or service (ElectricVehicle, "
    "CloudStorage, Antidepressant); roles and groups of people or "
    "organisations (Regulator, Supplier, Children, RetailInvestor); and kinds "
    "of document, rule, risk or program (AnnualReport, ExportControl, "
    "CreditRisk, LoyaltyProgram).\n"
    "  - Specific NAMED things are INSTANCES (MATCH NOT FOUND INSTANCES): "
    "named products and brands (iPhone 15, Model Y, Ozempic), named "
    "organisations, people, places, documents, laws, events and dated "
    "periods (Apple Inc, Jerome Powell, Strait of Hormuz, 2024 Form 10-K, "
    "Inflation Reduction Act, Q3 2025). Give each a TYPE_LABEL naming its "
    "kind -- propose that kind as a class if it is missing.\n"
    "  - When unsure, ask: could the text sensibly say 'a <name>' or "
    "'another <name>'? If yes ('a recession', 'another supplier'), it is a "
    "kind.\n"
    "  - Do not file a kind as an instance to save space: a kind filed as an "
    "instance is lost from the ontology."
)

# Relationship extraction (Qwen-only). Qwen's chunks are its own MARKDOWN
# summaries (18/18 chunks vs 0/8 for OpenAI's plain-text ones, 2026-10-14):
# assertions sit in bullet labels ("**Ozempic:** Approved for ...") and table
# rows, which a model reads as formatting, and a quote that drops the bold
# label no longer names both entities.
QWEN_RELATIONSHIP_RULES = (
    "\n\nTHE PASSAGE IS A MARKDOWN SUMMARY -- read its structure as text:\n"
    "  - A bullet is a sentence. '**Tesla:** Supplies batteries to Panasonic "
    "plants' or '**Ozempic:** Approved for type 2 diabetes' each state a "
    "relationship between the bold label and the rest of the bullet. Quote "
    "the WHOLE bullet INCLUDING its bold label, because the label is where "
    "the subject is named.\n"
    "  - A table row is a sentence. In a row '| Acme Corp | Berlin | Battery "
    "cells |' under headers '| Company | Headquarters | Main product |', Acme "
    "Corp is headquartered in Berlin and makes battery cells. Quote the "
    "row.\n"
    "  - A heading names the subject of the bullets beneath it; extend the "
    "quote back to the heading when a bullet does not name the subject.\n"
    "  - Markdown symbols (**, *, |, #) may be copied or left out -- the "
    "checker ignores them -- but every WORD must be copied exactly, in "
    "order, with no paraphrase.\n"
    "  - Expect MANY relationships in these summaries: they are dense lists "
    "of who owns, makes, supplies, regulates, causes, treats or competes with "
    "what. Work through every bullet and row.\n"
    "  - Write the JSON on ONE line with no indentation."
)



def qwen_class_proposal(
    ontology_slice: dict[str, Any],
    text_chunk: str,
    suggested_new_classes: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Qwen3.8-27B variant of Stage 2 (see module docstring). Given a sliced sub-ontology (only the classes relevant to a
    chunk, per Stage-1 narrowing) plus the chunk's text, match concepts to
    existing classes and propose new ones for unmatched concepts."""
    suggested_block = ""
    if suggested_new_classes:
        suggested_block = (
            "\n\nSUGGESTED NEW CLASSES (consider but don't blindly accept):\n"
            + json.dumps(suggested_new_classes, ensure_ascii=False)
        )

    system = (
        "You are an expert ontology curator. You will be given a slice of an "
        "ontology (DATA_CLASSES, a JSON dict keyed by class IRI) and a passage "
        "of TEXT.\n\n"
        "HARD RULE (read before anything else):\n"
        "  CLASSES are abstract categories (CarManufacturer, Country, "
        "ChemicalElement, Regulation, Report). They describe what something "
        "IS. They are NEVER specific named individuals.\n"
        "  Specific named companies (BYD Company Ltd., OCI N.V., Apple Inc), "
        "specific countries / regions / cities (Myanmar, Vietnam, Asia, "
        "Tokyo), specific named people, specific products, specific named "
        "events, and specific named reports / dashboards / factbooks "
        "(\"Sovereign Risk Tracker\", \"2025 Factbook\", \"Vietnam's "
        "Manufacturing & Supply Chain Industry Report\") are INSTANCES of "
        "categories, NOT categories themselves. Emit them as MATCH NOT FOUND "
        "INSTANCES entries -- NEVER as MATCH NOT FOUND class entries.\n"
        "  If you are tempted to add a class whose LABEL is a proper noun "
        "(corporate suffix like 'Inc' / 'Ltd' / 'N.V.' / 'Corp' / 'GmbH'; a "
        "country or place name on its own; a specific report or dashboard "
        "title; a named facility or series designator like 'Cholla Unit 4' "
        "or 'ASU 2019-01'), STOP -- demote that proposal to MATCH NOT FOUND "
        "INSTANCES (with TYPE_LABEL = the parent class it instantiates). A "
        "downstream deterministic pass rejects the clear-cut proper-noun "
        "classes and an audit pass reviews the borderline ones, so emitting "
        "them is wasted work at best and a corrupted ontology at worst.\n\n"
        "Your job:\n"
        "  1. Identify every passage substring that maps to an existing class "
        "in DATA_CLASSES. The IRI you return MUST be an exact key of the "
        "DATA_CLASSES dict.\n"
        "  2. For every distinct concept in the TEXT that has NO matching "
        "existing class, propose a new class with a clear LABEL, DESCRIPTION, "
        "and PARENT_LABEL (the most specific class in DATA_CLASSES that this "
        "new class should be a kind of -- e.g. 'Country' for 'Washington', "
        "'Manufacturer' for 'BMW', 'Chemical Element' for 'helium'). If no "
        "suitable parent exists in DATA_CLASSES, you MUST propose the parent "
        "as ANOTHER MATCH NOT FOUND entry in the same response (chain rule) "
        "and use its LABEL as the PARENT_LABEL. Only use PARENT_LABEL='NONE' "
        "if the concept genuinely has no natural parent.\n"
        "     The LABEL of a MATCH NOT FOUND CLASS must be a category name "
        "(CarManufacturer, FertilizerProducer, RiskMonitoringDashboard, "
        "TradeAgreement) -- never a specific named individual. ACCEPT: "
        "'FertilizerProducer', 'RiskMonitoringDashboard', 'TradeAgreement', "
        "'SoutheastAsianCountry'. REJECT (emit as INSTANCES instead): 'BYD "
        "Company Ltd.', 'OCI N.V.', 'Myanmar', 'Sovereign Risk Tracker', "
        "'2025 Factbook'.\n"
        "     CRITICAL: Do NOT use a geography class (GeographicEntity, "
        "Place, Country, Region, Continent, City, or any other class in "
        "the geography ontology) as PARENT_LABEL for a concept that is "
        "NOT itself a geographic place or landform. For example, things "
        "like 'EV bus', 'Subsidy', 'Trade flow', 'Ministry of Finance', "
        "'Supply shock', 'Subsidy', 'Hydrogen' are NOT geographic. If "
        "you can't find a non-geographic parent that fits, use "
        "PARENT_LABEL='NONE' -- a later step will assign a proper "
        "top-level concept.\n"
        "  3. Identify RELATIONS between classes the TEXT asserts or implies. "
        "Both endpoints may be existing classes (use their LABEL or IRI) or "
        "classes you just proposed in step 2 (use the same LABEL).\n"
        "  4. INSTANCES (named individuals). For specific time points / "
        "time periods (e.g. 'Jan 2004', 'January 2004', 'Jan04', '2030', "
        "'Q3 2025', 'next decade') and other PROPER-NOUN named individuals "
        "the text references (specific events, specific documents, named "
        "incidents that are one-of-a-kind), emit a MATCH NOT FOUND INSTANCES "
        "entry instead of a MATCH NOT FOUND class entry. Each instance "
        "entry must carry:\n"
        "       - LABEL: the surface form as found in the text\n"
        "       - CANONICAL_FORM: a single canonical form so equivalent "
        "surface forms ('Jan 2004' / 'Jan04' / 'January 2004') collapse to "
        "ONE instance. Pick the most readable canonical form (e.g. "
        "'January 2004').\n"
        "       - TYPE_LABEL: the most specific class in DATA_CLASSES (or a "
        "MATCH NOT FOUND label) that this individual is an instance of. "
        "For time points prefer time classes if available (Year, Month, "
        "DurationDescription, TemporalEntity, etc.). For named events, "
        "prefer 'Event' / 'Crisis' / 'Conflict' / etc.\n"
        "       - DESCRIPTION: a brief sentence of context.\n"
        "     Rule of thumb: classes are KINDS OF things ('Year', 'Crisis', "
        "'Country'); instances are SPECIFIC things ('January 2004', 'Iran "
        "war 2025', 'Kingdom of Saudi Arabia'). Do NOT create a new class "
        "for each year mentioned in the text -- create instances of Year.\n"
        "\n"
        "Review thoroughly -- do not stop after a few matches. When in doubt, "
        "INCLUDE the match. Specifically:\n"
        "  - Any proper-noun PLACE -- including political-administrative "
        "places (country, region, province, state, county, city, town, "
        "district, neighborhood) AND natural / physical-geography features "
        "(continent, ocean, sea, gulf, bay, strait, channel, isthmus, "
        "peninsula, island, archipelago, mountain, mountain range, valley, "
        "river, lake, desert, plateau, coast, basin) -- MUST be matched to "
        "an existing geographic class in DATA_CLASSES when one is present. "
        "Use the most specific matching class as PARENT_LABEL; fall back "
        "to the most general one (e.g. GeographicEntity / Place) when no "
        "specific kind-of class exists. Examples of the resolution pattern "
        "(your DATA_CLASSES will differ): a country name -> a Country "
        "class; a continent name -> a Continent class; a body of water -> "
        "a Sea / Ocean / Lake / River class; a landform -> an Island / "
        "Mountain / Peninsula / Strait class. NEVER root a place under "
        "owl:Thing when a geographic class is available, even if the chunk "
        "discusses the place in a non-geographic context (e.g. a strait "
        "named in a shipping story is still a Strait).\n"
        "  - Temporal mentions (years like '2030', durations like 'next "
        "decade', recurring intervals like 'annual', day-of-week references) "
        "MUST be matched to existing temporal classes if any are present.\n"
        "  - Regulatory, economic, and policy concepts should be matched even "
        "if the text discusses them indirectly.\n"
        "Only emit a MATCH NOT FOUND proposal when NO existing class in "
        "DATA_CLASSES plausibly fits the concept.\n\n"
        "ENTITY-TYPE RULES (critical -- get these right):\n"
        "  1. NAMED INDIVIDUAL PEOPLE (politicians, executives, public "
        "figures, researchers, named officials -- e.g. Donald Trump, "
        "Vladimir Putin, Xi Jinping, Tim Cook): emit as MATCH NOT FOUND "
        "INSTANCES.\n"
        "       - TYPE_LABEL = the foaf:Person class IRI if it appears "
        "in DATA_CLASSES (look for 'http://xmlns.com/foaf/0.1/Person' "
        "or any class labeled 'Person' / 'Agent'); else use 'Person' as "
        "the label and let the deterministic pass anchor it.\n"
        "       - LABEL = the person's full name as written in the "
        "text. CANONICAL_FORM = the same name in CamelCase "
        "(DonaldTrump, VladimirPutin, XiJinping).\n"
        "       - If the text mentions the person's ROLE (President, "
        "Prime Minister, CEO, Minister of X, Secretary of Y), emit a "
        "SEPARATE MATCH NOT FOUND RELATION with LABEL='hasRole', "
        "DOMAIN=<person canonical form>, RANGE=<role label>.\n"
        "       - NEVER create a CLASS for a specific individual person. "
        "NEVER bake the role into the instance label "
        "('President Donald Trump' is WRONG; emit 'Donald Trump' as the "
        "instance + a hasRole relation to 'President').\n"
        "  2. COMPANIES + GOVERNMENT BODIES + NGOS + AGENCIES + "
        "ASSOCIATIONS. Any entity whose name has a corporate suffix or "
        "agency pattern (Inc, Ltd, Corp, Corporation, Industries, "
        "Petrochemicals, Petroleum, Company, Co., Co. Ltd., Group, plc, "
        "GmbH, S.A., Holdings, Bank, University, Ministry of X, "
        "Department of X, Bureau of X, NGO, Council, Federation, "
        "Association, OPEC-style acronym) is an ORGANIZATION.\n"
        "       - PARENT_LABEL of any organization-shaped class MUST be "
        "an IRI of foaf:Organization or org:Organization (or one of "
        "their existing subclasses already in DATA_CLASSES, e.g. "
        "org:FormalOrganization, org:OrganizationalUnit). foaf:Organization "
        "AND org:Organization are GUARANTEED to be present in DATA_CLASSES "
        "for every chunk -- if you do not see them, you missed them. "
        "NEVER invent labels like 'BusinessEntity', 'Corporation', "
        "'Company', 'CommercialEntity', or any other made-up parent. "
        "NEVER place an organization under Material, Helium, "
        "ChemicalProcessSystem, ProcessUnit, Infrastructure, or any "
        "non-org class.\n"
        "       - For a domain-specific subclass like 'MiningCompany', "
        "'CarManufacturer', 'BankHolding', 'GovernmentAgency': emit the "
        "subclass as MATCH NOT FOUND with PARENT_LABEL set to the "
        "foaf:Organization (or org:Organization) IRI from DATA_CLASSES. "
        "Then attach the individual company as a MATCH NOT FOUND "
        "INSTANCES entry whose TYPE_LABEL is the subclass label.\n"
        "       - Examples: 'Samsung Electronics' -> instance of "
        "foaf:Organization. 'Air Products and Chemicals' -> instance of "
        "foaf:Organization. 'Saudi Aramco Total Refinery & Petrochemicals "
        "Co.' -> instance of foaf:Organization. 'Sinopec Maoming Company' "
        "-> instance of foaf:Organization. 'Gold Hunter Resources' -> "
        "instance of MiningCompany (which itself has PARENT_LABEL = "
        "foaf:Organization IRI).\n"
        "  3. PHYSICAL FACILITIES (refinery complexes, plants, ports, "
        "terminals, generating units, mines, warehouses) -- e.g. 'Jamnagar "
        "Refinery Complex', 'GS-Caltex Yeosu Refinery', 'Hormuz Oil "
        "Terminal', 'Cholla Unit 4'. A NAMED facility is an INSTANCE; the "
        "KIND of facility is the class.\n"
        "       - Emit the KIND as MATCH NOT FOUND with PARENT_LABEL = "
        "'Infrastructure' -- 'RefineryComplex', 'OilTerminal', "
        "'CoalFiredGeneratingUnit'.\n"
        "       - Emit the NAMED facility as a MATCH NOT FOUND INSTANCES "
        "entry with TYPE_LABEL = that kind. 'Jamnagar Refinery Complex' -> "
        "instance of RefineryComplex. 'Cholla Unit 4' -> instance of "
        "CoalFiredGeneratingUnit.\n"
        "       - NEVER emit the named facility itself as a class. Doing so "
        "produced a real failure: 'ChollaUnit4' -- ONE plant in Arizona -- "
        "became a class and was then used to type 18 unrelated plants in "
        "four other states.\n"
        "       - Emit a MATCH NOT FOUND RELATION linking the facility "
        "to its owner: LABEL='operatedBy' (or 'owner'), DOMAIN=<facility>, "
        "RANGE=<organization>.\n"
        "  3.5 SERIES DESIGNATORS. A name that identifies ONE item in a "
        "numbered or lettered series -- a standard, regulation, rule, "
        "docket, protocol, study, trial, model, release, patent, or unit "
        "number. Examples across domains: 'ASU 2019-01', 'IFRS 16', "
        "'SEC Rule 10b-5', 'NCT02345678', 'Study 301', 'Boeing 737', "
        "'Annex IV', 'Phase III of the ACME trial'.\n"
        "       - These are INSTANCES. Emit each as a MATCH NOT FOUND "
        "INSTANCES entry whose TYPE_LABEL is the SERIES it belongs to "
        "('AccountingStandardUpdate', 'Regulation', 'ClinicalTrial', "
        "'AircraftModel', 'GeneratingUnit'). Emit that series as a class "
        "if it does not already exist.\n"
        "       - Tell these apart from genuine CATEGORY names that merely "
        "contain a digit. 'Type2Diabetes', 'Phase3ClinicalTrial', "
        "'Scope3Emission', 'Tier1Supplier', 'CYP3A4', 'HER2', 'PM2.5', "
        "'CO2' are KINDS -- keep them as classes. The test is whether the "
        "name picks out ONE thing ('ASU 2019-01' is one document) or a "
        "category of things ('Type 2 diabetes' is a condition many people "
        "have).\n"
        "  4. EVENTS named after a place or entity. Patterns: '<X> "
        "crisis', '<X> closure', '<X> war', '<X> disruption', '<X> "
        "shortage', '<X> conflict', '<X> incident', '<X> shutdown', "
        "'<X> blockage', '<X> attack', '<X> embargo', '<X> escalation', "
        "'<X> summit'.\n"
        "       - PARENT_LABEL = 'Event' (or 'Crisis' / 'Conflict' "
        "subclass if present in DATA_CLASSES). EVEN IF the entity "
        "contains a geographic name: 'Strait of Hormuz crisis' -> "
        "Event, NOT Strait. 'Russia-Ukraine war' -> Event, NOT Russia.\n"
        "  5. ROLE TYPES (kinds of positions): 'President', "
        "'PrimeMinister', 'CEO', 'Chairman', 'Secretary of <X>', "
        "'Minister of <Y>', 'Director', 'Founder', 'Chair', "
        "'FederalReserveChair', 'TreasurySecretary'.\n"
        "       - Emit each role TYPE as a MATCH NOT FOUND INSTANCES "
        "entry with TYPE_LABEL = the org:Role IRI from DATA_CLASSES "
        "(org:Role is GUARANTEED to be in DATA_CLASSES for every chunk). "
        "DO NOT emit role types as new classes -- they are individuals "
        "of org:Role.\n"
        "       - NEVER conflate a role type with a person. 'President' "
        "is an org:Role instance; 'Donald Trump' is the foaf:Person "
        "who holds it. Two separate entities.\n"
        "       - A POST is the COMPOSITION of a role + an organization "
        "(e.g. 'ChairOfFederalReserve' = Chair role at Federal Reserve "
        "org). Emit each post as a MATCH NOT FOUND INSTANCES entry with "
        "TYPE_LABEL = the org:Post IRI from DATA_CLASSES. Then emit two "
        "MATCH NOT FOUND RELATIONS: org:role (Post -> Role) and "
        "org:postIn (Post -> Organization).\n"
        "  6.5 PEOPLE. Specific named human beings (Jerome Powell, "
        "Janet Yellen, Donald Trump, Tim Cook).\n"
        "       - Emit each person as a MATCH NOT FOUND INSTANCES entry "
        "with TYPE_LABEL = the foaf:Person IRI from DATA_CLASSES "
        "(foaf:Person is GUARANTEED to be in DATA_CLASSES for every "
        "chunk).\n"
        "       - When a chunk names a person AND their position, "
        "ALSO emit a MATCH NOT FOUND RELATIONS entry with LABEL='holds' "
        "(maps to org:holds), DOMAIN=<person canonical name>, "
        "RANGE=<post canonical name>. Example for 'Jerome Powell, Chair "
        "of the Federal Reserve': INSTANCES include 'Jerome Powell' "
        "(foaf:Person), 'Federal Reserve' (foaf:Organization), "
        "'FederalReserveChair' (org:Role), 'ChairOfFederalReserve' "
        "(org:Post); RELATIONS include holds(Jerome Powell, "
        "ChairOfFederalReserve), org:role(ChairOfFederalReserve, "
        "FederalReserveChair), org:postIn(ChairOfFederalReserve, "
        "Federal Reserve).\n"
        "  6.6 MEMBERSHIPS. When a chunk says 'X is a member of Y' or "
        "names a person/org as part of a body (board member, committee "
        "member, alliance member), emit a MATCH NOT FOUND INSTANCES "
        "entry for the membership with TYPE_LABEL = the org:Membership "
        "IRI from DATA_CLASSES, and use org:member / org:memberOf / "
        "org:organization as the predicate LABEL when emitting RELATIONS "
        "on the membership.\n"
        "  6. GEOGRAPHIC FEATURES: a class is a geographic place only "
        "when its label IS a place name with no event / process / role "
        "modifier. 'Strait of Hormuz' = yes. 'Strait of Hormuz crisis' "
        "= no (Event). 'Suez Canal' = yes. 'Suez Canal blockage' = no "
        "(Event).\n\n"
        "EXISTING PREDICATES (PREFER THESE OVER INVENTING NEW VERBS):\n"
        "  When you emit a MATCH NOT FOUND RELATIONS entry, set LABEL to "
        "one of these EXACT names whenever the semantics match. These are "
        "real predicates in the loaded ontology; using them prevents "
        "duplicate-predicate explosions in the merged graph.\n"
        "  FOAF: knows, member, topic_interest, topic, interest.\n"
        "  ORG:  holds (Person -> Post), heldBy (inverse), role "
        "(Post -> Role), hasPost (Organization -> Post), postIn (inverse),"
        " hasMember (Organization -> Person), memberOf (Person -> "
        "Organization), hasMembership (Organization -> Membership), "
        "memberDuring (Membership -> TimeInterval), organization "
        "(Membership -> Organization), hasSubOrganization, "
        "subOrganizationOf, originalOrganization, resultingOrganization, "
        "hasPrimarySite, hasSite, siteOf, changedBy, resultedFrom, "
        "resultedIn.\n"
        "  RULE: NEVER coin verbose predicates like 'hasChiefExecutiveOfficer',"
        " 'hasChiefFinancialOfficer', or 'hasVicePresidentOfEngineering'. "
        "Use 'hasMember' (Org -> Person) or the Post pattern instead. The "
        "specific role/position belongs in the org:Role / org:Post "
        "INSTANCES, not in the predicate name.\n\n"
        "SELF-CONSISTENCY (critical, do NOT skip):\n"
        "  - Every PARENT_LABEL you use in MATCH NOT FOUND MUST appear "
        "either (a) as an exact label/IRI in DATA_CLASSES, OR (b) as the "
        "LABEL of another entry in your own MATCH NOT FOUND list, OR (c) "
        "be 'NONE' as a last resort. NEVER invent free-floating parent "
        "names like 'BusinessEntity', 'Concept', 'Item', 'Entity', or "
        "any label that does not exist in DATA_CLASSES. If you are "
        "tempted to invent a parent, pick the closest existing class in "
        "DATA_CLASSES instead.\n"
        "  - Every DOMAIN and RANGE label you use in MATCH NOT FOUND "
        "RELATIONS MUST appear either (a) as an exact label/IRI in "
        "DATA_CLASSES, OR (b) as a LABEL in your own MATCH NOT FOUND list. "
        "If a relation references a concept that is not in DATA_CLASSES, "
        "you MUST propose that concept as a MATCH NOT FOUND class first.\n"
        "  - RELATION DOMAIN/RANGE MUST BE AS GENERAL AS THE RELATION "
        "TRULY IS -- this is the single most common mistake. Pick the MOST "
        "ABSTRACT class for which the relation still holds, NOT the specific "
        "class from the sentence in front of you. A relation is a reusable "
        "part of the vocabulary; typed too narrowly it can never apply to "
        "anything else and is dead on arrival.\n"
        "      WRONG: acquires [DOMAIN=MusicLabel, RANGE=MusicRightsAcquisition] "
        "-- derived from one music-industry sentence, so it can never express "
        "'Google acquired DeepMind'.\n"
        "      RIGHT: acquires [DOMAIN=Organization, RANGE=Organization].\n"
        "      WRONG: backs [DOMAIN=Mozilla, RANGE=Mammoth] -- a type "
        "signature naming two specific companies matches exactly one pair in "
        "the world.\n"
        "      RIGHT: backs [DOMAIN=Organization, RANGE=Organization].\n"
        "      Ask: 'could two OTHER things of this kind stand in this "
        "relation?' If yes, name that kind. Prefer broad classes such as "
        "Organization, Person, Agent, Product, Place, Event over any narrow "
        "subtype, unless the relation is genuinely meaningless for the "
        "broader class.\n"
        "  - NEVER use a NAMED INDIVIDUAL as a DOMAIN or RANGE. 'Mozilla', "
        "'OpenAI' and 'Meta' are instances of Organization, not classes. If "
        "you need one as a relation endpoint, use its TYPE.\n"
        "  - When the chunk introduces multiple concepts whose labels share "
        "a common stem or topic (e.g. 'helium', 'helium market', 'helium "
        "supply', 'helium price'; or 'BMW', 'Honda', 'Hyundai' as siblings "
        "of 'CarManufacturer'), propose has-X / part-of / sibling-of "
        "relations between them using the chunk text as evidence. Stay "
        "grounded -- don't invent relations the text doesn't support.\n"
        "  - When the text mentions a concept under multiple surface forms "
        "(e.g. 'Washington', 'Washington DC', 'Washington D.C.'), emit ONE "
        "MATCH NOT FOUND entry using the most canonical form. If two surface "
        "forms refer to different specific entities (e.g. 'Washington' the "
        "state vs 'Washington' the city), distinguish them with a qualifier "
        "in the label.\n\n"
        "For relations, stay precision-biased on endpoints: skip a relation "
        "only if either endpoint cannot be identified as existing OR proposed "
        "(which means you should have proposed it -- see self-consistency).\n\n"
        "Output strict JSON in the shape:\n"
        "{\n"
        '  "MATCH NOT FOUND": ['
        '    {"LABEL": "<short canonical label>", "DESCRIPTION": "<exactly one sentence>",'
        ' "PARENT_LABEL": "<label or IRI of most specific parent in DATA_CLASSES'
        " or another MATCH NOT FOUND entry, or 'NONE'>\"}, ..."
        "  ],\n"
        '  "MATCH NOT FOUND RELATIONS": ['
        '    {"LABEL": "<verb-phrase relation name, e.g. treats>",'
        ' "DESCRIPTION": "<at most 15 words>",'
        ' "DOMAIN": "<label or IRI of the source class (must exist or be proposed)>",'
        ' "RANGE": "<label or IRI of the target class (must exist or be proposed)>"}, ...'
        "  ],\n"
        '  "MATCH NOT FOUND INSTANCES": ['
        '    {"LABEL": "<surface form from text>",'
        ' "CANONICAL_FORM": "<single canonical form; OMIT this key when it equals LABEL>",'
        ' "TYPE_LABEL": "<label of class this is an instance of, from DATA_CLASSES or MATCH NOT FOUND>",'
        ' "DESCRIPTION": "<at most 12 words>"}, ...'
        "  ],\n"
        '  "MATCHES FOUND": ['
        '    {"IRI": "<exact iri from DATA_CLASSES, each IRI at most once>", "TEXT_SNIPPET": "<shortest phrase showing the match, <=8 words>"}, ...'
        "  ]\n"
        "}\n"
        "No prose. No comments. Only the JSON object."
        + QWEN_OUTPUT_RULES
        + suggested_block
    )
    ontology_repr = json.dumps(ontology_slice, ensure_ascii=False, default=str)
    user = (
        f"DATA_CLASSES:\n{ontology_repr}\n\n"
        f"TEXT TO EVALUATE:\n{text_chunk}\n\n"
        'Return JSON in THIS key order: {"MATCH NOT FOUND": [...], '
        '"MATCH NOT FOUND RELATIONS": [...], "MATCH NOT FOUND INSTANCES": [...], '
        '"MATCHES FOUND": [...]}'
    )
    return system, user


def qwen_relationship_extract(
    chunk_text: str,
    entities: list[dict[str, str]],
    candidate_predicates: list[dict[str, str]],
    *,
    max_relationships: int = 10,
    already_found: list[str] | None = None,
) -> tuple[str, str]:
    """Qwen3.8-27B variant (see QWEN_RELATIONSHIP_RULES). Phase 2 Milestone C, second pass: relationships between entities ALREADY
    extracted from this chunk.

    Why a second call rather than one combined request: the predicate menu can
    only be narrowed to the entities' ACTUAL classes once those entities exist.
    In the combined version predicates were derived from the chunk's top-50
    candidate CLASSES, so the model was shown a median of 12 predicates while
    holding 4-5 entities, most of which no offered predicate fitted -- measured
    669 of 1,140 proposals rejected on domain/range alone.

    `entities` are {canonical_name, class_label} for this chunk only.
    `candidate_predicates` are {iri, label, domain_label, range_label} whose
    domain AND range both match classes those entities actually have.

    Returns JSON:
      {relationships: [{subject, predicate_iri, object, evidence, confidence}]}

    `evidence` is a VERBATIM span from the chunk. The caller checks it really
    occurs in the text and drops the relationship otherwise -- narrowing the
    menu weakens the type check (offer and check now share a basis), so this
    replaces the protection that removes.
    """
    ent_lines = "\n".join(
        f"  - {e['canonical_name']}  [{e.get('class_label') or '?'}]"
        for e in entities
    )

    def _fmt(p: dict[str, str]) -> str:
        return (f"  - {p['iri']} \u2500 {p.get('label') or '?'}"
                f"  [{p.get('domain_label') or '?'} -> "
                f"{p.get('range_label') or '?'}]")

    # Split the menu. Catch-all predicates fit almost any pair of entities, so
    # offered flat they crowd out the specific ones: measured on the news
    # corpus, org:hasMember + org:linkedTo + foaf:topic took 37 of 65 edges
    # (57%) while every corpus-specific predicate the ontology minted for that
    # run -- competesWith, filesLawsuitAgainst, integrates, backs -- shared 12.
    # Listing them separately, under a rule that they are a fallback, makes the
    # model rule out a precise predicate before reaching for a vague one.
    specific = [p for p in candidate_predicates
                if p["iri"] not in GENERIC_PREDICATE_IRIS]
    generic = [p for p in candidate_predicates
               if p["iri"] in GENERIC_PREDICATE_IRIS]
    pred_lines = "\n".join(_fmt(p) for p in specific) or "  (none)"
    if generic:
        pred_lines += ("\n\nLAST-RESORT PREDICATES (vague -- use ONLY if no "
                       "predicate above fits):\n"
                       + "\n".join(_fmt(p) for p in generic))
    # already_found: the SECOND pass. The first pass's accepted relationships
    # are shown and only the ones it missed are asked for -- a single pass
    # finds a different ~70% of a chunk's relationships on each run.
    _gap = (
        "A FIRST PASS has already extracted the relationships listed under "
        "ALREADY FOUND. Your job now is to find what it MISSED: relationships "
        "the passage states that are NOT already listed (the same pair with "
        "subject and object swapped counts as listed). Return ONLY new "
        "relationships -- an empty list is the correct answer when nothing was "
        "missed. Every rule below still applies.\n\n"
    ) if already_found is not None else ""
    system = (
        _gap +
        "You identify RELATIONSHIPS that a passage explicitly asserts between "
        "entities that have ALREADY been extracted from it. Return ONE JSON "
        "object and nothing else -- no prose, no markdown fences.\n\n"
        "For each relationship:\n"
        "  - subject / object: copy a canonical_name VERBATIM from the ENTITIES "
        "list. Never introduce a name that is not on that list.\n"
        "  - predicate_iri: copy ONE IRI VERBATIM from the PREDICATES list. Do "
        "not invent, abbreviate or reformat it.\n"
        "  - evidence: the VERBATIM sentence or clause from the passage that "
        "states this relationship. It must appear in the passage character for "
        "character. Do not paraphrase. It MUST NAME BOTH the subject and the "
        "object -- a sentence mentioning only one of them does not state a "
        "relationship between them.\n"
        "  - WHEN THE STATING SENTENCE USES A PRONOUN OR ROLE for one of them "
        "(\"he\", \"the company\", \"her father\", \"the Gangs of New York "
        "director\"), extend the quote BACKWARDS to begin at the words that "
        "NAME that entity, and copy everything in between unchanged. Example: "
        "for \"Alameda Research is a hedge fund. Bankman-Fried founded it in "
        "2017.\" quote exactly that, not just \"Bankman-Fried founded it in "
        "2017.\" The quote must stay ONE continuous span of at most 3 "
        "sentences -- never skip text, never join separate passages. If the "
        "name is further back than that, do not emit the relationship.\n"
        "  - relation: ONLY when predicate_iri is "
        "https://veerla-ramrao.ai/ontology/graphrag#relatedTo -- a short verb "
        "phrase (2-6 words) taken from the quote, read subject -> object, "
        "saying HOW they relate: \"was born in\", \"alleges fraud "
        "against\", \"signed for\". Omit it for every other predicate.\n"
        "  - confidence: float in [0,1].\n\n"
        "RULES THAT MATTER:\n"
        "  - graphrag#relatedTo is the LAST resort. Use it ONLY when the "
        "passage explicitly states a relationship between the two AND no "
        "other listed predicate expresses it. It is never for two entities "
        "that merely appear together, sit in the same list, or share a "
        "topic -- the quote must still state how they relate.\n"
        "  - Assert ONLY what the passage states. Do NOT use world knowledge. "
        "Two entities appearing near each other is NOT a relationship.\n"
        "  - If you cannot quote a span that states it, do not emit it.\n"
        "  - READ YOUR OWN QUOTE BEFORE EMITTING. It must assert THIS claim in "
        "THIS direction. 'Lawsuits filed against A by B' means B sued A, NOT "
        "that A sued B. If the quote states the reverse, swap subject and "
        "object or drop the relationship.\n"
        "  - Prefer the MOST SPECIFIC predicate that fits. Reach for a "
        "last-resort predicate only when no specific one applies.\n"
        "  - Each predicate shows [Domain -> Range]: the subject must be of the "
        "domain type and the object of the range type. Respect direction -- "
        "'A causes B' and 'B causes A' are different claims.\n"
        "  - RETURNING AN EMPTY LIST IS THE CORRECT ANSWER when the passage "
        "asserts none of these relationships. Do not force a match merely "
        "because a predicate is offered.\n"
        f"  - At most {max_relationships} relationships.\n\n"
        "BEFORE YOU FINISH -- completeness check:\n"
        "  Every rule above tells you when NOT to emit a relationship, and "
        "they all still hold. But leaving out a relationship the passage "
        "plainly states is just as much an error as inventing one, and it is "
        "the more common failure: a passage naming eight entities usually "
        "asserts more than two or three links between them.\n"
        "  So before you return, walk the ENTITIES list and ask of each one: "
        "does the passage say anything about how this entity stands to "
        "another entity on the list -- who employs or leads it, who makes or "
        "owns it, what it treats or causes or competes with, where it sits? "
        "If a quotable sentence names both and states the link, and a "
        "predicate fits, emit it. You are not being asked to guess; you are "
        "being asked not to stop early.\n"
        "  A person named alongside their employer, a product named alongside "
        "its maker, and a drug named alongside what it treats are the three "
        "most frequently missed."
        + QWEN_RELATIONSHIP_RULES
    )
    user = (
        "ENTITIES (subject/object must come from here):\n" + ent_lines
        + "\n\nPREDICATES (predicate_iri must come from here):\n" + pred_lines
        + ("\n\nALREADY FOUND (do NOT repeat these):\n"
           + ("\n".join(f"  - {f}" for f in already_found) or "  (none)")
           if already_found is not None else "")
        + "\n\nPASSAGE:\n```\n" + chunk_text + "\n```\n\n"
        'Return JSON: {"relationships": [{"subject": ..., "predicate_iri": ..., '
        '"object": ..., "evidence": ..., "confidence": ..., '
        '"relation": ... (graphrag#relatedTo only)}]}'
    )
    return system, user
