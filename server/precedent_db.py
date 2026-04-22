"""
UndertriAI — Precedent Database
Lightweight keyword-based retrieval of landmark bail judgments.
"""

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Landmark precedents (embedded — no external DB needed)
# ---------------------------------------------------------------------------

PRECEDENTS = [
    {
        "title": "Sanjay Chandra v. CBI, (2012) 1 SCC 40",
        "summary": "SC held that bail is the rule and jail is the exception. Personal liberty under Article 21 cannot be sacrificed without compelling reasons. Courts must not treat pre-trial detention as punishment.",
        "keywords": ["personal liberty","article 21","rule","exception","pre-trial","punishment","default"],
        "crime_categories": ["fraud","white collar","economic","cheating"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Arnesh Kumar v. State of Bihar, (2014) 8 SCC 273",
        "summary": "Arrest is not mandatory for offences punishable up to 7 years. Police must record reasons before arrest. Courts must apply mind before granting remand. Safeguard against mechanical arrests.",
        "keywords": ["arrest","seven years","remand","mechanical","magistrate","section 41","non-bailable"],
        "crime_categories": ["498a","cruelty","domestic","cheating","fraud"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "State of Rajasthan v. Balchand, (1977) 4 SCC 308",
        "summary": "Bail is the rule, jail is the exception. The court must consider: (1) nature of accusation, (2) nature of evidence, (3) severity of punishment, (4) character of accused, (5) danger of accused fleeing.",
        "keywords": ["triple test","flight risk","evidence","accusation","severity","character"],
        "crime_categories": ["all"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Nikesh Tarachand Shah v. Union of India, (2018) 11 SCC 1",
        "summary": "Twin conditions under PMLA bail provisions struck down as unconstitutional. Bail cannot be denied mechanically. Even money-laundering accused entitled to bail if conditions are met.",
        "keywords": ["pmla","money laundering","twin conditions","unconstitutional","special law"],
        "crime_categories": ["money laundering","financial","pmla"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "P. Chidambaram v. Directorate of Enforcement, (2020) 13 SCC 441",
        "summary": "Anticipatory bail: Court must consider nature and gravity of accusation, antecedents, possibility of fleeing, and whether accusation is bona fide or mala fide. Economic offences of serious magnitude require stricter scrutiny.",
        "keywords": ["anticipatory","economic offence","gravity","antecedents","bona fide","mala fide","magnitude"],
        "crime_categories": ["fraud","economic","cheating","corruption"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Satender Kumar Antil v. CBI, (2022) 10 SCC 51",
        "summary": "Landmark ruling on undertrial prisoners. Default bail under Section 436A must be given without unnecessary delay. Courts have duty to suo motu consider bail for long-incarcerated undertrials.",
        "keywords": ["undertrial","default bail","436a","long custody","section 436","half sentence","suo motu"],
        "crime_categories": ["all"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Gudikanti Narasimhulu v. Public Prosecutor, (1978) 1 SCC 240",
        "summary": "Court must balance personal liberty against public interest. Factors: (a) nature of accusation, (b) punishment, (c) character of evidence, (d) possibility of repeating offence, (e) danger of fleeing.",
        "keywords": ["balance","public interest","punishment","repeat","character"],
        "crime_categories": ["all"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Union of India v. K.A. Najeeb, (2021) 3 SCC 713",
        "summary": "Courts can grant bail even under UAPA if incarceration extends for inordinate period. Constitutional courts must step in to protect fundamental rights despite statutory restrictions.",
        "keywords": ["uapa","inordinate","fundamental rights","constitutional","prolonged","statutory restriction"],
        "crime_categories": ["terror","uapa","security"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Moti Ram v. State of M.P., (1978) 4 SCC 47",
        "summary": "Court must not fix excessively high surety amounts that effectively deny bail to the poor. Surety should be commensurate with the accused's economic status. Poverty cannot be a ground for denial.",
        "keywords": ["surety","excessive","poor","economic","poverty","amount","commensurate"],
        "crime_categories": ["all"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Babu Singh v. State of Uttar Pradesh, (1978) 1 SCC 579",
        "summary": "Personal freedom is important. Bail should not be denied as punishment or to teach a lesson. Courts must not be influenced by the seriousness of the charge alone. Conditions of bail must be reasonable.",
        "keywords": ["personal freedom","punishment","conditions","lesson","reasonable","seriousness alone"],
        "crime_categories": ["all"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Siddharam Satlingappa Mhetre v. State of Maharashtra, (2011) 1 SCC 694",
        "summary": "Comprehensive ruling on anticipatory bail. Court enumerated 10 factors: nature of accusation, antecedents, possibility of fleeing, social standing, possibility of repeating offence, primacy of investigation.",
        "keywords": ["anticipatory","ten factors","investigation","social standing","repeat","antecedents"],
        "crime_categories": ["anticipatory","pre-arrest"],
        "jurisdiction": "Supreme Court",
    },
    {
        "title": "Anil Kumar Yadav v. State (NCT of Delhi), (2018) 12 SCC 129",
        "summary": "Parity in bail: When co-accused in similar position are granted bail, denial to one accused requires specific distinguishing factors. Parity principle is well-established.",
        "keywords": ["parity","co-accused","similar position","distinguishing","differentiate"],
        "crime_categories": ["all"],
        "jurisdiction": "Supreme Court",
    },
]


class PrecedentDB:
    """
    Keyword-based precedent retrieval.
    Returns relevant landmark judgments based on query and case context.
    """

    def __init__(self):
        self._precedents = PRECEDENTS

    def search(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        crime_category: Optional[str] = None,
        top_k: int = 3,
    ) -> List[str]:
        """Return top_k relevant precedent citations."""
        q = query.lower()
        scored = []
        for p in self._precedents:
            score = 0
            # Keyword match
            for kw in p["keywords"]:
                if kw in q:
                    score += 2
            # Crime category match
            if crime_category:
                for cat in p["crime_categories"]:
                    if cat in crime_category.lower() or cat == "all":
                        score += 3
            # Jurisdiction filter
            if jurisdiction:
                if jurisdiction.lower() in p["jurisdiction"].lower():
                    score += 1
            # Always include general principles (low score)
            if "all" in p["crime_categories"]:
                score += 0.5
            scored.append((score, p))

        scored.sort(key=lambda x: -x[0])
        results = []
        for score, p in scored[:top_k]:
            if score > 0:
                results.append(f"{p['title']}: {p['summary'][:160]}...")
        return results

    def get_initial_precedents(self, episode: Dict[str, Any]) -> List[str]:
        """Return 1–2 baseline precedents relevant to the case type."""
        crime = episode.get("crime_type", "").lower()
        bail_type = episode.get("bail_type", "Regular")
        query = f"{crime} {bail_type} bail"
        return self.search(query=query, crime_category=crime, top_k=2)
