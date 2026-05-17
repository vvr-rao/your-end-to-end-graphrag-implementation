## Drop your input ontologies here

# Enterprise Ontology Reference Guide

This document provides a concise reference list of useful ontologies for enterprise knowledge graphs and GraphRAG systems.

---

# 1. General / Foundational Ontologies

These ontologies provide core semantic standards, metadata, provenance, taxonomy, geography, and time modeling.

| Ontology | Purpose | Approx. Size | Download |
|---|---|---|---|
| Dublin Core | Document metadata, authorship, provenance | ~200 triples | https://www.dublincore.org/specifications/dublin-core/dcmi-terms/ |
| FOAF | People, organizations, social relationships | ~500 triples | http://xmlns.com/foaf/spec/ |
| SKOS | Taxonomies and controlled vocabularies | ~1K triples | https://www.w3.org/TR/skos-reference/ |
| PROV-O | Provenance and lineage tracking | ~2K triples | https://www.w3.org/TR/prov-o/ |
| OWL Time | Dates, time intervals, temporal events | ~1K triples | https://www.w3.org/TR/owl-time/ |
| WGS84 Geo Vocabulary | Geography and coordinate modeling | ~500 triples | https://www.w3.org/2003/01/geo/ |
| DBpedia Ontology | General-purpose knowledge graph ontology | ~100K+ triples | https://downloads.dbpedia.org/repo/dbpedia.org/ontology--DEV/ |
| QUDT | Units, quantities, measurements | ~50K+ triples | https://qudt.org/ |
| DOLCE | Foundational ontology framework | ~10K triples | https://www.loa.istc.cnr.it/dolce/ |
| UFO | Enterprise foundational ontology | ~10K triples | https://nemo.inf.ufes.br/en/projetos/ufo/ |

---

# 2. Pharma / Healthcare Domain Ontologies

These ontologies support healthcare, clinical trials, drugs, diseases, treatments, and biomedical relationships.

| Ontology | Purpose | Approx. Size | Download |
|---|---|---|---|
| OCRe | Clinical research ontology | ~100K triples | https://github.com/OCRediT/OCRe (I found the best place is to manually download the .zip from https://bioportal.bioontology.org/ontologies/OCRE) |
| DRON | Drug ontology | ~1M+ triples | https://github.com/ufbmi/DiseaseOntology/tree/master/src/ontology/releases |
| HPO | Human phenotype ontology | ~500K triples | https://hpo.jax.org/app/download/ontology |
| MAXO | Medical actions/treatments | ~100K triples | https://github.com/monarch-initiative/MAxO |
| SNOMED CT | Comprehensive clinical terminology | 10M+ triples | https://www.snomed.org/snomed-ct/get-snomed-ct |
| RxNorm | Standardized drug terminology | ~1M triples | https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html |
| MONDO | Disease ontology integration | ~500K triples | https://mondo.monarchinitiative.org/ |
| NCI Thesaurus (NCIT) | Cancer and biomedical concepts | ~2M triples | https://evs.nci.nih.gov/ftp1/NCI_Thesaurus/ |
| Gene Ontology (GO) | Gene function ontology | ~1M triples | https://geneontology.org/docs/download-ontology/ |

---

# 3. Finance Domain Ontologies

These ontologies model financial entities, instruments, contracts, securities, organizations, and markets.

| Ontology | Purpose | Approx. Size | Download |
|---|---|---|---|
| FIBO | Enterprise financial ontology | 5M+ triples | https://github.com/edmcouncil/fibo ( https://spec.edmcouncil.org/fibo/page/owl click on "Download FIBO OWL") |
| FIGI | Financial instrument identifiers | ~100K triples | https://www.openfigi.com/ |
| ISO 20022 Ontologies | Financial messaging concepts | ~500K triples | https://www.iso20022.org/ |
| XBRL Taxonomies | Financial reporting semantics | ~1M triples | https://specifications.xbrl.org/ |
| OMG Financial Ontologies | Finance and regulatory semantics | ~100K triples | https://www.omg.org/spec/ |

Some FIBO Modules:
- fibo-fnd-* (Foundation)
- fibo-be-* (Business entities)
- fibo-sec-* (Securities)
- fibo-fbc-* (Financial business & commerce)

---

# 4. Manufacturing / Supply Chain Ontologies

These ontologies support industrial systems, materials, products, IoT, telemetry, logistics, and supply-chain relationships.

| Ontology | Purpose | Approx. Size | Download |
|---|---|---|---|
| OntoCAPE | Industrial process and chemical engineering ontology | ~500K triples | https://www.avt.rwth-aachen.de/cms/AVT/Wirtschaft/SoftwareSimulation/~ipts/OntoCape/ ( https://www.avt.rwth-aachen.de/Ontocape click on "ontocape domin ontology")|
| SAREF | Smart manufacturing and IoT ontology | ~50K triples | https://saref.etsi.org/ |
| SSN/SOSA | Sensors and observations ontology | ~10K triples | https://www.w3.org/TR/vocab-ssn/ |
| GoodRelations | Products, commerce, suppliers | ~20K triples | http://www.heppnetz.de/projects/goodrelations/ |
| schema.org | General business/product semantics | ~100K triples | https://schema.org/ |
| QUDT | Measurements, units, quantities | ~50K triples | https://qudt.org/ |
| BOT | Building topology ontology | ~10K triples | https://w3c-lbd-cg.github.io/bot/ |
| Brick Schema | Smart buildings and industrial systems | ~200K triples | https://brickschema.org/ |

---
# Notes

- Many ontology URLs redirect and require HTTP headers.
- Some ontologies are distributed as ZIP files.
- SNOMED CT may require licensing depending on country/jurisdiction.
- Large ontologies (FIBO, SNOMED, DRON) may require significant RAM.
- Prefer local ontology imports where possible.

Example:
```python
get_ontology(path).load(only_local=True)