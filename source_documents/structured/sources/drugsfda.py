"""Drugs@FDA: openFDA drug/drugsfda API and the FDA tab-delimited data files."""

from __future__ import annotations

from .._base import register
from .openfda import OpenFdaSource


@register
class DrugsFdaSource(OpenFdaSource):
    name = "drugsfda"
    title = "Drugs@FDA"
    domain = "pharma"
    summary = ("FDA-approved drug applications: openFDA records (API) and the official "
               "Drugs@FDA data files (12 tab-delimited tables).")
    endpoint = "drug/drugsfda"
    flatfile_url = "https://www.fda.gov/media/89850/download?attachment"
    flatfile_delimiter = "\t"
    record_key = ("application_number",)
    modes = OpenFdaSource.build_modes("drug/drugsfda", "Drugs@FDA data files zip: Applications, Products, "
                                       "Submissions, ApplicationDocs, MarketingStatus, TE + "
                                       "lookups (tab-delimited)")
    docs_urls = ["https://open.fda.gov/apis/drug/drugsfda/",
                 "https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files"]
    data_dictionaries = [
        "https://www.fda.gov/drugs/drug-approvals-and-databases/drugsfda-data-files"]
    attribution = "U.S. Food and Drug Administration (Drugs@FDA, openFDA)"
    notes = ["Application numbers keep their leading zeros (all values are text).",
             "Document URLs (ApplicationDocs / application_docs) are preserved verbatim."]
