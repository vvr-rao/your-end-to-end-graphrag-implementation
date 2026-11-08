"""FDA Orange Book: openFDA drug/orangebook API and the tilde-delimited data files."""

from __future__ import annotations

from .._base import register
from .openfda import OpenFdaSource


@register
class OrangeBookSource(OpenFdaSource):
    name = "orangebook"
    title = "FDA Orange Book"
    domain = "pharma"
    summary = ("Approved drug products with therapeutic-equivalence evaluations, patents and "
               "exclusivities: openFDA records (API) and the official data files "
               "(Products / Patent / Exclusivity, tilde-delimited).")
    endpoint = "drug/orangebook"
    flatfile_url = "https://www.fda.gov/media/76860/download?attachment"
    flatfile_delimiter = "~"
    record_key = ("product_number", "approval_date")
    modes = OpenFdaSource.build_modes("drug/orangebook", "Orange Book data files zip: Products.txt, Patent.txt, "
                                       "Exclusivity.txt (tilde-delimited)")
    docs_urls = ["https://open.fda.gov/apis/drug/orangebook/",
                 "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"]
    data_dictionaries = [
        "https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files"]
    attribution = "U.S. Food and Drug Administration (Orange Book, openFDA)"
    notes = ["Appl_No / Product_No keep leading zeros (all values are text).",
             "openFDA's orangebook record has no single application key at the top level; "
             "normalized child rows carry product_number + approval_date + ordinals."]
