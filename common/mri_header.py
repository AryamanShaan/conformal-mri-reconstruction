"""
mri_header.py

ISMRMRD XML header query helper, extracted verbatim from /VarNet's
data/data_module.py (the same function also lives in /fastMRI's data/mri_data.py).

Lifted into common/ because BOTH e2evarnet (SliceDataset metadata parsing) and
quantile (inference/varnet_ssim_score_2.py) need `et_query`, without either
having to import the whole data_module.
"""

import xml.etree.ElementTree as etree
from typing import Sequence


def et_query(
    root: etree.Element,
    qlist: Sequence[str],
    namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    prefix = "ns"
    ns = {prefix: namespace}
    query = "."

    for el in qlist:
        query += f"//{prefix}:{el}"

    value = root.find(query, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)
