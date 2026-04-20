import re

import malaya_tagging_pipeline


def test_version() -> None:
    version = malaya_tagging_pipeline.__version__
    assert isinstance(version, str)
    assert re.search(r"\d+\.\d+\.\d+$", version) is not None