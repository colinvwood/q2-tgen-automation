from qiime2.plugin import Plugin

from . import __version__


plugin = Plugin(
    name="tgen-automation",
    version=__version__,
    website="https://qiime2.org",
    package="q2_tgen_automation",
    short_description="Reusable actions for automated amplicon workflows.",
    description=(
        "Sampling-depth selection and dada2 parameter selection amplicon "
        "workflow utilities."
    ),
)
