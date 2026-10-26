"""Publication tooling: the publish package and the public rebuild of every
figure from the shipped tables.

Module map:
    withheld       what is removed from Google Gemini and Google AI Mode rows
                   (Google's terms) and how the reader reads what is left
    bundle         the PUBLIC side: load the shipped tables, rebuild the
                   figures, compare with the shipped frozen copy
    docs           renders the package documents from config/publish_docs/
    private_build  owner only (does not ship): writes publish/ from the
                   frozen data root
"""
