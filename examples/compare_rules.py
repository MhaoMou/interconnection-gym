"""Compare allocation rules on an IEEE 14-bus batch, as an operator would.

    python examples/compare_rules.py [report.html]
"""

import sys

from icgym import PerMWCommitment, ieee14
from icgym.compare import compare_rules, format_table, report_html

region = ieee14((9, 13, 14))            # bundled batch region of buses 9, 13, 14
reports = compare_rules(region, ["uniform", "proportional", "utilitarian"],
                        fee=PerMWCommitment(1 / 2.46), value=1.0, beta=0.0)
print(format_table(reports, region.applicant_names))
if len(sys.argv) > 1:
    report_html(reports, sys.argv[1], subtitle="IEEE 14-bus, buses 9, 13, 14", applicant_names=region.applicant_names)
    print("wrote", sys.argv[1])
