#!/usr/bin/env bash
# run_overlap_tests.sh
#
# Runs the reference-range validator (validator/ref_validator_injection.py --test) on the
# ambiguous tests whose value distributions overlap (BIMODAL_OVERLAP > 0 in
# data/injection_results.tsv), and collects everything into one shareable markdown file.
#
# Usage:
#   bash scripts_ref/run_overlap_tests.sh [OUT_DIR] [-- extra validator args]
#   bash scripts_ref/run_overlap_tests.sh /mnt/disks/data/kanta/inject/ref/overlap
#   bash scripts_ref/run_overlap_tests.sh OUT -- --input /path/engine.parquet
#   TESTS="p-tni neutrofiilit" bash scripts_ref/run_overlap_tests.sh OUT
#
# Output (OUT_DIR, default ./overlap_validation):
#   <test>.log                 full validator output per test
#   overlap_validation.md      one-line summary table + every log (share this one)

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="$HERE/validator/ref_validator_injection.py"

OUT_DIR="./overlap_validation"
if [[ $# -gt 0 && "$1" != "--" ]]; then OUT_DIR="$1"; shift; fi
[[ $# -gt 0 && "$1" == "--" ]] && shift
EXTRA=("$@")

# Ambiguous tests with BIMODAL_OVERLAP > 0 (run of 2026-11-04), highest overlap first.
DEFAULT_TESTS="u-happamuusaste neutrofiilit p-tni s-mypnabg u-prot s-chpnabm du-prot lymf
u-epitel mono s-klotsa s-dmklots s-prealb i-stathct b-hkr"
read -r -a TEST_LIST <<< "$(echo ${TESTS:-$DEFAULT_TESTS})"

mkdir -p "$OUT_DIR"
export PYTHONDONTWRITEBYTECODE=1
MD="$OUT_DIR/overlap_validation.md"
SUMMARY="$OUT_DIR/.summary_lines"
: > "$SUMMARY"

n=${#TEST_LIST[@]}; i=0
for t in "${TEST_LIST[@]}"; do
    i=$((i + 1))
    echo "[$i/$n] $t"
    log="$OUT_DIR/$t.log"
    python3 "$VALIDATOR" --test "$t" --out-dir "$OUT_DIR" "${EXTRA[@]}" > "$log" 2>&1
    rc=$?
    # The validator prints one line per test: "[   1/1] <test> <type> value=... → <correction>"
    line=$(grep -E '^\[ *[0-9]+/[0-9]+\] ' "$log" | grep -v 'cand=' | tail -n 1)
    if [[ $rc -ne 0 ]]; then
        line="ERROR (exit $rc): $(tail -n 1 "$log")"
    elif [[ -z "$line" ]]; then
        line="no summary line (see log)"
    fi
    printf '%s\t%s\n' "$t" "$line" >> "$SUMMARY"
done

{
    echo "# Reference-range validation: ambiguous tests with overlapping value distributions"
    echo
    echo "Generated $(date '+%Y-%m-%d %H:%M') on $(hostname) by \`scripts_ref/run_overlap_tests.sh\`."
    echo "Validator: \`scripts_ref/validator/ref_validator_injection.py --test <name> ${EXTRA[*]}\`"
    echo
    echo "How to read the summary line:"
    echo
    echo "- \`value=\`: units picked by the value engine per sub-distribution (low|high) and their outcome."
    echo "- \`ranges=\`: what the reference ranges of the unit-less rows say (SINGLE / MIXED / NO_EVIDENCE / REVIEW) and the share per unit."
    echo "- \`valid=\`: per value-engine row, whether the ranged rows in that value interval agree (PASS / WEAK / CONFLICT / NO_EVIDENCE / NO_PROFILE)."
    echo "- \`agree base/ref=\`: share of range-labelled rows where the value-engine rule / the range rule gives the range's unit."
    echo "- \`→\`: correction applied (none / unit / split / merge / cutoff / rescue)."
    echo
    echo "## Summary"
    echo
    echo '```'
    cut -f2 "$SUMMARY"
    echo '```'
    echo
    echo "## Full output per test"
    for t in "${TEST_LIST[@]}"; do
        echo
        echo "### $t"
        echo
        echo '```'
        cat "$OUT_DIR/$t.log"
        echo '```'
    done
} > "$MD"
echo
echo "Summary:"
cut -f2 "$SUMMARY"
rm -f "$SUMMARY"
echo
echo "Shareable report: $MD"
