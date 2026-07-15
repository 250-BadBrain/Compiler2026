#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_dir="${TMPDIR:-/tmp}/compiler2026-lexer-test"
mkdir -p "$tmp_dir"

make -C "$root" >/dev/null

run_dump() {
    local name="$1"
    "$root/compiler" --dump-tokens "$root/tests/lexer/$name.sy" >"$tmp_dir/$name.tokens"
}

require_token() {
    local name="$1"
    local pattern="$2"
    if ! grep -Eq "$pattern" "$tmp_dir/$name.tokens"; then
        echo "missing token pattern in $name: $pattern" >&2
        echo "--- tokens ---" >&2
        cat "$tmp_dir/$name.tokens" >&2
        exit 1
    fi
}

run_dump token_basic
require_token token_basic 'KeywordConst const'
require_token token_basic 'KeywordInt int'
require_token token_basic 'KeywordFloat float'
require_token token_basic 'KeywordVoid void'
require_token token_basic 'KeywordIf if'
require_token token_basic 'LessEqual <='
require_token token_basic 'AmpAmp &&'
require_token token_basic 'NotEqual !='
require_token token_basic 'KeywordElse else'
require_token token_basic 'KeywordWhile while'
require_token token_basic 'KeywordBreak break'
require_token token_basic 'KeywordContinue continue'

run_dump comment
require_token comment 'KeywordInt int'
require_token comment 'KeywordReturn return'
require_token comment 'IntLiteral 0'
if grep -Eq 'comment' "$tmp_dir/comment.tokens"; then
    echo "comments leaked into token stream" >&2
    cat "$tmp_dir/comment.tokens" >&2
    exit 1
fi

run_dump number
require_token number 'IntLiteral 123'
require_token number 'IntLiteral 077'
require_token number 'IntLiteral 0x2a'
require_token number 'FloatLiteral 1\.0'
require_token number 'FloatLiteral \.5'
require_token number 'FloatLiteral 1\.'
require_token number 'FloatLiteral 1e-3'
require_token number 'FloatLiteral 0x1\.8p\+1'

run_dump string
require_token string 'StringLiteral "value=%d\\n"'

echo "lexer tests passed"
