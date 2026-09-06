#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Install PAIR's systemd user unit without starting it.

Usage:
  scripts/install-systemd-user-service.sh --bin-dir DIR [--dry-run]

Options:
  --bin-dir DIR  Directory containing a complete PAIR service bundle.
  --dry-run      Render the unit to stdout without writing or reloading systemd.
  -h, --help     Show this help.

The installer never enables or starts PAIR. Review the rendered unit, stop any
desktop or TUI-owned PAIR process tree, then enable it explicitly with:

  systemctl --user enable --now nvpair.service
EOF
}

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

bin_dir=''
dry_run=false
while (($# > 0)); do
    case "$1" in
        --bin-dir)
            (($# >= 2)) || fail '--bin-dir requires a value'
            bin_dir=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1"
            ;;
    esac
done

[[ -n "$bin_dir" ]] || fail '--bin-dir is required'
[[ -d "$bin_dir" ]] || fail "binary directory does not exist: $bin_dir"
bin_dir=$(cd "$bin_dir" && pwd -P)

# The template substitutes the path directly into systemd directive values.
# Restrict it to an unambiguous portable subset rather than attempting to quote
# every character systemd and sed interpret differently.
case "$bin_dir" in
    *[!A-Za-z0-9_./+-]*)
        fail "binary directory contains unsupported characters: $bin_dir"
        ;;
esac

[[ -x "$bin_dir/nvpair-ui-broker" ]] || fail "missing executable: $bin_dir/nvpair-ui-broker"
[[ -x "$bin_dir/nvpair-node-scanner" ]] || fail "missing required executable: $bin_dir/nvpair-node-scanner"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd "$script_dir/.." && pwd -P)
template="$repo_dir/services/installer/linux/nvpair.service.tmpl"
[[ -f "$template" ]] || fail "unit template not found: $template"

render_unit() {
    sed "s|@NVPAIR_BIN_DIR@|$bin_dir|g" "$template"
}

if [[ "$dry_run" == true ]]; then
    render_unit
    exit 0
fi

command -v systemctl >/dev/null 2>&1 || fail 'systemctl is required'
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
unit_path="$unit_dir/nvpair.service"
mkdir -p -m 0700 "$unit_dir"

if [[ -e "$unit_path" ]] && ! grep -Fq 'Managed by scripts/install-systemd-user-service.sh.' "$unit_path"; then
    fail "refusing to overwrite unmanaged unit: $unit_path"
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT
rendered="$tmp_dir/nvpair.service"
render_unit > "$rendered"

if command -v systemd-analyze >/dev/null 2>&1; then
    systemd-analyze --user verify "$rendered"
fi

install -m 0644 "$rendered" "$unit_path"
systemctl --user daemon-reload

printf 'Installed %s\n' "$unit_path"
printf 'PAIR was not started. Review the unit, stop any desktop/TUI PAIR instance, then run:\n'
printf '  systemctl --user enable --now nvpair.service\n'
