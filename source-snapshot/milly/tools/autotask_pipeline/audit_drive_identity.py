#!/usr/bin/env python3
"""Find seeded Drive cases whose prompt pins identities injection cannot preserve.

The evaluator creates a new Google file for every ``seed.drive_files`` entry.
Its readiness probe proves that the new file exists, but the created file ID and
sheet gid are not the source ID/gid embedded in the user's original task.  Such
cases are unsuitable for SFT: a compliant agent should report the exact resource
as inaccessible, while a same-name fallback teaches identity substitution.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


RESOURCE_ID_RE = re.compile(
    r"(?i)(?:spreadsheet|sheet|document|doc|file|folder)"
    r"(?:\s+[\w-]+){0,3}\s*(?:\(|\[)?\s*"
    r"(?:id|spreadsheetid)\s*[:=]?\s*['\"]?([A-Za-z0-9_-]{20,})"
)
DOCS_URL_RE = re.compile(
    r"https?://docs\.google\.com/[^\s)]+?/d/([A-Za-z0-9_-]{20,})",
    re.IGNORECASE,
)
GID_RE = re.compile(r"(?i)\bgid\s*[:=]\s*(\d+)")


def load_ids(path: str) -> list[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip()]


def audit(items: Path, seeds: Path, selected_ids: set[str] | None = None):
    rows = []
    counts = Counter()
    seen = set()
    with items.open() as source:
        for line in source:
            item = json.loads(line)
            inp = item.get('input') or {}
            iid = str(inp.get('id') or '')
            if selected_ids is not None and iid not in selected_ids:
                continue
            seen.add(iid)
            counts['selected'] += 1
            cid = iid.rsplit('__r', 1)[0]
            seed_path = seeds / f'{cid}.json'
            if not seed_path.exists():
                counts['missing_seed'] += 1
                continue
            seed = json.loads(seed_path.read_text())
            drive_files = seed.get('drive_files') or []
            if not drive_files:
                continue
            counts['with_drive'] += 1
            question = str(inp.get('question') or '')
            seed_ids = {str(row.get('id')) for row in drive_files if row.get('id')}
            literal_seed_ids = sorted(resource_id for resource_id in seed_ids
                                      if resource_id in question)
            explicit_ids = sorted(set(RESOURCE_ID_RE.findall(question)) |
                                  set(DOCS_URL_RE.findall(question)))
            gids = sorted(set(GID_RE.findall(question)))
            reasons = []
            if literal_seed_ids:
                reasons.append('seed_drive_id_literal_in_prompt')
            if explicit_ids:
                reasons.append('exact_drive_identity_in_prompt')
            if gids:
                reasons.append('sheet_gid_in_prompt')
            if not reasons:
                continue
            counts['blocked'] += 1
            for reason in reasons:
                counts[reason] += 1
            rows.append({
                'id': iid,
                'arm': (item.get('metadata') or {}).get('arm'),
                'reasons': reasons,
                'prompt_resource_ids': explicit_ids,
                'seed_ids_in_prompt': literal_seed_ids,
                'prompt_gids': gids,
                'seed_drive_files': [row.get('filename') for row in drive_files],
            })
    if selected_ids is not None:
        missing = sorted(selected_ids - seen)
        if missing:
            raise SystemExit(f'⛔ ID 文件中有 {len(missing)} 条不在 items，示例:{missing[:3]}')
    return rows, counts


def write_lines(path: str, values: list[str]) -> None:
    if path:
        Path(path).write_text(''.join(f'{value}\n' for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', required=True)
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--ids', default='', help='只审计该 ID 文件；默认全部 items')
    parser.add_argument('--report', default='', help='JSON 审计报告')
    parser.add_argument('--bad-ids', default='', help='写出必须隔离的 ID')
    parser.add_argument('--safe-ids', default='', help='写出输入 ID 中可继续运行的 ID')
    parser.add_argument('--strict', action='store_true', help='发现身份错配即退出 2')
    args = parser.parse_args()

    requested = load_ids(args.ids)
    selected = set(requested) if args.ids else None
    blocked, counts = audit(Path(args.items), Path(args.seeds), selected)
    bad_ids = [row['id'] for row in blocked]
    bad_set = set(bad_ids)
    safe_ids = [iid for iid in requested if iid not in bad_set]
    result = {
        'items': str(Path(args.items).resolve()),
        'seeds': str(Path(args.seeds).resolve()),
        'counts': dict(counts),
        'blocked': blocked,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    write_lines(args.bad_ids, bad_ids)
    write_lines(args.safe_ids, safe_ids)
    print('Drive identity audit: '
          f"selected={counts['selected']} drive={counts['with_drive']} "
          f"blocked={counts['blocked']}")
    if blocked:
        for row in blocked[:10]:
            print(f"  ⛔ {row['id']}: {','.join(row['reasons'])}")
        if args.strict:
            raise SystemExit(2)
    else:
        print('✅ 没有注入后无法保持的 Drive 精确身份')


if __name__ == '__main__':
    main()
