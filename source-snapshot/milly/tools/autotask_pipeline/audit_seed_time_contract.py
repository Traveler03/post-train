#!/usr/bin/env python3
"""Fail closed when frozen seeds still depend on runner-local time.

Generated cron worlds author mail/calendar times in the source user's timezone.
The evaluator resolves relative seed specs in Asia/Shanghai, so leaving them
relative can shift facts by up to fifteen hours and across a date boundary.
Assembly must materialize those specs as absolute ISO instants first.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timedelta
from email.utils import parseaddr
from pathlib import Path


def offset_seconds(value: str) -> int:
    match = re.fullmatch(r'([+-])(\d{2}):(\d{2})', str(value or ''))
    if not match:
        raise ValueError(f'invalid metadata tz_offset: {value!r}')
    seconds = (int(match.group(2)) * 60 + int(match.group(3))) * 60
    return seconds if match.group(1) == '+' else -seconds


def validate_abs(spec, expected_offset: int) -> str | None:
    if not isinstance(spec, dict) or not spec.get('abs'):
        return 'relative_or_missing'
    try:
        instant = datetime.fromisoformat(str(spec['abs']).replace('Z', '+00:00'))
    except (TypeError, ValueError):
        return 'invalid_iso'
    actual = instant.utcoffset()
    if actual is None:
        return 'naive_iso'
    if actual != timedelta(seconds=expected_offset):
        return 'offset_mismatch'
    return None


def participant_address(value: object, target_email: str) -> str:
    if isinstance(value, dict):
        value = value.get('email')
    raw = str(value or '').strip()
    if re.match(r'^self(?:\b|\s*\()', raw, re.IGNORECASE):
        return target_email.lower()
    angle = re.search(r'<\s*([^<>\s]+@[^<>\s]+)\s*>', raw)
    return (angle.group(1) if angle else parseaddr(raw)[1]).strip().lower()


def safe_calendar_address(address: str, target_email: str) -> bool:
    if not address or '@' not in address:
        return False
    if address == target_email.lower():
        return bool(re.search(r'(testbeeai|onboarding_testmigoo)', address, re.I))
    if re.fullmatch(r'[^@]*(testbeeai|onboarding_testmigoo)[^@]*@shopee\.com',
                    address, re.I):
        return True
    domain = address.rsplit('@', 1)[1]
    return bool(re.search(
        r'(?:^|\.)(?:test|invalid|example)$|(?:^|\.)example\.(?:com|net|org)$',
        domain, re.I,
    ))


def audit(items_file: Path, seeds_dir: Path) -> dict:
    counts = Counter()
    examples: list[dict] = []
    for line in items_file.open():
        item = json.loads(line)
        metadata = item.get('metadata') or {}
        is_cron = metadata.get('trigger_type') == 'cron'
        run_anytime_event = (metadata.get('trigger_type') == 'event' and
                             bool((metadata.get('temporal_contract') or {}).get(
                                 'run_anytime')))
        requires_absolute = is_cron or run_anytime_event
        if not is_cron:
            counts['event_or_other'] += 1
            expected_offset = 8 * 3600 if run_anytime_event else None
            if run_anytime_event:
                counts['run_anytime_event'] += 1
        else:
            counts['cron'] += 1
            try:
                expected_offset = offset_seconds(metadata.get('tz_offset'))
            except ValueError:
                counts['invalid_metadata_timezone'] += 1
                if len(examples) < 10:
                    examples.append({'id': (item.get('input') or {}).get('id'),
                                     'problem': 'invalid_metadata_timezone',
                                     'tz_offset': metadata.get('tz_offset')})
                continue
        seed_name = str(metadata.get('seed_data_file') or '').rsplit('/', 1)[-1]
        seed_path = seeds_dir / seed_name
        if not seed_name or not seed_path.exists():
            counts['missing_seed'] += 1
            if len(examples) < 10:
                examples.append({'id': (item.get('input') or {}).get('id'),
                                 'problem': 'missing_seed', 'seed': seed_name})
            continue
        seed = json.loads(seed_path.read_text())
        problems = []
        # ⚠️ 0825:题目默认不再写死 user_email(交给平台账号池分配),这里会拿到空串。
        #    空串会让种子里的 `self` 解析成 ''、`safe_calendar_address` 的自身分支永不命中
        #    ⇒ **安全域检查静默失效**(闸还在跑、永远不报)。所以:
        #    ①允许用 --assume-self-email 显式给一个代表号做形态校验;
        #    ②都没有时**明说跳过**并计数,绝不假装检查过。
        target_email = str(metadata.get('user_email') or args.assume_self_email or '').lower()
        if not target_email:
            counts['skipped_no_account'] = counts.get('skipped_no_account', 0) + 1
        if requires_absolute:
            for email in seed.get('emails') or []:
                spec = email.get('date_offset')
                problem = validate_abs(spec, expected_offset)
                if problem == 'relative_or_missing':
                    counts['relative_email_times'] += 1
                    problems.append(f"email:{email.get('id')}")
                elif problem:
                    counts[f'email_{problem}'] += 1
                    problems.append(f"email:{email.get('id')}:{problem}")
        for event in seed.get('calendar_events') or []:
            if event.get('attendees'):
                counts['attendee_events'] += 1
                if not event.get('organizer'):
                    counts['attendee_events_without_import_organizer'] += 1
                    problems.append(
                        f"calendar:{event.get('id')}:attendees_without_import_organizer"
                    )
                else:
                    counts['attendee_events_private_import'] += 1
            participants = list(event.get('attendees') or [])
            if event.get('organizer'):
                participants.append(event['organizer'])
            for participant in participants:
                raw_participant = (participant.get('email')
                                   if isinstance(participant, dict) else participant)
                if re.match(r'^self(?:\b|\s*\()', str(raw_participant or '').strip(),
                            re.IGNORECASE):
                    counts['unresolved_calendar_self'] += 1
                    problems.append(
                        f"calendar:{event.get('id')}:unresolved_self"
                    )
                address = participant_address(participant, target_email)
                counts['calendar_participant_addresses'] += 1
                if not safe_calendar_address(address, target_email):
                    counts['unsafe_calendar_participants'] += 1
                    problems.append(
                        f"calendar:{event.get('id')}:unsafe_participant:{address or participant!r}"
                    )
            if event.get('all_day'):
                counts['all_day_events'] += 1
                continue
            if requires_absolute:
                for key in ('start', 'end'):
                    spec = event.get(key)
                    problem = validate_abs(spec, expected_offset)
                    if problem == 'relative_or_missing':
                        counts['relative_calendar_times'] += 1
                        problems.append(f"calendar:{event.get('id')}.{key}")
                    elif problem:
                        counts[f'calendar_{problem}'] += 1
                        problems.append(f"calendar:{event.get('id')}.{key}:{problem}")
        if problems:
            counts['cases_with_seed_contract_problems'] += 1
            if requires_absolute and any('unsafe_participant' not in problem for problem in problems):
                counts['cron_cases_with_relative_time'] += 1
            if len(examples) < 10:
                examples.append({'id': (item.get('input') or {}).get('id'),
                                 'problem': 'seed_time_or_participant_contract',
                                 'fields': problems[:8]})
        elif requires_absolute:
            counts['cron_cases_absolute' if is_cron
                   else 'run_anytime_event_cases_absolute'] += 1
    failed = sum(value for key, value in counts.items()
                 if key == 'missing_seed' or key == 'invalid_metadata_timezone'
                 or key.startswith('relative_') or key.endswith('_invalid_iso')
                 or key.endswith('_naive_iso') or key.endswith('_offset_mismatch')
                 or key in {'unsafe_calendar_participants',
                            'unresolved_calendar_self',
                            'attendee_events_without_import_organizer'})
    return {
        'schema_version': 1,
        'passed': failed == 0,
        'items_file': str(items_file),
        'seeds_dir': str(seeds_dir),
        'counts': dict(counts),
        'examples': examples,
        'note': ('every attendee-bearing event must carry a safe synthetic organizer, '
                 'forcing the deployed evaluator onto events.import private-copy mode'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--items', required=True)
    parser.add_argument('--assume-self-email', default='migoo_testbeeai_59@shopee.com',
                        help='题目不写死账号时,用它把种子里的 self 解析出来做形态校验(不代表实跑账号)')
    parser.add_argument('--seeds', required=True)
    parser.add_argument('--out', default='')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    result = audit(Path(args.items), Path(args.seeds))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + '\n')
    if args.strict and not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
