#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""USPTO office-action label tables, step 1: resumable raw fetch.

Datasets (POST form body, X-API-KEY header):
  E = oa/enriched_cited_reference_metadata/v3   (one row per cited reference per action, dated by officeActionDate)
  R = oa/oa_rejections/v2                        (one row per rejection/objection paragraph, dated by submissionDate)
  C = oa/oa_citations/v2                         (one row per reference per section, NO date -> sliced by application number)

API limits found by probing: start+rows must be <= 10000 (deeper paging returns {}), sort/fl/facet are ignored.
So every query slice must hold <= 10000 rows.  Slices are made by bisecting the 8-digit patentApplicationNumber
range inside a day (E, R) or inside a 100K application-number block (C); the sum of slice counts is verified
against the unsplit count, and each unit's result is written to raw/<ds>/<unit>.parquet followed by a .json marker.
Re-running skips units that already have a marker.

Usage:  python build/oa_fetch.py run [E R C P]      (default: all, in that order; P = pre-scope rejections, only used for action_seq)
        python build/oa_fetch.py status
"""
import os, sys, json, time, threading, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import requests
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("PAB_SRC", "src") + '/uspto_oa'
RAW = ROOT + '/raw'
STATE = ROOT + '/state'
B = 'https://api.uspto.gov/api/v1/patent/oa/'
KEY = os.environ['USPTO_ODP_API_KEY']   # free USPTO Open Data Portal API key
LIMIT = 10000
WORKERS = 8
SCOPE_START = dt.date(2017, 3, 1)      # enriched data begins 2017-03-13
PRE_START = dt.date(2011, 10, 1)       # rejections before scope, for action_seq only
DSNAME = {'E': 'enriched_cited_reference_metadata/v3', 'R': 'oa_rejections/v2', 'C': 'oa_citations/v2'}
DATEFIELD = {'E': 'officeActionDate', 'R': 'submissionDate'}

S = pa.string()
SCHEMA = {
    'E': pa.schema([('oa_id', S), ('app', S), ('date', S), ('doc_code', S), ('art_unit', S), ('cat', S),
                    ('rel_claims', S), ('kind', S), ('country', S), ('cited_doc', S), ('pubnum', S),
                    ('passage', S), ('quality', S), ('ex_cited', pa.bool_()), ('app_cited', pa.bool_()),
                    ('npl', pa.bool_())]),
    'R': pa.schema([('oa_id', S), ('app', S), ('date', S), ('doc_code', S), ('art_unit', S), ('legal_raw', S),
                    ('cat_raw', S), ('claims_raw', S), ('para', S)]),
    'C': pa.schema([('oa_id', S), ('app', S), ('legal_raw', S), ('cat_raw', S), ('ref_raw', S), ('ref_parsed', S),
                    ('ex_cited', pa.bool_()), ('app_cited', pa.bool_()), ('art_unit', S), ('para', S)]),
}


def s(v):
    return None if v is None else str(v)


def conv(ds, d):
    if ds == 'E':
        p = d.get('passageLocationText')
        if isinstance(p, list):
            p = ' || '.join(p)
        return dict(oa_id=s(d.get('obsoleteDocumentIdentifier')), app=s(d.get('patentApplicationNumber')),
                    date=(d.get('officeActionDate') or '')[:10], doc_code=s(d.get('officeActionCategory')),
                    art_unit=s(d.get('groupArtUnitNumber')), cat=s(d.get('citationCategoryCode')),
                    rel_claims=s(d.get('relatedClaimNumberText')), kind=s(d.get('kindCode')),
                    country=s(d.get('countryCode')), cited_doc=s(d.get('citedDocumentIdentifier')),
                    pubnum=s(d.get('publicationNumber')), passage=p, quality=s(d.get('qualitySummaryText')),
                    ex_cited=d.get('examinerCitedReferenceIndicator'),
                    app_cited=d.get('applicantCitedExaminerReferenceIndicator'), npl=d.get('nplIndicator'))
    if ds == 'R':
        c = d.get('claimNumberArrayDocument')
        c = ','.join(c) if isinstance(c, list) else s(c)
        return dict(oa_id=s(d.get('obsoleteDocumentIdentifier')), app=s(d.get('patentApplicationNumber')),
                    date=(d.get('submissionDate') or '')[:10], doc_code=s(d.get('legacyDocumentCodeIdentifier')),
                    art_unit=s(d.get('groupArtUnitNumber')), legal_raw=s(d.get('legalSectionCode')),
                    cat_raw=s(d.get('actionTypeCategory')), claims_raw=c, para=s(d.get('paragraphNumber')))
    return dict(oa_id=s(d.get('obsoleteDocumentIdentifier')), app=s(d.get('patentApplicationNumber')),
                legal_raw=s(d.get('legalSectionCode')), cat_raw=s(d.get('actionTypeCategory')),
                ref_raw=s(d.get('referenceIdentifier')), ref_parsed=s(d.get('parsedReferenceIdentifier')),
                ex_cited=d.get('examinerCitedReferenceIndicator'),
                app_cited=d.get('applicantCitedExaminerReferenceIndicator'),
                art_unit=s(d.get('groupArtUnitNumber')), para=s(d.get('paragraphNumber')))


# ---------------------------------------------------------------- http
_tl = threading.local()
_lock = threading.Lock()
PAUSE_UNTIL = [0.0]
STATS = dict(calls=0, retries=0, rows=0, units=0, errors=0, t0=time.time(), r429=0)


def sess():
    if not hasattr(_tl, 's'):
        _tl.s = requests.Session()
        _tl.s.headers['X-API-KEY'] = KEY
    return _tl.s


def call(ds, crit, rows, start=0):
    """POST one query; retry on transient failures; returns the 'response' dict."""
    last = None
    n403 = 0
    for attempt in range(14):
        w = PAUSE_UNTIL[0] - time.time()
        if w > 0:
            time.sleep(w)                                   # global back-off set by any worker that saw 429/403
        try:
            r = sess().post(B + DSNAME[ds] + '/records', data={'criteria': crit, 'start': start, 'rows': rows},
                            timeout=180)
            with _lock:
                STATS['calls'] += 1
            if r.status_code == 429:
                with _lock:
                    STATS['r429'] += 1
                PAUSE_UNTIL[0] = max(PAUSE_UNTIL[0], time.time() + min(float(r.headers.get('Retry-After', 10 * (attempt + 1))), 120))
                last = '429'
                continue
            if r.status_code in (401, 403):
                # seen once after ~1h: a transient 403 with a valid key. Pause everybody, retry; give up after ~15 min.
                n403 += 1
                PAUSE_UNTIL[0] = max(PAUSE_UNTIL[0], time.time() + 60 * min(n403, 5))
                print('HTTP %s (%d) pausing all workers: %s' % (r.status_code, n403, r.text[:80]), flush=True)
                if n403 >= 8:
                    raise SystemExit('auth failure %s persisted: %s' % (r.status_code, r.text[:200]))
                last = str(r.status_code)
                continue
            if r.status_code == 200:
                j = r.json()
                if 'response' in j:
                    return j['response']
                last = 'empty-body ' + r.text[:80]
            else:
                last = '%s %s' % (r.status_code, r.text[:120])
        except SystemExit:
            raise
        except Exception as e:  # network / json
            last = repr(e)[:160]
        with _lock:
            STATS['retries'] += 1
        time.sleep(min(2 ** attempt, 60))
    raise RuntimeError('call failed: %s | %s' % (last, crit[:160]))


def count(ds, crit):
    return call(ds, crit, 0)['numFound']


# ---------------------------------------------------------------- slicing
def rng(base, lo, hi):
    a = 'patentApplicationNumber:[%08d TO %08d]' % (lo, hi)
    return (base + ' AND ' + a) if base else a


def plan(ds, base, lo, hi, n_all):
    """Split [lo,hi] application numbers into slices of <= LIMIT rows. n_all = known count of the whole range."""
    out = []
    stack = [(lo, hi, n_all)]
    while stack:
        lo, hi, n = stack.pop()
        if n == 0:
            continue
        if n <= LIMIT:
            out.append((rng(base, lo, hi), n))
            continue
        if lo == hi:
            raise RuntimeError('single application with >%d rows: %s' % (LIMIT, rng(base, lo, hi)))
        mid = (lo + hi) // 2
        nl = count(ds, rng(base, lo, mid))
        stack.append((mid + 1, hi, n - nl))     # right count by subtraction -> one count call per split
        stack.append((lo, mid, nl))
    return out


def fetch_slice(ds, crit, n):
    rows = None
    for _ in range(3):
        resp = call(ds, crit, n)
        rows = resp.get('docs', [])
        if len(rows) == n:
            break
        time.sleep(3)
    return rows


def write_unit(ds, unit, rows, meta):
    tmp = '%s/%s/%s.parquet.tmp' % (RAW, ds, unit)
    tbl = pa.Table.from_pylist([conv(ds, d) for d in rows], schema=SCHEMA[ds])
    pq.write_table(tbl, tmp, compression='zstd', compression_level=6)
    os.replace(tmp, '%s/%s/%s.parquet' % (RAW, ds, unit))
    with open('%s/%s/%s.json.tmp' % (RAW, ds, unit), 'w') as f:
        json.dump(meta, f)
    os.replace('%s/%s/%s.json.tmp' % (RAW, ds, unit), '%s/%s/%s.json' % (RAW, ds, unit))


def do_unit(ds, unit, base, lo=0, hi=99999999):
    if os.path.exists('%s/%s/%s.json' % (RAW, ds, unit)):
        return
    t = time.time()
    total = count(ds, base) if base else count(ds, rng('', lo, hi))
    if total == 0:
        write_unit(ds, unit, [], dict(unit=unit, numFound=0, fetched=0, slices=0, mismatch=0))
        with _lock:
            STATS['units'] += 1
        return
    if total <= LIMIT and base:
        slices = [(base, total)]
    else:
        slices = plan(ds, base, lo, hi, total)
    rows, mism = [], 0
    for crit, n in slices:
        got = fetch_slice(ds, crit, n)
        if len(got) != n:
            mism += abs(n - len(got))
        rows.extend(got)
    if len(rows) != total:
        mism = abs(total - len(rows))
    write_unit(ds, unit, rows, dict(unit=unit, numFound=total, fetched=len(rows), slices=len(slices),
                                    mismatch=mism, secs=round(time.time() - t, 1)))
    with _lock:
        STATS['units'] += 1
        STATS['rows'] += len(rows)
        if mism:
            STATS['errors'] += 1
            print('MISMATCH', ds, unit, total, len(rows), flush=True)


# ---------------------------------------------------------------- unit lists
def days(a, b):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def day_unit(ds, d):
    f = DATEFIELD[ds]
    base = '%s:[%sT00:00:00Z TO %sT23:59:59Z]' % (f, d.isoformat(), d.isoformat())
    return (ds, d.strftime('%Y%m%d'), base)


def unit_list(parts):
    # last office-action date fetched: PAB_OA_CUTOFF=YYYY-MM-DD (the benchmark's cut-off: REPRODUCING.md, level 5); default today
    today = dt.date.fromisoformat(os.environ['PAB_OA_CUTOFF']) if os.environ.get('PAB_OA_CUTOFF') else dt.date.today()
    U = []
    for p in parts:
        if p == 'E':
            U += [day_unit('E', d) for d in days(SCOPE_START, today)]
        elif p == 'R':
            U += [day_unit('R', d) for d in days(SCOPE_START, today)]
        elif p == 'C':
            U += [('C', 'p%03d' % k, None, k * 100000, k * 100000 + 99999) for k in range(1000)]
        elif p == 'P':   # pre-scope rejections, stored apart so they never mix with scope rows
            U += [('R', 'pre' + d.strftime('%Y%m%d'), day_unit('R', d)[2]) for d in days(PRE_START, SCOPE_START - dt.timedelta(days=1))]
    return U


def run(parts):
    for ds in 'ERC':
        os.makedirs('%s/%s' % (RAW, ds), exist_ok=True)
    U = unit_list(parts)
    todo = [u for u in U if not os.path.exists('%s/%s/%s.json' % (RAW, u[0], u[1]))]
    print('units total %d, todo %d, workers %d' % (len(U), len(todo), WORKERS), flush=True)

    def work(u):
        try:
            do_unit(*u)
        except SystemExit:
            raise
        except Exception as e:
            print('UNIT FAILED', u[:2], repr(e)[:300], flush=True)
            with _lock:
                STATS['errors'] += 1

    def beat():
        while True:
            time.sleep(60)
            el = time.time() - STATS['t0']
            msg = dict(elapsed_min=round(el / 60, 1), units=STATS['units'], rows=STATS['rows'], calls=STATS['calls'],
                       retries=STATS['retries'], r429=STATS['r429'], errors=STATS['errors'],
                       rows_per_s=round(STATS['rows'] / max(el, 1)), todo_left=None)
            print('HEARTBEAT', json.dumps(msg), flush=True)
            with open(STATE + '/progress.json', 'w') as f:
                json.dump(msg, f)
    threading.Thread(target=beat, daemon=True).start()
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, todo))
    print('DONE', json.dumps(STATS), flush=True)


def status():
    for ds in 'ERC':
        d = '%s/%s' % (RAW, ds)
        if not os.path.isdir(d):
            continue
        js = [f for f in os.listdir(d) if f.endswith('.json')]
        n = sum(json.load(open(d + '/' + f))['fetched'] for f in js)
        bad = sum(1 for f in js if json.load(open(d + '/' + f))['mismatch'])
        print(ds, 'units', len(js), 'rows', n, 'mismatch-units', bad)


if __name__ == '__main__':
    os.makedirs(STATE, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        status()
    else:
        parts = sys.argv[2:] or ['E', 'R', 'C', 'P']
        run(parts)
