#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Patent Mechanics LLC
"""USPTO office-action label tables, step 2: raw parquet -> v1 tables + stats.json.

  python build/oa_build.py [outdir]        (default $PAB_SRC/uspto_oa/v1 ; use another dir for tests)
  env OA_E_GLOB / OA_R_GLOB / OA_C_GLOB override the raw globs (for partial test builds).

Scope window: mail_date >= 2017-03-01 (the enriched dataset starts 2017-03-13; rejections are fetched from 2017-03-01).
Rejections from 2011-10 .. 2017-02 are read ONLY to compute action_seq / is_first_action and are not written out.
"""
import os, re, sys, json, time
import duckdb
import pyarrow as pa

ROOT = os.environ.get("PAB_SRC", "src") + '/uspto_oa'
OUT = sys.argv[1] if len(sys.argv) > 1 else ROOT + '/v1'
os.makedirs(OUT, exist_ok=True)
EG = os.environ.get('OA_E_GLOB', ROOT + '/raw/E/*.parquet')
RG = os.environ.get('OA_R_GLOB', ROOT + '/raw/R/*.parquet')
CG = os.environ.get('OA_C_GLOB', ROOT + '/raw/C/*.parquet')
SCOPE = '2017-03-01'
T0 = time.time()


def log(*a):
    print('[%6.0fs]' % (time.time() - T0), *a, flush=True)


# ------------------------------------------------------------------ reference normalisation
KIND = re.compile(r'^(?P<num>.*?\d)[\s\-]*(?P<kind>[A-Z]\d?)?$')
VALID = {
    'US': re.compile(r'^US((19|20)\d{8}|\d{4,8}|RE\d{4,6}|D\d{5,7}|PP\d{3,6}|[HTX]\d{4,7})$'),
    'WO': re.compile(r'^WO(\d{7,8}|(19|20)\d{8})$'),
    'EP': re.compile(r'^EP\d{6,7}$'), 'CN': re.compile(r'^CN\d{6,12}$'), 'JP': re.compile(r'^JP[HSR]?\d{6,10}$'),
    'KR': re.compile(r'^KR\d{7,13}$'), 'DE': re.compile(r'^DE\d{5,12}$'), 'GB': re.compile(r'^GB\d{5,7}$'),
    'FR': re.compile(r'^FR\d{6,7}$'), 'CA': re.compile(r'^CA\d{6,7}$'), 'TW': re.compile(r'^TW\d{5,9}$'),
}
GENERIC = re.compile(r'^[A-Z]{2}[A-Z0-9]{4,16}$')


def norm_ref(cited, pubnum, country):
    """-> (docdb or None, country or None, valid, key1, key2). key1/key2 are digit strings used to match oa_citations."""
    src = (cited or '').strip()
    if not src and country and pubnum:
        src = '%s %s' % (country, pubnum)
    key2 = re.sub(r'\D', '', pubnum or '')
    if not src:
        return None, None, False, None, key2 or None
    m = re.match(r'^([A-Za-z]{2})[\s\-]*(.*)$', src)
    if not m:
        return None, None, False, None, key2 or None          # not a patent-style reference (NPL, free text)
    cc, rest = m.group(1).upper(), m.group(2).strip().upper()
    if re.match(r'^%s[\s\-]' % cc, rest):                       # doubled prefix: "US US-2007/0229526 A"
        rest = re.sub(r'^%s[\s\-]+' % cc, '', rest)
    if not re.search(r'\d', rest):                               # "US None" and similar: no number at all
        return None, None, False, None, key2 or None
    k = KIND.match(rest)
    num = (k.group('num') if k else rest).strip()
    key1 = re.sub(r'\D', '', num) or None
    if cc == 'US':
        c = re.sub(r'[\s,\-/]', '', num).upper()
        digits = re.sub(r'\D', '', c)
        if c.isdigit():
            if len(digits) == 11 and digits[:2] in ('19', '20'):
                doc = 'US' + digits[:4] + digits[4:].lstrip('0').zfill(6) if digits[4] == '0' else 'US' + digits
            elif len(digits) == 10 and digits[:2] in ('19', '20'):
                doc = 'US' + digits
            else:
                doc = 'US' + digits.lstrip('0')
        else:
            doc = 'US' + c                                   # RE12345, D612345, PP1234 ...
    elif cc == 'WO':
        sl = re.match(r'^(\d{2}|\d{4})\s*/\s*(\d+)$', num)
        if sl:
            y, ser = sl.group(1), sl.group(2)
        else:
            d = re.sub(r'\D', '', num)
            if len(d) == 10:
                y, ser = d[:4], d[4:]
            else:
                y, ser = None, d
        if y is None:
            doc = 'WO' + ser                                  # already yy+serial (7-8 digits)
        elif len(y) == 4:
            doc = ('WO' + y + ser.zfill(6)) if int(y) >= 2004 else ('WO' + y[2:] + ser.lstrip('0'))
        else:
            doc = 'WO' + y + ser                              # pre-2004 short form: WO0063437
    else:
        doc = cc + re.sub(r'[\s,\-/.]', '', num).upper()
    ok = bool(VALID.get(cc, GENERIC).match(doc))
    return doc, cc, ok, key1, key2 or None


def parse_claims_sql(col):
    """SQL expr: '1-4,9,11' -> [1,2,3,4,9,11] (BIGINT[]), tolerant of junk."""
    return ("list_sort(list_distinct(list_filter(flatten(list_transform(string_split(coalesce(%s,''), ','), "
            "lambda x: CASE WHEN x LIKE '%%-%%' AND try_cast(trim(split_part(x,'-',1)) AS BIGINT) IS NOT NULL "
            "AND try_cast(trim(split_part(x,'-',2)) AS BIGINT) IS NOT NULL "
            "AND try_cast(trim(split_part(x,'-',2)) AS BIGINT) - try_cast(trim(split_part(x,'-',1)) AS BIGINT) BETWEEN 0 AND 2000 "
            "THEN range(try_cast(trim(split_part(x,'-',1)) AS BIGINT), try_cast(trim(split_part(x,'-',2)) AS BIGINT)+1) "
            "ELSE [try_cast(trim(x) AS BIGINT)] END)), lambda v: v IS NOT NULL)))") % col


SECTION = ("CASE WHEN {c} IS NULL OR trim({c})='' THEN NULL WHEN trim({c}) IN ('101','102','103','112') THEN trim({c}) "
           "ELSE 'other' END")

con = duckdb.connect(ROOT + '/state/build.duckdb' if OUT == ROOT + '/v1' else ':memory:')
con.execute("SET memory_limit='20GB'; SET threads=4; SET temp_directory='%s/tmp'; SET preserve_insertion_order=false;" % ROOT)
os.makedirs(ROOT + '/tmp', exist_ok=True)
con.execute("CREATE OR REPLACE VIEW E AS SELECT * FROM read_parquet('%s')" % EG)
con.execute("CREATE OR REPLACE VIEW R AS SELECT * FROM read_parquet('%s')" % RG)
con.execute("CREATE OR REPLACE VIEW C AS SELECT * FROM read_parquet('%s')" % CG)
log('raw rows  E=%d R=%d C=%d' % tuple(con.execute('SELECT count(*) FROM %s' % v).fetchone()[0] for v in 'ERC'))

# ------------------------------------------------------------------ 1. actions (needs R incl. pre-scope, E)
log('building actions')
con.execute("""
CREATE OR REPLACE TABLE act AS
SELECT oa_id,
       regexp_replace(any_value(app), '[^0-9]', '', 'g') AS app_number,
       any_value(doc_code) FILTER (WHERE doc_code IS NOT NULL AND doc_code<>'') AS doc_code,
       min(date) AS mail_date_iso, count(DISTINCT date) AS n_dates,
       max(art_unit) FILTER (WHERE art_unit IS NOT NULL AND art_unit<>'') AS art_unit,
       bool_or(src='E') AS have_enriched, bool_or(src='R') AS have_rejections
FROM (SELECT oa_id, app, date, doc_code, art_unit, 'E' AS src FROM E
      UNION ALL SELECT oa_id, app, date, doc_code, art_unit, 'R' FROM R)
WHERE oa_id IS NOT NULL AND oa_id<>'' AND date<>''
GROUP BY oa_id""")
con.execute("""
CREATE OR REPLACE TABLE act2 AS
SELECT *, row_number() OVER (PARTITION BY app_number ORDER BY mail_date_iso, oa_id) AS action_seq,
       count(*) OVER (PARTITION BY app_number) AS n_actions_known
FROM act""")
log('actions total', con.execute('SELECT count(*) FROM act2').fetchone()[0])

# ------------------------------------------------------------------ 2. reference normalisation (distinct values)
log('normalising references')
con.execute("""CREATE OR REPLACE TABLE ed AS SELECT DISTINCT cited_doc, pubnum, country FROM E
               WHERE oa_id IN (SELECT oa_id FROM act2 WHERE mail_date_iso >= '%s')""" % SCOPE)
nd = con.execute('SELECT count(*) FROM ed').fetchone()[0]
log('distinct (cited_doc,pubnum,country):', nd)
rcur = con.cursor()
rdr = rcur.execute('SELECT cited_doc, pubnum, country FROM ed').to_arrow_reader(1_000_000)
sch = pa.schema([('cited_doc', pa.string()), ('pubnum', pa.string()), ('country', pa.string()), ('docdb', pa.string()),
                 ('cc', pa.string()), ('valid', pa.bool_()), ('key1', pa.string()), ('key2', pa.string())])
con.execute('CREATE OR REPLACE TABLE refmap (cited_doc VARCHAR, pubnum VARCHAR, country VARCHAR, docdb VARCHAR, cc VARCHAR, valid BOOLEAN, key1 VARCHAR, key2 VARCHAR)')
done = 0
for batch in rdr:
    cd, pn, co = (batch.column(i).to_pylist() for i in range(3))
    res = [norm_ref(a, b, c) for a, b, c in zip(cd, pn, co)]
    t = pa.table({'cited_doc': cd, 'pubnum': pn, 'country': co, 'docdb': [r[0] for r in res], 'cc': [r[1] for r in res],
                  'valid': [r[2] for r in res], 'key1': [r[3] for r in res], 'key2': [r[4] for r in res]}, schema=sch)
    con.register('t_batch', t)
    con.execute('INSERT INTO refmap SELECT * FROM t_batch')
    con.unregister('t_batch')
    done += len(cd)
    log('  normalised %d / %d' % (done, nd))

# ------------------------------------------------------------------ 3. write oa_actions
log('writing oa_actions')
con.execute("""
COPY (SELECT oa_id, app_number, doc_code, strftime(CAST(mail_date_iso AS DATE), '%%Y%%m%%d') AS mail_date,
             CAST(action_seq AS INTEGER) AS action_seq, (action_seq = 1) AS is_first_action,
             art_unit,
             -- extras (not in the spec): reliability of the "first action" flag and which datasets saw the action
             (CAST(app_number AS BIGINT) >= 13000000) AS first_action_reliable,
             CAST(n_actions_known AS INTEGER) AS n_actions_known,
             have_enriched, have_rejections,
             (oa_id IN (SELECT DISTINCT oa_id FROM C)) AS have_citations,
             (n_dates > 1) AS date_conflict
      FROM act2 WHERE mail_date_iso >= '%s' ORDER BY app_number, action_seq)
TO '%s/oa_actions.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)""" % (SCOPE, OUT))

# ------------------------------------------------------------------ 4. write oa_rejections
log('writing oa_rejections')
con.execute("""
COPY (SELECT r.oa_id, regexp_replace(r.app, '[^0-9]', '', 'g') AS app_number,
             %s AS legal_section,
             nullif(lower(trim(r.cat_raw)), '') AS action_category,
             %s AS claims,
             nullif(trim(r.legal_raw), '') AS legal_section_raw, r.para AS paragraph_number
      FROM R r WHERE r.date >= '%s')
TO '%s/oa_rejections.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"""
            % (SECTION.format(c='r.legal_raw'), parse_claims_sql('r.claims_raw'), '2017-03-01', OUT))

# ------------------------------------------------------------------ 5. write oa_citations (E rows + legal section from C)
log('building citation sections from oa_citations')
con.execute("""
CREATE OR REPLACE TABLE csec AS
SELECT oa_id, regexp_replace(coalesce(ref_parsed,''), '[^0-9]', '', 'g') AS key,
       list_sort(list_distinct(list(%s))) AS secs
FROM C WHERE oa_id IN (SELECT oa_id FROM act2 WHERE mail_date_iso >= '%s')
  AND legal_raw IS NOT NULL AND trim(legal_raw)<>'' AND coalesce(ref_parsed,'')<>''
GROUP BY 1,2""" % (SECTION.format(c='legal_raw'), SCOPE))
log('csec rows', con.execute('SELECT count(*) FROM csec').fetchone()[0])
log('writing oa_citations')
con.execute("""
COPY (
 SELECT e.oa_id, regexp_replace(e.app, '[^0-9]', '', 'g') AS app_number,
        CASE WHEN list_contains(sc.secs,'102') THEN '102' WHEN list_contains(sc.secs,'103') THEN '103'
             WHEN list_contains(sc.secs,'101') THEN '101' WHEN list_contains(sc.secs,'112') THEN '112'
             WHEN len(sc.secs) > 0 THEN 'other' END AS legal_section,
        nullif(e.cat,'') AS category,
        %s AS related_claims,
        CASE WHEN e.ex_cited AND NOT coalesce(e.app_cited,false) THEN 'examiner'
             WHEN e.app_cited AND NOT coalesce(e.ex_cited,false) THEN 'applicant'
             WHEN e.app_cited AND e.ex_cited THEN 'both' END AS cited_by,
        nullif(e.cited_doc,'') AS ref_raw,
        m.docdb AS ref_number_docdb,
        coalesce(m.cc, nullif(e.country,'')) AS ref_country,
        nullif(e.passage,'') AS passage_location,
        nullif(e.quality,'') AS quality_summary,
        sc.secs AS legal_sections, coalesce(m.valid,false) AS ref_valid, nullif(e.kind,'') AS kind_code
 FROM E e
 LEFT JOIN refmap m ON m.cited_doc IS NOT DISTINCT FROM e.cited_doc AND m.pubnum IS NOT DISTINCT FROM e.pubnum
                   AND m.country IS NOT DISTINCT FROM e.country
 LEFT JOIN csec s1 ON s1.oa_id = e.oa_id AND s1.key = coalesce(nullif(m.key1,''), '#none')
 LEFT JOIN csec s2 ON s2.oa_id = e.oa_id AND s2.key = coalesce(nullif(m.key2,''), '#none')
 CROSS JOIN LATERAL (SELECT coalesce(s1.secs, s2.secs) AS secs) sc
 WHERE e.date >= '%s'
) TO '%s/oa_citations.parquet' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)"""
            % (parse_claims_sql('e.rel_claims'), SCOPE, OUT))
log('done tables')

# ------------------------------------------------------------------ 6. stats.json
log('stats')
q = lambda sql: con.execute(sql).fetchall()
one = lambda sql: con.execute(sql).fetchone()[0]
A = "read_parquet('%s/oa_actions.parquet')" % OUT
Rj = "read_parquet('%s/oa_rejections.parquet')" % OUT
Cj = "read_parquet('%s/oa_citations.parquet')" % OUT
st = {}
st['rows'] = {'oa_actions': one('SELECT count(*) FROM %s' % A), 'oa_rejections': one('SELECT count(*) FROM %s' % Rj),
              'oa_citations': one('SELECT count(*) FROM %s' % Cj)}
yr = lambda tbl, join: dict(q("SELECT substr(a.mail_date,1,4), count(*) FROM %s t JOIN %s a USING(oa_id) GROUP BY 1 ORDER BY 1" % (tbl, A)))
st['rows_per_year'] = {'oa_actions': dict(q("SELECT substr(mail_date,1,4), count(*) FROM %s GROUP BY 1 ORDER BY 1" % A)),
                       'oa_rejections': dict(q("SELECT substr(a.mail_date,1,4), count(*) FROM %s t JOIN %s a USING(oa_id) GROUP BY 1 ORDER BY 1" % (Rj, A))),
                       'oa_citations': dict(q("SELECT substr(a.mail_date,1,4), count(*) FROM %s t JOIN %s a USING(oa_id) GROUP BY 1 ORDER BY 1" % (Cj, A)))}
nR, nC = st['rows']['oa_rejections'], st['rows']['oa_citations']
st['date_range'] = {'oa_actions': q('SELECT min(mail_date), max(mail_date) FROM %s' % A)[0],
                    'enriched_raw': q("SELECT min(date), max(date) FROM E WHERE date<>''")[0],
                    'rejections_raw': q("SELECT min(date), max(date) FROM R WHERE date<>''")[0]}
st['rejections_quality'] = {'blank_legal_section_share': one('SELECT avg((legal_section IS NULL)::int) FROM %s' % Rj),
                            'legal_section_distribution': dict(q('SELECT coalesce(legal_section,\'<blank>\'), count(*) FROM %s GROUP BY 1 ORDER BY 2 DESC' % Rj)),
                            'action_category_distribution': dict(q('SELECT coalesce(action_category,\'<blank>\'), count(*) FROM %s GROUP BY 1 ORDER BY 2 DESC' % Rj)),
                            'empty_claims_share': one('SELECT avg((len(claims)=0)::int) FROM %s' % Rj)}
st['citations_quality'] = {
    'blank_reference_share': one('SELECT avg((ref_raw IS NULL)::int) FROM %s' % Cj),
    'blank_legal_section_share': one('SELECT avg((legal_section IS NULL)::int) FROM %s' % Cj),
    'blank_legal_section_share_among_nonblank_refs': one('SELECT avg((legal_section IS NULL)::int) FROM %s WHERE ref_raw IS NOT NULL' % Cj),
    'category_share': dict(q("SELECT coalesce(category,'<blank>'), count(*)*1.0/%d FROM %s GROUP BY 1 ORDER BY 2 DESC" % (nC, Cj))),
    'cited_by_share': dict(q("SELECT coalesce(cited_by,'<none>'), count(*)*1.0/%d FROM %s GROUP BY 1 ORDER BY 2 DESC" % (nC, Cj))),
    'ref_docdb_valid_share_of_nonblank_refs': one('SELECT avg(ref_valid::int) FROM %s WHERE ref_raw IS NOT NULL' % Cj),
    'ref_docdb_valid_share_of_all_rows': one('SELECT avg(ref_valid::int) FROM %s' % Cj),
    'ref_country_top': dict(q('SELECT coalesce(ref_country,\'<none>\'), count(*) FROM %s GROUP BY 1 ORDER BY 2 DESC LIMIT 15' % Cj)),
    'rows_with_multiple_sections': one('SELECT count(*) FROM %s WHERE len(legal_sections) > 1' % Cj)}
st['actions_quality'] = {
    'first_action_share': one('SELECT avg(is_first_action::int) FROM %s' % A),
    'first_action_reliable_share': one('SELECT avg(first_action_reliable::int) FROM %s' % A),
    'doc_code_distribution': dict(q("SELECT coalesce(doc_code,'<blank>'), count(*) FROM %s GROUP BY 1 ORDER BY 2 DESC LIMIT 20" % A)),
    'have_enriched_share': one('SELECT avg(have_enriched::int) FROM %s' % A),
    'have_rejections_share': one('SELECT avg(have_rejections::int) FROM %s' % A),
    'have_citations_share': one('SELECT avg(have_citations::int) FROM %s' % A),
    'date_conflicts': one('SELECT sum(date_conflict::int) FROM %s' % A)}
json.dump(st, open(OUT + '/stats.json', 'w'), indent=1, default=str)
log('wrote', OUT + '/stats.json')
print(json.dumps(st, indent=1, default=str)[:6000])
