#!/usr/bin/env python3
"""Unified CLI for the sft-extractor pipeline.

Usage:
  python run.py extract <name> <input> <output>   Run a single extractor
  python run.py bleed [--test] [--seed N] [--count N]  Run/test bleed pass
  python run.py ocr   [--test] [--seed N] [--count N]  Run/test OCR pass
  python run.py pair  [--test] [--seed N] [--count N]  Run/test pair pass
  python run.py sample <stage> [--seed N] [--count N]  Sample from any stage

Stages: extracted | bleed | ocr | paired

Set environment variables for model passes:
    export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
    export ANTHROPIC_API_KEY=<your deepseek key>
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

from authentic.extractors.base import (
    AdvancedQuestionsExtractor,
    AgricultureExtractor,
    AstronomyExtractor,
    BotanyExtractor,
    BrewersGuideExtractor,
    ChemistryExtractor,
    CivilWarExtractor,
    CommonCoreExtractor,
    ConstitutionExtractor,
    ElectricityExtractor,
    EngineeringExtractor,
    EthicsExtractor,
    GrammarExtractor,
    FamiliarThingsExtractor,
    InvestorsExtractor,
    LaborersExtractor,
    LogicExtractor,
    MusicExtractor,
    MythologyExtractor,
    NewYorkBarExtractor,
    PatriotismExtractor,
    Questions1001Extractor,
    SchoolBulletinExtractor,
    SeeleysExtractor,
    StokersExtractor,
    SymbologicalExtractor,
    WorldHistoryExtractor,
)

EXTRACTORS = {
    "advanced_questions": AdvancedQuestionsExtractor,
    "common_core": CommonCoreExtractor,
    "brewers_guide": BrewersGuideExtractor,
    "familiar_things": FamiliarThingsExtractor,
    "1001_questions": Questions1001Extractor,
    "logic": LogicExtractor,
    "seeleys": SeeleysExtractor,
    "stokers": StokersExtractor,
    "symbological": SymbologicalExtractor,
    "agriculture": AgricultureExtractor,
    "astronomy": AstronomyExtractor,
    "botany": BotanyExtractor,
    "chemistry": ChemistryExtractor,
    "civil_war": CivilWarExtractor,
    "constitution": ConstitutionExtractor,
    "electricity": ElectricityExtractor,
    "engineering": EngineeringExtractor,
    "ethics": EthicsExtractor,
    "grammar": GrammarExtractor,
    "mythology": MythologyExtractor,
    "new_york_bar": NewYorkBarExtractor,
    "patriotism": PatriotismExtractor,
    "school_bulletin": SchoolBulletinExtractor,
    "investors": InvestorsExtractor,
    "music": MusicExtractor,
    "laborers": LaborersExtractor,
    "world_history": WorldHistoryExtractor,
}

OUTPUT_DIRS = {
    "extracted": ROOT / "authentic" / "output" / "extracted",
    "bleed":     ROOT / "authentic" / "output" / "bleed",
    "ocr":       ROOT / "authentic" / "output" / "ocr",
    "paired":    ROOT / "authentic" / "output" / "paired",
    "enriched":  ROOT / "authentic" / "output" / "enriched",
    "filtered":  ROOT / "authentic" / "output" / "filtered",
    "scored":    ROOT / "authentic" / "output" / "scored",
}


def cmd_extract(args):
    if args.name not in EXTRACTORS:
        print(f"Unknown extractor: {args.name}. Available: {', '.join(EXTRACTORS)}")
        sys.exit(1)
    EXTRACTORS[args.name](args.input, args.output).run()


def cmd_bleed(args):
    from authentic.clean import bleed
    if args.retry_bleed:
        pairs = bleed.load_ocr_pairs()
        state = bleed.load_retry_state()
        asyncio.run(bleed.retry_bleed(pairs, state))
        print("Writing output...")
        bleed.write_retry_output(state)
        print("Done. Re-run `pair` to rebuild multi-turn conversations.")
        return
    pairs = bleed.load_all_pairs()
    if args.test:
        asyncio.run(bleed.test_run(pairs, args.seed, args.size))
        return
    state = bleed.load_state()
    resolved = sum(1 for d, i, *_ in pairs if f"{d}--{i}" in state)
    pending = len(pairs) - resolved
    print(f"Total: {len(pairs)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(bleed.run_async(pairs, state))
        bleed.save_state(state)
    print("Writing output...")
    bleed.write_output(pairs, state)
    print("Done.")


def cmd_ocr(args):
    from authentic.clean import ocr
    pairs = ocr.load_all_pairs()
    if args.test and args.retry_flagged:
        state = ocr.load_state()
        asyncio.run(ocr.test_retry_flagged(pairs, state, args.seed, args.size))
        return
    if args.test:
        asyncio.run(ocr.test_run(pairs, args.seed, args.size))
        return
    state = ocr.load_state()

    if args.retry_flagged:
        asyncio.run(ocr.retry_flagged(pairs, state))
        print("Writing output...")
        ocr.write_output(pairs, state)
        print("Done.")
        return

    resolved = sum(1 for d, i, *_ in pairs if f"{d}--{i}" in state)
    pending = len(pairs) - resolved
    print(f"Total: {len(pairs)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(ocr.run_async(pairs, state))
        ocr.save_state(state)
    print("Writing output...")
    ocr.write_output(pairs, state)
    print("Done.")


def cmd_filter(args):
    from authentic.clean import filter as f
    if args.sample:
        f.sample_dropped(n=args.size, seed=args.seed)
        return
    f.run()


def cmd_score(args):
    from authentic.clean import score
    datasets = score.load_all_conversations()
    if args.test:
        asyncio.run(score.test_run(datasets, args.seed, args.size))
        return
    state = score.load_state()
    resolved = sum(1 for d, items in datasets.items() for i in range(len(items))
                   if f"{d}--{i}" in state)
    total = sum(len(items) for items in datasets.values())
    pending = total - resolved
    print(f"Total: {total}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(score.run_async(datasets, state))
        score.save_state(state)
    print("Writing output...")
    score.write_output(datasets, state, filter_pct=args.filter_pct)
    print("Done.")


def cmd_enrich(args):
    from authentic.clean import enrich
    datasets = enrich.load_all_conversations()
    if args.test:
        asyncio.run(enrich.test_run(datasets, args.seed, args.size))
        return
    state = enrich.load_state()
    asyncio.run(enrich.run_async(datasets, state))
    print("Writing output...")
    enrich.write_output(datasets, state)
    print("Done.")


def cmd_pair(args):
    from authentic.clean import pair
    datasets = pair.load_all_pairs()
    if args.test:
        asyncio.run(pair.test_run(datasets, args.seed, args.size))
        return
    state = pair.load_state()
    all_keys = {f"{d}--w{w}" for d, pairs in datasets.items()
                for w in range(len(pair.make_windows(pairs)))}
    resolved = sum(1 for k in all_keys if k in state)
    pending = len(all_keys) - resolved
    print(f"Total windows: {len(all_keys)}  Resolved: {resolved}  Pending: {pending}")
    if pending:
        asyncio.run(pair.run_async(datasets, state))
        pair.save_state(state)
    print("Writing output...")
    pair.write_output(datasets, state)
    print("Done.")





def cmd_synthq(args):
    from synth import questions
    pairs = questions.load_authentic_pairs()
    styles = questions.STYLES if args.style == "all" else (args.style,)
    if args.size and not args.test:
        pairs = questions.sample_pairs(pairs, args.size, args.seed)
    for style in styles:
        if args.test:
            asyncio.run(questions.test_run(pairs, style, args.seed, args.size))
            continue
        state = questions.load_state(style)
        print(f"Total: {len(pairs)}  Resolved: {sum(1 for d, i, *_ in pairs if f'{d}--{i}' in state)}")
        asyncio.run(questions.run_async(pairs, style, state))
        questions.write_output(pairs, style, state)
    print("Done.")


def cmd_harvest(args):
    from synth import corpus
    if args.relabel:
        corpus.relabel_excerpts()
        return
    if args.stem:
        target = float("inf") if args.exhaust else args.total
        corpus.harvest_stem(target, min_conf=args.min_conf, seed=args.seed,
                            max_shards=args.max_shards)
        return
    if args.verse:
        target = float("inf") if args.exhaust else args.total
        corpus.harvest_verse(target, seed=args.seed, max_shards=args.max_shards)
        return
    if args.calibration:
        target = float("inf") if args.exhaust else args.total
        corpus.harvest_calibration(target, seed=args.seed, min_conf=args.min_conf,
                                    max_shards=args.max_shards)
        return
    corpus.harvest(args.total, alpha=args.alpha, min_conf=args.min_conf,
                   n_words=args.words, per_doc=args.per_doc, seed=args.seed,
                   max_shards=args.max_shards)


def cmd_stem(args):
    from synth.routes import stem_reasoning as sr
    excerpts = sr.source_excerpts(args.size, seed=args.seed)
    print(f"Sourced {len(excerpts)} STEM-reasoning excerpts")
    if args.test:
        asyncio.run(sr.test_run(excerpts))
        return
    if args.sample:
        asyncio.run(sr.sample_run(excerpts, seed=args.seed))
        return
    state = sr.load_state()
    asyncio.run(sr.run_async(excerpts, state))
    sr.write_output(excerpts, state)
    print("Done.")


def cmd_reasoning(args):
    from synth.routes import reasoning_qa as rq
    excerpts = rq.source_excerpts(args.size, seed=args.seed)
    print(f"Sourced {len(excerpts)} reasoning excerpts (classes={rq.CLASSES})")
    if args.test:
        asyncio.run(rq.test_run(excerpts))
        return
    if args.sample:
        asyncio.run(rq.sample_run(excerpts, seed=args.seed))
        return
    state = rq.load_state()
    resolved = sum(1 for e in excerpts if str(e["doc_index"]) in state)
    print(f"Excerpts: {len(excerpts)}  Resolved: {resolved}")
    asyncio.run(rq.run_async(excerpts, state))
    rq.write_output(excerpts, state)
    print("Done.")


def cmd_filter_pool(args):
    from synth import corpus
    recs = corpus.load_excerpts()
    scored = [(corpus.ocr_score(r["excerpt"]), r) for r in recs]
    over = [(s, r) for s, r in scored if s > args.threshold]
    ss = sorted(s for s, _ in scored)
    def pct(q):
        return ss[min(int(len(ss) * q), len(ss) - 1)] if ss else 0.0
    print(f"pool: {len(recs)}   over threshold {args.threshold}: {len(over)} "
          f"({len(over) * 100 // max(len(recs), 1)}%)")
    print(f"  ocr_score  p50={pct(.5):.3f}  p90={pct(.9):.3f}  p99={pct(.99):.3f}  "
          f"max={ss[-1] if ss else 0:.3f}")
    print("  worst offenders:")
    for s, r in sorted(over, key=lambda x: -x[0])[:8]:
        print(f"   [{s:.3f}] {r['excerpt'][:96].strip()}")
    if args.apply:
        keep = [r for s, r in scored if s <= args.threshold]
        with corpus.EXCERPTS_FILE.open("w", encoding="utf-8") as fh:
            for r in keep:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  REMOVED {len(over)}; pool now {len(keep)} -> {corpus.EXCERPTS_FILE}")
    else:
        print("  (dry-run — pass --apply to remove; --threshold to tune)")


def cmd_classify(args):
    from synth import classify, corpus
    if args.concurrency:
        classify.CONCURRENCY = args.concurrency
    if args.model:
        classify.MODEL = args.model
    records = corpus.load_excerpts()
    state = classify.load_state()
    if args.coverage:
        classify.coverage(records)
        return
    if args.report:
        classify.report(records, state)
        return
    if args.sample:
        asyncio.run(classify.sample_run(records, seed=args.seed, n=args.size))
        return
    if args.test:
        sample = random.Random(args.seed).sample(records, min(args.size, len(records)))
        tmp = {}
        asyncio.run(classify.run_async(sample, tmp, save=False))
        for r in sample:
            cl = tmp.get(str(r["doc_index"]))
            print(f"[aff={r.get('affordance'):11}] -> {cl}  :: {r['excerpt'][:64].strip()}")
        return
    pool = records
    if args.limit:
        pool = random.Random(args.seed).sample(records, min(args.limit, len(records)))
    asyncio.run(classify.run_async(pool, state))
    classify.write_back(records, state)
    classify.report(records, state)
    remaining = sum(1 for r in records if str(r["doc_index"]) not in state)
    if remaining:
        print(f"\nINCOMPLETE — {remaining:,} of {len(records):,} still unclassified. "
              f"Re-run to finish (usage-cap limited).")
    else:
        print(f"\nCOMPLETE — all {len(records):,} excerpts classified.")


def cmd_composition(args):
    from synth.routes import composition_qa as cq
    excerpts = cq.source_excerpts(args.size, seed=args.seed)
    print(f"Sourced {len(excerpts)} composition excerpts (classes={cq.CLASSES})")
    if args.test:
        asyncio.run(cq.test_run(excerpts))
        return
    if args.sample:
        asyncio.run(cq.sample_run(excerpts, seed=args.seed))
        return
    state = cq.load_state()
    resolved = sum(1 for e in excerpts if str(e["doc_index"]) in state)
    print(f"Excerpts: {len(excerpts)}  Resolved: {resolved}")
    asyncio.run(cq.run_async(excerpts, state))
    cq.write_output(excerpts, state)
    print("Done.")


def cmd_narrative(args):
    from synth.routes import narrative_qa as nq
    from synth import engine
    # Two baked-in prompts: grounded (fact) and fiction (story). Drive both passes.
    for route in nq.ROUTES:
        excerpts = route.source(args.size, seed=args.seed)
        print(f"Sourced {len(excerpts)} {route.name} excerpts")
        if args.test:
            asyncio.run(engine.test_run(route, excerpts))
            continue
        if args.sample:
            asyncio.run(engine.sample_run(route, excerpts, seed=args.seed))
            continue
        state = engine.load_state(route)
        resolved = sum(1 for e in excerpts if str(e["doc_index"]) in state)
        print(f"  resolved {resolved}/{len(excerpts)}")
        asyncio.run(engine.run_async(route, excerpts, state))
        engine.write_output(route, excerpts, state)
    print("Done.")


def _run_extractive_route(args, module, label):
    """Shared driver for the simple one/two-phase routes (source, test/sample, run)."""
    excerpts = module.source_excerpts(args.size, seed=args.seed)
    print(f"Sourced {len(excerpts)} {label} excerpts (classes={module.CLASSES})")
    if args.test:
        asyncio.run(module.test_run(excerpts))
        return
    if args.sample:
        asyncio.run(module.sample_run(excerpts, seed=args.seed))
        return
    state = module.load_state()
    resolved = sum(1 for e in excerpts if str(e["doc_index"]) in state)
    print(f"Excerpts: {len(excerpts)}  Resolved: {resolved}")
    asyncio.run(module.run_async(excerpts, state))
    module.write_output(excerpts, state)
    print("Done.")


def cmd_howto(args):
    from synth.routes import how_to_qa
    _run_extractive_route(args, how_to_qa, "how-to")


def cmd_opinion(args):
    from synth.routes import opinion_qa
    _run_extractive_route(args, opinion_qa, "opinion")


def cmd_verse(args):
    from synth.routes import verse_qa
    _run_extractive_route(args, verse_qa, "verse")


def cmd_multiturn(args):
    from synth.routes import multiturn_qa
    _run_extractive_route(args, multiturn_qa, "multiturn")


def cmd_calibration(args):
    from synth.routes import calibration_qa
    _run_extractive_route(args, calibration_qa, "calibration")


def cmd_knowledge(args):
    from synth.routes import knowledge_qa as kq
    excerpts = kq.source_excerpts(args.size, seed=args.seed)
    print(f"Sourced {len(excerpts)} knowledge excerpts (classes={kq.CLASSES})")
    if args.test:
        asyncio.run(kq.test_run(excerpts))
        return
    if args.sample:
        asyncio.run(kq.sample_run(excerpts, seed=args.seed))
        return
    state = kq.load_state()
    resolved = sum(1 for e in excerpts if str(e["doc_index"]) in state)
    print(f"Excerpts: {len(excerpts)}  Resolved: {resolved}")
    asyncio.run(kq.run_async(excerpts, state))
    kq.write_output(excerpts, state)
    print("Done.")


def cmd_sample_corpus(args):
    from synth import corpus
    ewords = [int(x) for x in args.excerpt_words.split(",")]
    if args.list_categories:
        corpus.list_categories()
    elif args.coverage:
        corpus.report_coverage(args.total, alpha=args.alpha, min_conf=args.min_conf)
    elif args.excerpts:
        corpus.report_excerpts(n=args.size, alpha=args.alpha, min_conf=args.min_conf,
                               n_words=ewords[0], seed=args.seed)
    elif args.category:
        corpus.report_by_category(args.category, n_text=args.show, seed=args.seed,
                                  excerpt_words=ewords)
    else:
        corpus.report(n=args.size, seed=args.seed, excerpt_words=ewords, show=args.show)


def cmd_sample(args):
    directory = OUTPUT_DIRS[args.stage]
    if not directory.exists():
        print(f"Directory not found: {directory}. Has this stage been run yet?")
        sys.exit(1)

    convs_list = []
    for json_file in sorted(directory.glob("*.json")):
        dataset = json_file.stem
        items = json.loads(json_file.read_text())
        for i, item in enumerate(items):
            convs_list.append((dataset, i, item["conversations"], item.get("chained", False)))

    random.seed(args.seed)
    sample = random.sample(convs_list, min(args.size, len(convs_list)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_dir = ROOT / "authentic" / "samples" / args.stage
    samples_dir.mkdir(parents=True, exist_ok=True)
    out_path = samples_dir / f"{ts}_seed{args.seed}_n{args.size}.txt"

    lines = [f"stage: {args.stage}", f"seed:  {args.seed}", f"n:     {args.size}", ""]
    for dataset, i, convs, chained in sample:
        label = f"[{dataset}--{i}]" + (" [multi-turn]" if chained else "")
        lines.append(label)
        turns = [(c["role"], c["content"]) for c in convs]
        for role, content in turns:
            prefix = "  Q:" if role == "user" else "  A:"
            lines.append(f"{prefix} {content}")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="sft-extractor pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract
    p_ext = sub.add_parser("extract", help="Run an extractor")
    p_ext.add_argument("name", choices=sorted(EXTRACTORS), metavar="name",
                       help=f"Extractor name. Choices: {', '.join(sorted(EXTRACTORS))}")
    p_ext.add_argument("input", help="Path to input .txt file")
    p_ext.add_argument("output", help="Path to output .json file")

    # bleed
    p_bleed = sub.add_parser("bleed", help="Run bleed detection/recovery pass (pass 0)")
    p_bleed.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_bleed.add_argument("--seed", type=int, default=42)
    p_bleed.add_argument("--count", type=int, default=10, dest="size", metavar="N")
    p_bleed.add_argument("--retry-bleed", action="store_true",
                         help="Re-run bleed detection on output/ocr/ and update in place")

    # ocr
    p_ocr = sub.add_parser("ocr", help="Run OCR correction pass (pass 1)")
    p_ocr.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_ocr.add_argument("--seed", type=int, default=42)
    p_ocr.add_argument("--count", type=int, default=10, dest="size", metavar="N")
    p_ocr.add_argument("--retry-flagged", action="store_true",
                        help="Re-process only currently flagged pairs using greedy bin-packing batches")

    # pair
    p_pair = sub.add_parser("pair", help="Run dependency-chain grouping pass (pass 2)")
    p_pair.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_pair.add_argument("--seed", type=int, default=42)
    p_pair.add_argument("--count", type=int, default=10, dest="size", metavar="N")

    # enrich
    p_enrich = sub.add_parser("enrich", help="Detect and enrich context-bare opening questions (pass 3)")
    p_enrich.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_enrich.add_argument("--seed", type=int, default=42)
    p_enrich.add_argument("--count", type=int, default=20, dest="size", metavar="N")

    # filter
    p_filter = sub.add_parser("filter", help="Rule-based garbage filtering (pass 4)")
    p_filter.add_argument("--sample", action="store_true", help="Preview what would be dropped without writing output")
    p_filter.add_argument("--seed", type=int, default=42)
    p_filter.add_argument("--count", type=int, default=20, dest="size", metavar="N")

    # score
    p_score = sub.add_parser("score", help="Score conversations and filter bottom N%% (pass 4)")
    p_score.add_argument("--test", action="store_true", help="Sample and test without full run")
    p_score.add_argument("--seed", type=int, default=42)
    p_score.add_argument("--count", type=int, default=20, dest="size", metavar="N")
    p_score.add_argument("--filter-pct", type=float, default=0.05, metavar="F",
                         help="Fraction to drop (default 0.05 = bottom 5%%)")

    # synth-questions
    p_synthq = sub.add_parser("synth-questions",
                              help="Generate synthetic questions from authentic answers (matched pairs)")
    # Keep in sync with synth.questions.STYLES (not imported here — that would pull
    # in anthropic on every CLI invocation, not just synth-questions).
    p_synthq.add_argument("--style", choices=["naive", "period", "cold", "examiner", "working",
                                              "scoped", "noblind", "pronoun", "paired", "combo", "all"],
                          default="working",
                          help="Prompt style to generate (default: working, the living prompt; "
                               "the named styles are the frozen experiment)")
    p_synthq.add_argument("--test", action="store_true", help="Sample and print pairs without writing output")
    p_synthq.add_argument("--seed", type=int, default=42)
    p_synthq.add_argument("--count", type=int, default=None, dest="size", metavar="N",
                          help="Limit to N answers (default: all)")

    # harvest
    p_hv = sub.add_parser("harvest",
                          help="Shard-major sweep: materialize gated, tagged excerpts to synth/output/excerpts.jsonl")
    p_hv.add_argument("--total", type=int, default=2000, help="excerpt budget across the coverage plan")
    p_hv.add_argument("--alpha", type=float, default=0.5, help="tempering: 1=corpus, 0=uniform")
    p_hv.add_argument("--min-conf", type=float, default=0.7, help="label-confidence floor")
    p_hv.add_argument("--seed", type=int, default=0)
    p_hv.add_argument("--max-shards", type=int, default=473, help="cap shards swept per run (resumable)")
    p_hv.add_argument("--words", type=int, default=None,
                      help="fixed excerpt length; default (unset) varies length per excerpt "
                           "(40%% short, 35%% medium, 25%% long) for answer-length diversity")
    p_hv.add_argument("--per-doc", type=int, default=3, dest="per_doc",
                      help="prose windows to cut per document (default 3); raises the pool "
                           "ceiling from the eligible-doc count to docs x per-doc")
    p_hv.add_argument("--relabel", action="store_true",
                      help="re-tag affordances of the existing excerpts in place (no fetch)")
    p_hv.add_argument("--stem", action="store_true",
                      help="targeted STEM overlay: sweep quantitative categories into "
                           "excerpts.jsonl via multi-window (uses --total as the STEM target)")
    p_hv.add_argument("--verse", action="store_true",
                      help="targeted verse overlay: line-based windows from poetry/hymn books")
    p_hv.add_argument("--calibration", action="store_true",
                      help="targeted calibration overlay: sentence windows around factual non-answers")
    p_hv.add_argument("--exhaust", action="store_true",
                      help="with --stem/--verse: ignore the count target "
                           "and sweep ALL remaining shards, filling the bucket to the brim")

    # stem-reasoning
    p_sr = sub.add_parser("stem-reasoning",
                          help="Generate step-showing STEM Q/A from SCIENCE/TECHNOLOGY reasoning windows")
    p_sr.add_argument("--count", type=int, default=20, dest="size", metavar="N",
                      help="target STEM excerpts to source")
    p_sr.add_argument("--seed", type=int, default=0)
    p_sr.add_argument("--test", action="store_true", help="generate a few and print without writing")
    p_sr.add_argument("--sample", action="store_true", help="generate a few, print, AND save a dated record (prompt + Q/A) to samples/")

    # reasoning-qa
    p_rq = sub.add_parser("reasoning-qa",
                          help="Generate step-showing Q/A from argument excerpts (reasoning route)")
    p_rq.add_argument("--seed", type=int, default=0)
    p_rq.add_argument("--count", type=int, default=100, dest="size", metavar="N",
                      help="max argument excerpts to process")
    p_rq.add_argument("--test", action="store_true", help="generate a few and print without writing")
    p_rq.add_argument("--sample", action="store_true", help="generate a few, print, AND save a dated record (prompt + Q/A) to samples/")

    # filter-pool
    p_fp = sub.add_parser("filter-pool",
                          help="Flag/remove the worst OCR-garbled excerpts (mangled equations) from the pool")
    p_fp.add_argument("--threshold", type=float, default=0.12,
                      help="ocr_score above which an excerpt is removed (default 0.12)")
    p_fp.add_argument("--apply", action="store_true",
                      help="actually rewrite excerpts.jsonl without the offenders (default: dry-run)")

    # classify
    p_cls = sub.add_parser("classify",
                           help="Model-classify harvested excerpts into route classes (writes classes/primary)")
    p_cls.add_argument("--limit", type=int, default=None,
                       help="classify only N excerpts (sampled) — a cheap trial run")
    p_cls.add_argument("--count", type=int, default=12, dest="size", metavar="N",
                       help="excerpts to show for --test / --sample")
    p_cls.add_argument("--seed", type=int, default=0)
    p_cls.add_argument("--test", action="store_true",
                       help="classify a small sample and print model labels vs regex (truncated), no write")
    p_cls.add_argument("--sample", action="store_true",
                       help="classify a sample and save a dated record with FULL excerpts + labels to samples/")
    p_cls.add_argument("--report", action="store_true",
                       help="print the distribution + regex-agreement report from existing labels")
    p_cls.add_argument("--coverage", action="store_true",
                       help="print per-class confirmed supply vs. row target (the harvest dashboard)")
    p_cls.add_argument("--concurrency", type=int, default=None,
                       help="in-flight requests (default 40; raise until the API rate-limits)")
    p_cls.add_argument("--model", type=str, default=None,
                       help="override the classifier model (use a non-reasoning model for speed)")

    # composition-qa
    p_cq = sub.add_parser("composition-qa",
                          help="Generate compose-this-document Q/A from formal documents (generative route)")
    p_cq.add_argument("--seed", type=int, default=0)
    p_cq.add_argument("--count", type=int, default=100, dest="size", metavar="N",
                      help="max composition excerpts to process")
    p_cq.add_argument("--test", action="store_true", help="generate a few and print without writing")
    p_cq.add_argument("--sample", action="store_true", help="generate a few and save a dated record (prompt + Q/A) to samples/")

    # how-to-qa / opinion-qa / verse-qa (same simple arg set)
    for _name, _help in [
        ("how-to-qa", "Generate method Q/A from how_to excerpts"),
        ("opinion-qa", "Generate view-and-ground Q/A from opinion excerpts"),
        ("verse-qa", "Generate compose-a-verse Q/A from verse excerpts"),
        ("multiturn-qa", "Generate interrelated multi-turn Q&A from knowledge excerpts (hybrid route)"),
        ("calibration-qa", "Generate calibrated-uncertainty Q/A from hedging passages"),
    ]:
        _p = sub.add_parser(_name, help=_help)
        _p.add_argument("--seed", type=int, default=0)
        _p.add_argument("--count", type=int, default=100, dest="size", metavar="N",
                        help="max excerpts to process")
        _p.add_argument("--test", action="store_true", help="generate a few and print without writing")
        _p.add_argument("--sample", action="store_true", help="generate a few and save a dated record to samples/")

    # narrative-qa
    p_nq = sub.add_parser("narrative-qa",
                          help="Generate comprehension/retell Q/A from narrative excerpts (narrative route)")
    p_nq.add_argument("--seed", type=int, default=0)
    p_nq.add_argument("--count", type=int, default=100, dest="size", metavar="N",
                      help="max narrative excerpts to process")
    p_nq.add_argument("--test", action="store_true", help="generate a few and print without writing")
    p_nq.add_argument("--sample", action="store_true", help="generate a few and save a dated record (prompt + Q/A) to samples/")

    # knowledge-qa
    p_kq = sub.add_parser("knowledge-qa",
                          help="Generate grounded Q/A from expository excerpts (knowledge-QA route)")
    p_kq.add_argument("--count", type=int, default=100, dest="size", metavar="N",
                      help="excerpts to sample before keeping the expository ones")
    p_kq.add_argument("--test", action="store_true", help="sample a few and print Q/A without writing")
    p_kq.add_argument("--sample", action="store_true", help="generate a few, print, AND save a dated record (prompt + Q/A) to samples/")
    p_kq.add_argument("--alpha", type=float, default=0.5, help="tempering: 1=corpus, 0=uniform")
    p_kq.add_argument("--min-conf", type=float, default=0.7, help="label-confidence floor")
    p_kq.add_argument("--seed", type=int, default=0)

    # sample-corpus
    p_sc = sub.add_parser("sample-corpus",
                          help="Sample the pretrain corpus and preview excerpt windows (needs duckdb)")
    p_sc.add_argument("--count", type=int, default=10, dest="size", metavar="N",
                      help="documents to sample")
    p_sc.add_argument("--category", type=str, default=None,
                      help="sample within one LoC category (see --list-categories)")
    p_sc.add_argument("--list-categories", action="store_true",
                      help="print the taxonomy with per-category document counts")
    p_sc.add_argument("--coverage", action="store_true",
                      help="print the coverage plan (per-category excerpt quotas)")
    p_sc.add_argument("--excerpts", action="store_true",
                      help="sample plan-weighted excerpts across categories and print them")
    p_sc.add_argument("--total", type=int, default=2000, help="excerpt budget for the coverage plan")
    p_sc.add_argument("--alpha", type=float, default=0.5, help="tempering: 1=corpus, 0=uniform")
    p_sc.add_argument("--min-conf", type=float, default=0.5, help="label-confidence floor")
    p_sc.add_argument("--seed", type=int, default=0)
    p_sc.add_argument("--excerpt-words", type=str, default="60,120,240",
                      help="comma-separated excerpt lengths to preview")
    p_sc.add_argument("--show", type=int, default=3,
                      help="documents to show excerpt windows for")

    # sample
    p_sample = sub.add_parser("sample", help="Sample conversations from any pipeline stage")
    p_sample.add_argument("stage", choices=list(OUTPUT_DIRS))
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--count", type=int, default=10, dest="size", metavar="N")

    args = parser.parse_args()

    if args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "bleed":
        cmd_bleed(args)
    elif args.cmd == "ocr":
        cmd_ocr(args)
    elif args.cmd == "pair":
        cmd_pair(args)
    elif args.cmd == "enrich":
        cmd_enrich(args)
    elif args.cmd == "filter":
        cmd_filter(args)
    elif args.cmd == "score":
        cmd_score(args)
    elif args.cmd == "synth-questions":
        cmd_synthq(args)
    elif args.cmd == "harvest":
        cmd_harvest(args)
    elif args.cmd == "stem-reasoning":
        cmd_stem(args)
    elif args.cmd == "reasoning-qa":
        cmd_reasoning(args)
    elif args.cmd == "narrative-qa":
        cmd_narrative(args)
    elif args.cmd == "composition-qa":
        cmd_composition(args)
    elif args.cmd == "how-to-qa":
        cmd_howto(args)
    elif args.cmd == "opinion-qa":
        cmd_opinion(args)
    elif args.cmd == "verse-qa":
        cmd_verse(args)
    elif args.cmd == "multiturn-qa":
        cmd_multiturn(args)
    elif args.cmd == "calibration-qa":
        cmd_calibration(args)
    elif args.cmd == "filter-pool":
        cmd_filter_pool(args)
    elif args.cmd == "classify":
        cmd_classify(args)
    elif args.cmd == "knowledge-qa":
        cmd_knowledge(args)
    elif args.cmd == "sample-corpus":
        cmd_sample_corpus(args)
    elif args.cmd == "sample":
        cmd_sample(args)


if __name__ == "__main__":
    main()
