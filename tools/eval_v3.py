import argparse
import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from db.connection import close_pool
from services.rag_v3 import run_rag_pipeline_stream_v3
from tools.retriever_v3 import search_rag_v3

load_dotenv()


@dataclass(frozen=True)
class EvalCase:
    name: str
    query: str
    domain: str | None
    expected_domain: str | None = None
    expected_text: tuple[str, ...] = ()


EVAL_CASES = [
    EvalCase(
        name="item_craft_salewa",
        query="살레와 어디서 만들어?",
        domain="item",
        expected_domain="item",
        expected_text=("Salewa", "의료 시설"),
    ),
    EvalCase(
        name="item_obtain_ledx",
        query="레덱스 어디서 얻어?",
        domain="item",
        expected_domain="item",
        expected_text=("LEDX",),
    ),
    EvalCase(
        name="quest_shortage",
        query="재고 부족에 뭐 필요해?",
        domain="quest",
        expected_domain="quest",
        expected_text=("재고 부족", "Salewa"),
    ),
    EvalCase(
        name="boss_killa_spawn",
        query="킬라 어디 나와?",
        domain="boss",
        expected_domain="boss",
        expected_text=("킬라", "Killa"),
    ),
    EvalCase(
        name="hideout_medstation_level2",
        query="의료 시설 2렙 필요 재료",
        domain="hideout",
        expected_domain="hideout",
        expected_text=("의료 시설", "Lv.2"),
    ),
    EvalCase(
        name="trader_therapist_salewa_barter",
        query="테라피스트 살레와 바터",
        domain="trader",
        expected_domain="trader",
        expected_text=("테라피스트", "Salewa"),
    ),
    EvalCase(
        name="story_terminal_access",
        query="터미널 접근 스토리",
        domain="story",
        expected_domain="story",
        expected_text=("터미널",),
    ),
    EvalCase(
        name="information_smugglers_2026",
        query="Smugglers 2026 어디 나와?",
        domain="information",
        expected_domain="information",
        expected_text=("Smugglers 2026",),
    ),
    EvalCase(
        name="quest_debut_exact",
        query="퀘스트: 데뷔",
        domain="quest",
        expected_domain="quest",
        expected_text=("데뷔", "MP-133"),
    ),
    EvalCase(
        name="quest_collector_exact",
        query="퀘스트: 수집가",
        domain="quest",
        expected_domain="quest",
        expected_text=("수집가", "보안 컨테이너 카파"),
    ),
    EvalCase(
        name="quest_gunsmith_24_exact",
        query="퀘스트: Gunsmith - Part 24",
        domain="quest",
        expected_domain="quest",
        expected_text=("Gunsmith - Part 24", "Gunsmith - Part 23"),
    ),
    EvalCase(
        name="boss_tagilla_info",
        query="보스: 타길라 정보 알려줘",
        domain="boss",
        expected_domain="boss",
        expected_text=("타길라", "스폰 위치"),
    ),
    EvalCase(
        name="boss_cultist_priest_info",
        query="보스: 광신도 사제",
        domain="boss",
        expected_domain="boss",
        expected_text=("광신도 사제", "스폰 위치"),
    ),
    EvalCase(
        name="map_customs_extracts_ko",
        query="세관 탈출구 한글로 번역해줄수있어?",
        domain="map",
        expected_domain="map",
        expected_text=("세관", "탈출구"),
    ),
    EvalCase(
        name="map_factory_short",
        query="팩토리",
        domain="map",
        expected_domain="map",
        expected_text=("공장", "탈출구"),
    ),
    EvalCase(
        name="item_rusted_bloody_key_exact",
        query="아이템: Rusted bloody key",
        domain="item",
        expected_domain="item",
        expected_text=("Rusted bloody key", "The Door"),
    ),
    EvalCase(
        name="item_m80_exact",
        query="아이템: 7.62x51mm M80",
        domain="item",
        expected_domain="item",
        expected_text=("7.62x51mm M80",),
    ),
    EvalCase(
        name="item_gas_analyzer_korean",
        query="가스분석기",
        domain="item",
        expected_domain="item",
        expected_text=("Gas analyzer", "가스 분석기"),
    ),
    EvalCase(
        name="item_kappa_container_exact",
        query="아이템: 보안 컨테이너 카파",
        domain="item",
        expected_domain="item",
        expected_text=("보안 컨테이너 카파", "수집가"),
    ),
    EvalCase(
        name="story_tour_exact",
        query="스토리: Tour",
        domain="story",
        expected_domain="story",
        expected_text=("Tour", "그라운드 제로"),
    ),
    EvalCase(
        name="information_patch_notes",
        query="패치노트 내용 알려줘",
        domain="information",
        expected_domain="information",
        expected_text=("패치",),
    ),
]


def _doc_text(doc) -> str:
    metadata = doc.metadata or {}
    return "\n".join(
        [
            doc.domain,
            doc.entity_id,
            doc.chunk_id,
            doc.chunk_type,
            str(metadata.get("entity_name") or ""),
            doc.content,
        ]
    )


def _evaluate_docs(case: EvalCase, docs: list) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not docs:
        return False, ["no docs"]

    top = docs[0]
    if case.expected_domain and top.domain != case.expected_domain:
        reasons.append(f"top domain expected={case.expected_domain} actual={top.domain}")

    combined = "\n".join(_doc_text(doc) for doc in docs[:5]).lower()
    for expected in case.expected_text:
        if expected.lower() not in combined:
            reasons.append(f"missing text: {expected}")

    return not reasons, reasons


async def _collect_answer(case: EvalCase, limit: int) -> tuple[str, int, list[str]]:
    answer_parts: list[str] = []
    event_types: list[str] = []
    docs_count = 0

    async for event in run_rag_pipeline_stream_v3(
        session_id=str(uuid.uuid4()),
        user_query=case.query,
        lang="ko",
        rag_limit=limit,
        history_limit=3,
        domain=case.domain,
    ):
        if not event.startswith("data: "):
            continue
        data = json.loads(event[6:])
        event_type = data.get("type")
        event_types.append(event_type)
        if event_type == "docs":
            docs_count = len(data.get("docs", []))
        elif event_type == "token":
            answer_parts.append(data.get("content", ""))

    return "".join(answer_parts), docs_count, event_types


async def run_eval(
    cases: list[EvalCase],
    limit: int,
    with_answer: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        docs = await search_rag_v3(
            query=case.query,
            lang="ko",
            limit=limit,
            domain=case.domain,
        )
        passed, reasons = _evaluate_docs(case, docs)
        top_doc = docs[0] if docs else None
        result: dict[str, Any] = {
            "name": case.name,
            "query": case.query,
            "domain": case.domain,
            "passed": passed,
            "reasons": reasons,
            "docs": len(docs),
            "top": None,
        }
        if top_doc:
            result["top"] = {
                "domain": top_doc.domain,
                "entity_id": top_doc.entity_id,
                "chunk_id": top_doc.chunk_id,
                "chunk_type": top_doc.chunk_type,
                "entity_name": top_doc.metadata.get("entity_name"),
                "fused_score": top_doc.fused_score,
                "similarity": top_doc.similarity,
                "lexical_score": top_doc.lexical_score,
                "trigram_score": top_doc.trigram_score,
            }

        if with_answer:
            answer, docs_count, event_types = await _collect_answer(case, limit)
            result["answer_docs"] = docs_count
            result["answer_preview"] = answer[:600]
            result["event_types"] = event_types

        results.append(result)

    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    passed_count = sum(1 for result in results if result["passed"])
    print(f"V3 eval: {passed_count}/{len(results)} passed")
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"\n[{status}] {result['name']}")
        print(f"query: {result['query']}")
        print(f"docs: {result['docs']}")
        if result["top"]:
            top = result["top"]
            print(
                "top: "
                f"{top['domain']}/{top['entity_id']} "
                f"{top['chunk_type']} "
                f"name={top['entity_name']} "
                f"fused={top['fused_score']}"
            )
        if result["reasons"]:
            print("reasons:", "; ".join(result["reasons"]))
        if result.get("answer_preview"):
            print("answer:")
            print(result["answer_preview"])


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed V3 RAG evaluation cases.")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--case", action="append", dest="case_names")
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = EVAL_CASES
    if args.case_names:
        wanted = set(args.case_names)
        cases = [case for case in EVAL_CASES if case.name in wanted]
        missing = wanted - {case.name for case in cases}
        if missing:
            raise SystemExit(f"unknown eval cases: {', '.join(sorted(missing))}")

    try:
        results = await run_eval(cases, limit=args.limit, with_answer=args.with_answer)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            _print_results(results)
        if not all(result["passed"] for result in results):
            raise SystemExit(1)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
