import argparse
import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from db.connection import close_pool
from services.rag_v3 import run_rag_pipeline_stream_v3
from tools.retriever_v3 import search_rag_v3

load_dotenv()

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalCase:
    name: str
    query: str
    domain: str | None = None
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
        query="데뷔 정보 알려줘",
        domain="quest",
        expected_domain="quest",
        expected_text=("데뷔",),
    ),
    EvalCase(
        name="quest_debut_prereq",
        query="데뷔 이전에는 어떤 퀘스트를 완료해야 하니?",
        domain="quest",
        expected_domain="quest",
        expected_text=("데뷔", "선행 퀘스트", "사격 연습"),
    ),
    EvalCase(
        name="quest_shooting_practice_next",
        query="사격 연습 다음에는 어떤 퀘스트가 열려?",
        domain="quest",
        expected_domain="quest",
        expected_text=("사격 연습", "후행 퀘스트", "데뷔"),
    ),
    EvalCase(
        name="quest_collector_exact",
        query="수집가 정보 알려줘",
        domain="quest",
        expected_domain="quest",
        expected_text=("수집가",),
    ),
    EvalCase(
        name="quest_gunsmith_24_exact",
        query="Gunsmith - Part 24 정보 알려줘",
        domain="quest",
        expected_domain="quest",
        expected_text=("Gunsmith - Part 24", "Gunsmith - Part 23"),
    ),
    EvalCase(
        name="quest_gunsmith_7_unlocks",
        query="건스미스파트 7을 깨면 뭐가 해금돼?",
        domain="quest",
        expected_domain="quest",
        expected_text=("건스미스 - 파트 7", "구매 해금", "Daniel Defense MK12"),
    ),
    EvalCase(
        name="quest_shortage_required_items_relation",
        query="재고 부족에 뭐 필요해?",
        domain="quest",
        expected_domain="quest",
        expected_text=("재고 부족", "필요 아이템", "Salewa"),
    ),
    EvalCase(
        name="boss_tagilla_info",
        query="타길라 정보 알려줘",
        domain="boss",
        expected_domain="boss",
        expected_text=("타길라", "스폰 위치"),
    ),
    EvalCase(
        name="boss_tagilla_spawn_relation",
        query="타길라 어디 나와?",
        domain="boss",
        expected_domain="boss",
        expected_text=("타길라", "스폰 위치", "공장"),
    ),
    EvalCase(
        name="boss_killa_drops_relation",
        query="킬라가 드랍하는 아이템",
        domain="boss",
        expected_domain="boss",
        expected_text=("킬라", "드랍 아이템"),
    ),
    EvalCase(
        name="quest_glukhar_kill_auto_route",
        query="글루하 잡는 퀘스트",
        domain=None,
        expected_domain="quest",
        expected_text=("글루하", "Payback", "사냥꾼의 길 - 말살자 - 파트 1"),
    ),
    EvalCase(
        name="boss_cultist_priest_info",
        query="광신도 사제 정보 알려줘",
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
        query="Rusted bloody key 정보",
        domain="item",
        expected_domain="item",
        expected_text=("Rusted bloody key", "The Door"),
    ),
    EvalCase(
        name="item_m80_exact",
        query="7.62x51mm M80 정보",
        domain="item",
        expected_domain="item",
        expected_text=("7.62x51mm M80",),
    ),
    EvalCase(
        name="item_m80_unlock_relation",
        query="M80 구매 해금 퀘스트 뭐야?",
        domain="item",
        expected_domain="item",
        expected_text=("7.62x51mm M80", "구매를 해금하는 퀘스트", "개정 작업 - 등대"),
    ),
    EvalCase(
        name="item_ak103_reward_quest_relation",
        query="Kalashnikov AK-103을 보상으로 얻을 수 있는 퀘스트",
        domain="item",
        expected_domain="item",
        expected_text=("Kalashnikov AK-103", "이 아이템을 보상으로 주는 퀘스트"),
    ),
    EvalCase(
        name="item_gas_analyzer_korean",
        query="가스분석기 정보",
        domain="item",
        expected_domain="item",
        expected_text=("Gas analyzer", "가스 분석기"),
    ),
    EvalCase(
        name="item_gas_analyzer_quest_relation",
        query="가스분석기 필요한 퀘스트",
        domain="item",
        expected_domain="item",
        expected_text=("Gas analyzer", "이 아이템과 관련된 퀘스트 목표"),
    ),
    EvalCase(
        name="item_duct_tape_uses_relation",
        query="덕트 테이프 어디에 써?",
        domain="item",
        expected_domain="item",
        expected_text=("Duct tape", "이 아이템을 재료로 제작 가능한 아이템"),
    ),
    EvalCase(
        name="item_kappa_container_exact",
        query="아이템: 보안 컨테이너 카파",
        domain="item",
        expected_domain="item",
        expected_text=("보안 컨테이너 카파", "수집가"),
    ),
    EvalCase(
        name="hideout_bitcoin_requirements_relation",
        query="비트코인 채굴 시설 필요 재료",
        domain="hideout",
        expected_domain="hideout",
        expected_text=("비트코인 채굴 시설", "필요 아이템"),
    ),
    EvalCase(
        name="trader_therapist_level2_barters",
        query="테라피스트 2레벨 상점에서 뭐 팔아?",
        domain="trader",
        expected_domain="trader",
        expected_text=("테라피스트", "LL2"),
    ),
    EvalCase(
        name="story_tour_exact",
        query="스토리 Tour 정보",
        domain="story",
        expected_domain="story",
        expected_text=("Tour", "그라운드 제로"),
    ),
    EvalCase(
        name="story_savior_route_relation",
        query="구원자 루트는 모든 스토리 챕터를 클리어해야해?",
        domain="story",
        expected_domain="story",
        expected_text=("구원자", "모든 스토리"),
    ),
    EvalCase(
        name="information_patch_notes",
        query="최신 패치노트 내용 알려줘",
        domain="information",
        expected_domain="information",
        expected_text=("Patch",),
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


async def _collect_answer(
    case: EvalCase,
    limit: int,
    domain: str | None,
) -> tuple[str, int, list[str]]:
    answer_parts: list[str] = []
    event_types: list[str] = []
    docs_count = 0

    async for event in run_rag_pipeline_stream_v3(
        session_id=str(uuid.uuid4()),
        user_query=case.query,
        lang="ko",
        rag_limit=limit,
        history_limit=3,
        domain=domain,
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
    use_case_domain: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(cases)
    for index, case in enumerate(cases, 1):
        search_domain = case.domain if use_case_domain else None
        log.info(
            "[eval_v3] %s/%s start case=%s domain=%s query=%s",
            index,
            total,
            case.name,
            search_domain or "auto",
            case.query,
        )
        docs = await search_rag_v3(
            query=case.query,
            lang="ko",
            limit=limit,
            domain=search_domain,
        )
        passed, reasons = _evaluate_docs(case, docs)
        top_doc = docs[0] if docs else None
        result: dict[str, Any] = {
            "name": case.name,
            "query": case.query,
            "domain": search_domain,
            "case_domain": case.domain,
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
            log.info("[eval_v3] %s/%s answer case=%s", index, total, case.name)
            answer, docs_count, event_types = await _collect_answer(
                case,
                limit,
                search_domain,
            )
            result["answer_docs"] = docs_count
            result["answer_preview"] = answer[:600]
            result["event_types"] = event_types

        results.append(result)
        top_summary = "none"
        if top_doc:
            top_summary = (
                f"{top_doc.domain}/{top_doc.entity_id} "
                f"{top_doc.chunk_type} name={top_doc.metadata.get('entity_name')} "
                f"fused={top_doc.fused_score}"
            )
        log.info(
            "[eval_v3] %s/%s done case=%s passed=%s docs=%s top=%s reasons=%s",
            index,
            total,
            case.name,
            passed,
            len(docs),
            top_summary,
            "; ".join(reasons) if reasons else "-",
        )

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
    parser.add_argument(
        "--use-case-domain",
        action="store_true",
        help=(
            "Use each eval case's domain hint. By default eval searches without a "
            "domain to better match real user traffic."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cases = EVAL_CASES
    if args.case_names:
        wanted = set(args.case_names)
        cases = [case for case in EVAL_CASES if case.name in wanted]
        missing = wanted - {case.name for case in cases}
        if missing:
            raise SystemExit(f"unknown eval cases: {', '.join(sorted(missing))}")

    try:
        results = await run_eval(
            cases,
            limit=args.limit,
            with_answer=args.with_answer,
            use_case_domain=args.use_case_domain,
        )
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
