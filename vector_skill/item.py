"""
item_i18n + item_detail_i18n 배치 임베딩 스크립트
- item_i18n + item_detail_i18n joined
- 카테고리별 info 파서 분리
- asyncio 동시 처리 (CONCURRENT_LIMIT)
- 이미 처리된 아이템 스킵 (재시작 안전)
- bge-m3로 임베딩 생성
- rag_documents 테이블에 upsert
"""

import asyncio
import asyncpg
import httpx
import json
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# 설정
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")
BATCH_SIZE = 50
CONCURRENT_LIMIT = 5  # 동시 처리 수
SKIP_EXISTING = False  # 기존 항목 스킵

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# 유틸
def get_lang_value(jsonb_field: dict | str | None, lang: str) -> str:
    if not jsonb_field:
        return ""
    if isinstance(jsonb_field, str):
        try:
            jsonb_field = json.loads(jsonb_field)
        except json.JSONDecodeError:
            return ""
    return jsonb_field.get(lang, "") or ""


def parse_jsonb(value) -> list | dict | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None


def yn(value: bool | None, lang: str) -> str:
    yes = {"ko": "예", "en": "Yes", "ja": "はい"}
    no = {"ko": "아니오", "en": "No", "ja": "いいえ"}
    return yes[lang] if value else no[lang]


def fmt(value, suffix="") -> str:
    if value is None:
        return ""
    return f"{value}{suffix}"


# 카테고리별 info 파서
def parse_ammo(info: dict, lang: str) -> str:
    label = {
        "ko": "데미지: {damage} | 관통력: {pen} | 장갑 데미지: {ad}%\n반동 보정: {rec} | 명중률 보정: {acc}",
        "en": "Damage: {damage} | Penetration: {pen} | Armor Damage: {ad}%\nRecoil: {rec} | Accuracy: {acc}",
        "ja": "ダメージ: {damage} | 貫通力: {pen} | アーマーダメージ: {ad}%\n反動補正: {rec} | 命中補正: {acc}",
    }[lang]
    return label.format(
        damage=fmt(info.get("damage")),
        pen=fmt(info.get("penetration_power")),
        ad=fmt(info.get("armor_damage")),
        rec=fmt(info.get("recoil_modifier")),
        acc=fmt(info.get("accuracy_modifier")),
    )


def parse_armor(info: dict, lang: str) -> str:
    zones_key = f"zones_{lang}"
    zones = (info.get("zones") or {}).get(zones_key, [])
    zone_str = ", ".join(zones) if zones else ""
    label = {
        "ko": f"방어 등급: {fmt(info.get('class_value'))} | 내구도: {fmt(info.get('durability'))} | 무게: {fmt(info.get('weight'))}kg\n재질: {fmt((info.get('material') or {}).get('name'))}\n방어 부위: {zone_str}",
        "en": f"Class: {fmt(info.get('class_value'))} | Durability: {fmt(info.get('durability'))} | Weight: {fmt(info.get('weight'))}kg\nMaterial: {fmt((info.get('material') or {}).get('name'))}\nZones: {zone_str}",
        "ja": f"防護クラス: {fmt(info.get('class_value'))} | 耐久度: {fmt(info.get('durability'))} | 重量: {fmt(info.get('weight'))}kg\n素材: {fmt((info.get('material') or {}).get('name'))}\n防護部位: {zone_str}",
    }[lang]
    return label


def parse_gun(info: dict, lang: str) -> str:
    modes_key = f"modes_{lang}"
    modes = (info.get("modes") or {}).get(modes_key, [])
    modes_str = ", ".join(modes) if modes else ""
    allowed = info.get("allowed_ammo") or []
    ammo_str = ", ".join(a.get("name", "") for a in allowed[:5])
    if len(allowed) > 5:
        ammo_str += f" 외 {len(allowed)-5}종"
    label = {
        "ko": f"종류: {fmt(info.get('gun_category'))} | 구경: {fmt(info.get('caliber'))} | 무게: {fmt(info.get('weight'))}kg\n발사 속도: {fmt(info.get('fire_rate'))}rpm | 발사 모드: {modes_str}\n기본 탄약: {fmt(info.get('default_ammo'))}\n사용 가능 탄약: {ammo_str}",
        "en": f"Type: {fmt(info.get('gun_category'))} | Caliber: {fmt(info.get('caliber'))} | Weight: {fmt(info.get('weight'))}kg\nFire Rate: {fmt(info.get('fire_rate'))}rpm | Modes: {modes_str}\nDefault Ammo: {fmt(info.get('default_ammo'))}\nAllowed Ammo: {ammo_str}",
        "ja": f"種類: {fmt(info.get('gun_category'))} | 口径: {fmt(info.get('caliber'))} | 重量: {fmt(info.get('weight'))}kg\n発射速度: {fmt(info.get('fire_rate'))}rpm | 発射モード: {modes_str}\nデフォルト弾薬: {fmt(info.get('default_ammo'))}\n使用可能弾薬: {ammo_str}",
    }[lang]
    return label


def parse_medical(info: dict, lang: str) -> str:
    cures = (info.get("cures") or {}).get(lang, [])
    cures_str = ", ".join(cures) if cures else ""
    buff = info.get("buff") or []
    malus = info.get("malus") or []
    skill_key = f"skill_name_{lang}"

    buff_str = ", ".join(
        f"{b.get(skill_key, b.get('type', ''))} +{b.get('value')} ({b.get('duration')}초)"
        for b in buff
        if b.get("value")
    )
    malus_str = ", ".join(
        f"{m.get(skill_key, m.get('type', ''))} ({m.get('duration')}초)" for m in malus
    )
    label = {
        "ko": f"종류: {fmt(info.get('medical_category'))} | 무게: {fmt(info.get('weight'))}kg | 사용 횟수: {fmt(info.get('uses'))}\n치료 효과: {cures_str}\n버프: {buff_str}\n디버프: {malus_str}",
        "en": f"Type: {fmt(info.get('medical_category'))} | Weight: {fmt(info.get('weight'))}kg | Uses: {fmt(info.get('uses'))}\nCures: {cures_str}\nBuff: {buff_str}\nDebuff: {malus_str}",
        "ja": f"種類: {fmt(info.get('medical_category'))} | 重量: {fmt(info.get('weight'))}kg | 使用回数: {fmt(info.get('uses'))}\n治療効果: {cures_str}\nバフ: {buff_str}\nデバフ: {malus_str}",
    }[lang]
    return label


def parse_key(info: dict, lang: str) -> str:
    use_map = (info.get("use_map") or {}).get(lang, []) or []
    map_str = ", ".join(use_map) if use_map else ""
    label = {
        "ko": f"사용 횟수: {fmt(info.get('uses'))} | 무게: {fmt(info.get('weight'))}kg\n사용 지도: {map_str}",
        "en": f"Uses: {fmt(info.get('uses'))} | Weight: {fmt(info.get('weight'))}kg\nUsable Map: {map_str}",
        "ja": f"使用回数: {fmt(info.get('uses'))} | 重量: {fmt(info.get('weight'))}kg\n使用マップ: {map_str}",
    }[lang]
    return label


def parse_backpack(info: dict, lang: str) -> str:
    label = {
        "ko": f"용량: {fmt(info.get('capacity'))}칸 | 무게: {fmt(info.get('weight'))}kg\n이동 패널티: {fmt(info.get('speed_penalty'))} | 인체공학 패널티: {fmt(info.get('ergo_penalty'))}",
        "en": f"Capacity: {fmt(info.get('capacity'))} | Weight: {fmt(info.get('weight'))}kg\nSpeed Penalty: {fmt(info.get('speed_penalty'))} | Ergo Penalty: {fmt(info.get('ergo_penalty'))}",
        "ja": f"容量: {fmt(info.get('capacity'))} | 重量: {fmt(info.get('weight'))}kg\n移動ペナルティ: {fmt(info.get('speed_penalty'))} | エルゴペナルティ: {fmt(info.get('ergo_penalty'))}",
    }[lang]
    return label


def parse_rig(info: dict, lang: str) -> str:
    zones_key = f"zones_{lang}"
    zones = (info.get("zones") or {}).get(zones_key, [])
    zone_str = ", ".join(zones) if zones else ""
    label = {
        "ko": f"용량: {fmt(info.get('capacity'))}칸 | 방어 등급: {fmt(info.get('class_value'))} | 내구도: {fmt(info.get('durability'))}\n무게: {fmt(info.get('weight'))}kg | 방어 부위: {zone_str}",
        "en": f"Capacity: {fmt(info.get('capacity'))} | Class: {fmt(info.get('class_value'))} | Durability: {fmt(info.get('durability'))}\nWeight: {fmt(info.get('weight'))}kg | Zones: {zone_str}",
        "ja": f"容量: {fmt(info.get('capacity'))} | 防護クラス: {fmt(info.get('class_value'))} | 耐久度: {fmt(info.get('durability'))}\n重量: {fmt(info.get('weight'))}kg | 防護部位: {zone_str}",
    }[lang]
    return label


def parse_headwear(info: dict, lang: str) -> str:
    zones_key = f"zones_{lang}"
    zones = (info.get("zones") or {}).get(zones_key, [])
    ricochet_key = f"ricochet_chance_{lang}"
    ricochet = (info.get("ricochet_chance") or {}).get(ricochet_key, "")
    label = {
        "ko": f"방어 등급: {fmt(info.get('class_value'))} | 내구도: {fmt(info.get('durability'))} | 무게: {fmt(info.get('weight'))}kg\n재질: {fmt((info.get('material') or {}).get('name'))} | 도탄 확률: {ricochet}\n방어 부위: {', '.join(zones)}",
        "en": f"Class: {fmt(info.get('class_value'))} | Durability: {fmt(info.get('durability'))} | Weight: {fmt(info.get('weight'))}kg\nMaterial: {fmt((info.get('material') or {}).get('name'))} | Ricochet: {ricochet}\nZones: {', '.join(zones)}",
        "ja": f"防護クラス: {fmt(info.get('class_value'))} | 耐久度: {fmt(info.get('durability'))} | 重量: {fmt(info.get('weight'))}kg\n素材: {fmt((info.get('material') or {}).get('name'))} | 跳弾確率: {ricochet}\n防護部位: {', '.join(zones)}",
    }[lang]
    return label


def parse_facecover(info: dict, lang: str) -> str:
    zones_key = f"zones_{lang}"
    zones = (info.get("zones") or {}).get(zones_key, [])
    ricochet_key = f"ricochet_chance_{lang}"
    ricochet = (info.get("ricochet_chance") or {}).get(ricochet_key, "")
    label = {
        "ko": f"방어 등급: {fmt(info.get('class_value'))} | 내구도: {fmt(info.get('durability'))} | 무게: {fmt(info.get('weight'))}kg\n도탄 확률: {ricochet} | 방어 부위: {', '.join(zones)}",
        "en": f"Class: {fmt(info.get('class_value'))} | Durability: {fmt(info.get('durability'))} | Weight: {fmt(info.get('weight'))}kg\nRicochet: {ricochet} | Zones: {', '.join(zones)}",
        "ja": f"防護クラス: {fmt(info.get('class_value'))} | 耐久度: {fmt(info.get('durability'))} | 重量: {fmt(info.get('weight'))}kg\n跳弾確率: {ricochet} | 防護部位: {', '.join(zones)}",
    }[lang]
    return label


def parse_throwable(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg | 퓨즈: {fmt(info.get('fuse'))}초\n파편: {fmt(info.get('fragments'))}개 | 폭발 범위: {fmt(info.get('min_explosion_distance'))}~{fmt(info.get('max_explosion_distance'))}m",
        "en": f"Weight: {fmt(info.get('weight'))}kg | Fuse: {fmt(info.get('fuse'))}s\nFragments: {fmt(info.get('fragments'))} | Blast Radius: {fmt(info.get('min_explosion_distance'))}~{fmt(info.get('max_explosion_distance'))}m",
        "ja": f"重量: {fmt(info.get('weight'))}kg | 信管: {fmt(info.get('fuse'))}秒\n破片: {fmt(info.get('fragments'))}個 | 爆発範囲: {fmt(info.get('min_explosion_distance'))}~{fmt(info.get('max_explosion_distance'))}m",
    }[lang]
    return label


def parse_provisions(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg | 에너지: {fmt(info.get('energy'))} | 수분: {fmt(info.get('hydration'))}",
        "en": f"Weight: {fmt(info.get('weight'))}kg | Energy: {fmt(info.get('energy'))} | Hydration: {fmt(info.get('hydration'))}",
        "ja": f"重量: {fmt(info.get('weight'))}kg | エネルギー: {fmt(info.get('energy'))} | 水分: {fmt(info.get('hydration'))}",
    }[lang]
    return label


def parse_loot(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg | 카테고리: {fmt(info.get('loot_category'))}",
        "en": f"Weight: {fmt(info.get('weight'))}kg | Category: {fmt(info.get('loot_category'))}",
        "ja": f"重量: {fmt(info.get('weight'))}kg | カテゴリ: {fmt(info.get('loot_category'))}",
    }[lang]
    return label


def parse_knife(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg | 찌르기 데미지: {fmt(info.get('stab_damage'))} | 베기 데미지: {fmt(info.get('slash_damage'))}\n범위: {fmt(info.get('hit_radius'))}m",
        "en": f"Weight: {fmt(info.get('weight'))}kg | Stab: {fmt(info.get('stab_damage'))} | Slash: {fmt(info.get('slash_damage'))}\nRange: {fmt(info.get('hit_radius'))}m",
        "ja": f"重量: {fmt(info.get('weight'))}kg | 刺突ダメージ: {fmt(info.get('stab_damage'))} | 斬撃ダメージ: {fmt(info.get('slash_damage'))}\n射程: {fmt(info.get('hit_radius'))}m",
    }[lang]
    return label


def parse_headset(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg | 거리 보정: {fmt(info.get('distance_modifier'))}",
        "en": f"Weight: {fmt(info.get('weight'))}kg | Distance Modifier: {fmt(info.get('distance_modifier'))}",
        "ja": f"重量: {fmt(info.get('weight'))}kg | 距離補正: {fmt(info.get('distance_modifier'))}",
    }[lang]
    return label


def parse_container(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg | 용량: {fmt(info.get('capacity'))}칸 | 크기: {fmt(info.get('width'))}x{fmt(info.get('height'))}",
        "en": f"Weight: {fmt(info.get('weight'))}kg | Capacity: {fmt(info.get('capacity'))} | Size: {fmt(info.get('width'))}x{fmt(info.get('height'))}",
        "ja": f"重量: {fmt(info.get('weight'))}kg | 容量: {fmt(info.get('capacity'))} | サイズ: {fmt(info.get('width'))}x{fmt(info.get('height'))}",
    }[lang]
    return label


def parse_armband(info: dict, lang: str) -> str:
    label = {
        "ko": f"무게: {fmt(info.get('weight'))}kg",
        "en": f"Weight: {fmt(info.get('weight'))}kg",
        "ja": f"重量: {fmt(info.get('weight'))}kg",
    }[lang]
    return label


def parse_glasses(info: dict, lang: str) -> str:
    label = {
        "ko": f"방어 등급: {fmt(info.get('class_value'))} | 내구도: {fmt(info.get('durability'))} | 무게: {fmt(info.get('weight'))}kg\n실명 방어: {fmt(info.get('blindness_protection'))}",
        "en": f"Class: {fmt(info.get('class_value'))} | Durability: {fmt(info.get('durability'))} | Weight: {fmt(info.get('weight'))}kg\nBlindness Protection: {fmt(info.get('blindness_protection'))}",
        "ja": f"防護クラス: {fmt(info.get('class_value'))} | 耐久度: {fmt(info.get('durability'))} | 重量: {fmt(info.get('weight'))}kg\n盲目防御: {fmt(info.get('blindness_protection'))}",
    }[lang]
    return label


def parse_default(info: dict, lang: str) -> str:
    weight = info.get("weight")
    if not weight:
        return ""
    label = {
        "ko": f"무게: {weight}kg",
        "en": f"Weight: {weight}kg",
        "ja": f"重量: {weight}kg",
    }[lang]
    return label


INFO_PARSERS = {
    "Ammo": parse_ammo,
    "ArmorVest": parse_armor,
    "Gun": parse_gun,
    "Medical": parse_medical,
    "Key": parse_key,
    "Backpack": parse_backpack,
    "Rig": parse_rig,
    "Headwear": parse_headwear,
    "FaceCover": parse_facecover,
    "Throwable": parse_throwable,
    "Provisions": parse_provisions,
    "Loot": parse_loot,
    "Knife": parse_knife,
    "Headset": parse_headset,
    "Container": parse_container,
    "Armband": parse_armband,
    "Glasses": parse_glasses,
    "Other": parse_default,
}


# item_detail 파서
DETAIL_LABELS = {
    "ko": {
        "hideout": "은신처 건설 필요",
        "crafts": "제작 재료",
        "npc": "상인 교환",
        "quest_reward": "퀘스트 보상",
        "quest_item": "퀘스트 제출 아이템",
        "craft_unlock": "제작 잠금해제 퀘스트",
        "offer_unlock": "구매 잠금해제 퀘스트",
        "lv": "레벨",
    },
    "en": {
        "hideout": "Required for Hideout",
        "crafts": "Used in Crafts",
        "npc": "Barter (NPC)",
        "quest_reward": "Quest Reward",
        "quest_item": "Quest Item",
        "craft_unlock": "Craft Unlock Quest",
        "offer_unlock": "Offer Unlock Quest",
        "lv": "Level",
    },
    "ja": {
        "hideout": "隠れ家建設に必要",
        "crafts": "クラフト素材",
        "npc": "商人交換",
        "quest_reward": "クエスト報酬",
        "quest_item": "クエストアイテム",
        "craft_unlock": "クラフト解放クエスト",
        "offer_unlock": "購入解放クエスト",
        "lv": "レベル",
    },
}

NAME_KEY = {"ko": "name_ko", "en": "name_en", "ja": "name_ja"}


def parse_detail(detail: dict | None, lang: str) -> str:
    if not detail:
        return ""
    lb = DETAIL_LABELS[lang]
    nk = NAME_KEY[lang]
    parts = []

    # 은신처 건설 필요
    hideout = detail.get("hideout_items") or []
    if hideout:
        lines = []
        for h in hideout:
            master = get_lang_value(h.get("master_name"), lang)
            level = h.get("level", "")
            count = h.get("count", "")
            lines.append(f"- {master} {lb['lv']}{level} x{count}")
        parts.append(f"[{lb['hideout']}]\n" + "\n".join(lines))

    # 제작 재료
    crafts = detail.get("used_in_crafts") or []
    if crafts:
        lines = []
        for c in crafts:
            name = get_lang_value(c.get("name"), lang)
            master = get_lang_value(c.get("master_name"), lang)
            level = c.get("level", "")
            lines.append(f"- {name} ({master} {lb['lv']}{level})")
        parts.append(f"[{lb['crafts']}]\n" + "\n".join(lines))

    # 상인 교환
    npcs = detail.get("rewarded_by_npcs") or []
    if npcs:
        lines = []
        for n in npcs:
            npc_name = get_lang_value(n.get("npc_name"), lang)
            barter = n.get("barter_info") or {}
            level = barter.get("level", "")
            req = barter.get("requiredItems") or []
            req_str = ", ".join(
                f"{r['item'].get(nk, '')} x{r.get('quantity', '')}"
                for r in req
                if r.get("item")
            )
            lines.append(f"- {npc_name} {lb['lv']}{level} | 필요: {req_str}")
        parts.append(f"[{lb['npc']}]\n" + "\n".join(lines))

    # 퀘스트 보상
    quest_rewards = detail.get("rewarded_by_quests") or []
    if quest_rewards:
        lines = []
        for q in quest_rewards:
            quest_name = get_lang_value(q.get("name"), lang)
            npc_name = get_lang_value(q.get("npc_name"), lang)
            reward = q.get("reward") or {}
            quantity = reward.get("quantity") or reward.get("count", "")
            lines.append(f"- {quest_name} ({npc_name}) → x{quantity}")
        parts.append(f"[{lb['quest_reward']}]\n" + "\n".join(lines))

    # 퀘스트 제출 아이템
    quest_items = detail.get("required_by_quest_item") or []
    if quest_items:
        lines = []
        for q in quest_items:
            quest_name = get_lang_value(q.get("name"), lang)
            npc_name = get_lang_value(q.get("npc_name"), lang)
            obj = q.get("objective") or {}
            desc_key = f"description_{lang}"
            desc = obj.get(desc_key, "")
            lines.append(f"- {quest_name} ({npc_name}): {desc}")
        parts.append(f"[{lb['quest_item']}]\n" + "\n".join(lines))

    # 제작 잠금해제
    craft_unlocks = detail.get("rewarded_by_quests_craft_unlock") or []
    if craft_unlocks:
        lines = []
        for q in craft_unlocks:
            quest_name = get_lang_value(q.get("name"), lang)
            npc_name = get_lang_value(q.get("npc_name"), lang)
            reward = q.get("reward") or {}
            trader = reward.get("trader") or {}
            trader_name = trader.get(nk, "")
            level = reward.get("level", "")
            lines.append(
                f"- {quest_name} ({npc_name}) → {trader_name} {lb['lv']}{level}"
            )
        parts.append(f"[{lb['craft_unlock']}]\n" + "\n".join(lines))

    # 구매 잠금해제
    offer_unlocks = detail.get("rewarded_by_quests_offer_unlock") or []
    if offer_unlocks:
        lines = []
        for q in offer_unlocks:
            quest_name = get_lang_value(q.get("name"), lang)
            npc_name = get_lang_value(q.get("npc_name"), lang)
            reward = q.get("reward") or {}
            trader = reward.get("trader") or {}
            trader_name = trader.get(nk, "")
            level = reward.get("level", "")
            lines.append(
                f"- {quest_name} ({npc_name}) → {trader_name} {lb['lv']}{level}"
            )
        parts.append(f"[{lb['offer_unlock']}]\n" + "\n".join(lines))

    return "\n\n".join(parts)


# content 조합
CATEGORY_LABEL = {
    "ko": "카테고리",
    "en": "Category",
    "ja": "カテゴリ",
}
SPEC_LABEL = {
    "ko": "스펙",
    "en": "Spec",
    "ja": "スペック",
}
ITEM_LABEL = {
    "ko": "아이템",
    "en": "Item",
    "ja": "アイテム",
}


def build_content(item_row: dict, detail_row: dict | None, lang: str) -> str:
    name = get_lang_value(item_row["name"], lang)
    category = item_row.get("category") or ""
    info = parse_jsonb(item_row.get("info")) or {}

    parts = [
        f"{ITEM_LABEL[lang]}: {name}",
        f"{CATEGORY_LABEL[lang]}: {category}",
    ]

    # 카테고리별 스펙 파싱
    parser = INFO_PARSERS.get(category, parse_default)
    spec = parser(info, lang).strip()
    if spec:
        parts.append(f"\n[{SPEC_LABEL[lang]}]\n{spec}")

    # 상세 정보 (hideout, quest, npc 등)
    if detail_row:
        detail = {
            "hideout_items": parse_jsonb(detail_row.get("hideout_items")) or [],
            "used_in_crafts": parse_jsonb(detail_row.get("used_in_crafts")) or [],
            "rewarded_by_npcs": parse_jsonb(detail_row.get("rewarded_by_npcs")) or [],
            "rewarded_by_quests": parse_jsonb(detail_row.get("rewarded_by_quests"))
            or [],
            "required_by_quest_item": parse_jsonb(
                detail_row.get("required_by_quest_item")
            )
            or [],
            "rewarded_by_quests_craft_unlock": parse_jsonb(
                detail_row.get("rewarded_by_quests_craft_unlock")
            )
            or [],
            "rewarded_by_quests_offer_unlock": parse_jsonb(
                detail_row.get("rewarded_by_quests_offer_unlock")
            )
            or [],
        }
        detail_text = parse_detail(detail, lang).strip()
        if detail_text:
            parts.append(f"\n{detail_text}")

    return "\n".join(parts).strip()


# 임베딩 생성
async def get_embedding(client: httpx.AsyncClient, text: str) -> list[float]:
    response = await client.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": text},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


# DB upsert
async def upsert_rag_document(
    conn: asyncpg.Connection,
    source_id: str,
    lang: str,
    content: str,
    embedding: list[float],
    metadata: dict,
):
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    await conn.execute(
        """
        INSERT INTO rag_documents (source_table, source_id, lang, content, embedding, metadata)
        VALUES ($1, $2, $3, $4, $5::vector, $6)
        ON CONFLICT (source_table, source_id, lang)
        DO UPDATE SET
            content    = EXCLUDED.content,
            embedding  = EXCLUDED.embedding,
            metadata   = EXCLUDED.metadata,
            updated_at = NOW()
    """,
        "item_i18n",
        source_id,
        lang,
        content,
        embedding_str,
        json.dumps(metadata, ensure_ascii=False),
    )


# 아이템 처리 (동시 처리용)
async def process_item(
    pool: asyncpg.Pool,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    item_row: dict,
    detail_row: dict | None,
):
    item_id = item_row["id"]

    async with semaphore:
        for lang in LANGS:
            content = build_content(item_row, detail_row, lang)

            if not content.strip():
                log.warning(f"  ⚠ 빈 content 스킵: {item_id} [{lang}]")
                continue

            try:
                embedding = await get_embedding(client, content)

                metadata = {
                    "content_type": "joined",
                    "source_tables": ["item_i18n", "item_detail_i18n"],
                    "item_id": item_id,
                    "item_name": {
                        "ko": get_lang_value(item_row["name"], "ko"),
                        "en": get_lang_value(item_row["name"], "en"),
                        "ja": get_lang_value(item_row["name"], "ja"),
                    },
                    "category": item_row.get("category") or "",
                    "url": f"https://eftlibrary.com/item/info/{item_row.get('url_mapping') or item_id}",
                }

                # ── 확인용 출력 ──────────────────────────────────────
                # print(f"\n{'='*60}")
                # print(f"[{item_id}] [{lang}]")
                # print(f"{'─'*60}")
                # print(f"[content]\n{content}")
                # print(f"{'─'*60}")
                # print(f"[embedding] 차원: {len(embedding)} | 앞 5개: {embedding[:5]}")
                # print(f"{'='*60}")
                # ────────────────────────────────────────────────────

                async with pool.acquire() as conn:
                    await upsert_rag_document(
                        conn, item_id, lang, content, embedding, metadata
                    )
                log.info(f"  ✓ {item_id} [{lang}] 완료")

            except httpx.HTTPError as e:
                log.error(f"  ✗ 임베딩 실패: {item_id} [{lang}] - {e}")
            except asyncpg.PostgresError as e:
                log.error(f"  ✗ DB 저장 실패: {item_id} [{lang}] - {e}")


# 메인
LANGS = ["ko", "en", "ja"]


async def main():
    log.info("=== item_i18n 배치 임베딩 시작 ===")
    log.info(
        f"모델: {EMBED_MODEL} | 동시처리: {CONCURRENT_LIMIT} | 기존 스킵: {SKIP_EXISTING}"
    )

    # 커넥션 풀 사용 (동시 처리 시 커넥션 충돌 방지)
    pool = await asyncpg.create_pool(
        DATABASE_URL, min_size=2, max_size=CONCURRENT_LIMIT + 2
    )
    client = httpx.AsyncClient()
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)

    try:
        # 이미 처리된 아이템 ID 조회
        existing_ids = set()
        if SKIP_EXISTING:
            async with pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT DISTINCT source_id FROM rag_documents
                    WHERE source_table = 'item_i18n'
                """)
            existing_ids = {r["source_id"] for r in rows}
            log.info(f"이미 처리된 아이템: {len(existing_ids)}개 스킵")

        async with pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM item_i18n")
        log.info(f"전체 아이템: {total}개 | 처리 대상: {total - len(existing_ids)}개")

        offset = 0
        processed = 0

        while offset < total:
            async with pool.acquire() as conn:
                item_rows = await conn.fetch(
                    """
                    SELECT i.id, i.name, i.category, i.url_mapping, i.info
                    FROM item_i18n i
                    ORDER BY i.id ASC
                    LIMIT $1 OFFSET $2
                """,
                    BATCH_SIZE,
                    offset,
                )

            if not item_rows:
                break

            # 배치 내 아이템 ID 목록
            batch_ids = [r["id"] for r in item_rows]

            # 스킵 필터
            target_ids = [id for id in batch_ids if id not in existing_ids]
            if not target_ids:
                offset += BATCH_SIZE
                continue

            # item_detail 한번에 조회
            async with pool.acquire() as conn:
                detail_rows = await conn.fetch(
                    """
                    SELECT id, hideout_items, used_in_crafts, rewarded_by_npcs,
                           rewarded_by_quests, required_by_quest_item,
                           rewarded_by_quests_craft_unlock, rewarded_by_quests_offer_unlock
                    FROM item_detail_i18n
                    WHERE id = ANY($1)
                """,
                    target_ids,
                )

            detail_map = {r["id"]: dict(r) for r in detail_rows}

            log.info(
                f"배치 처리중: {offset + 1} ~ {offset + len(item_rows)} / {total} | 대상: {len(target_ids)}개"
            )

            # 동시 처리 (pool을 넘겨서 각 태스크가 개별 커넥션 획득)
            tasks = [
                process_item(pool, client, semaphore, dict(r), detail_map.get(r["id"]))
                for r in item_rows
                if r["id"] in target_ids
            ]
            await asyncio.gather(*tasks)

            processed += len(target_ids)
            offset += BATCH_SIZE

            log.info(f"누적 처리: {processed}개")

        log.info(
            f"=== 완료: {processed}개 아이템, {processed * 3}개 row 생성/업데이트 ==="
        )

    finally:
        await client.aclose()
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
