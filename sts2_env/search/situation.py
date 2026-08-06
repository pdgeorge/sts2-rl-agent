"""A combat about to start, as plain data, and how to rebuild it.

Two things need this and they must not grow separate implementations. The
benchmark needs to replay the same 200 fights against every candidate agent, and
the live bridge needs to turn the mod's JSON into something the searcher can
clone. Both are "build a CombatState from a description"; keeping them one
module is what stops them disagreeing later about what a description means.

It is JSON rather than a pickle deliberately. A pickled CombatState is exact and
free, and it breaks the first time a field is renamed -- which is precisely the
event this project has to survive. A named field that no longer resolves fails
loudly at load, which is the failure this codebase keeps choosing on purpose.

The encounter is stored as the setup function's name plus the seed it was rolled
with, so the enemies come back with the same HP rolls rather than merely the same
species. When the bridge path lands it will carry explicit enemies instead --
there is no encounter function to name when the fight is already in progress --
which is why `to_combat` dispatches rather than assuming.

WHAT IS AND IS NOT REPRODUCED

Reproduced exactly, every time: the deck, HP, relics, potions, the enemies and
their HP rolls, their opening intents and powers. Two rebuilds of one situation
are identical, which is the property a benchmark actually rests on -- every agent
faces the same fight.

Not reproduced: the opening shuffle of the run it was harvested from. A run's
`shuffle` stream has been advanced by every fight before this one, and the rebuilt
run starts that stream at zero, so the same deck deals a different opening hand.
Restoring the position would mean serialising a `_DotNetCompatRandom`'s internal
array -- brittle across builds in exactly the way this module chose JSON to avoid.
A fresh draw from the right deck is also the better test: it does not bake one
lucky opening into the fixture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from sts2_env.cards.base import CardInstance, reset_instance_counter
from sts2_env.cards.factory import create_card
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import CardId, RoomType
from sts2_env.core.rng import Rng
from sts2_env.potions.base import create_potion
from sts2_env.run.rooms import create_room
from sts2_env.run.run_state import RunState


# ---------------------------------------------------------------------------
# Encounter registry
# ---------------------------------------------------------------------------

_ENCOUNTER_MODULES = (
    "sts2_env.encounters.act1",
    "sts2_env.encounters.act2",
    "sts2_env.encounters.act3",
    "sts2_env.encounters.act4",
)

_ENCOUNTER_CACHE: dict[str, Callable[..., None]] | None = None


def encounter_registry() -> dict[str, Callable[..., None]]:
    """Every `setup_*` encounter function, by name.

    Built by import rather than by hand: an encounter added to an act module is
    resolvable here without anyone remembering to register it.
    """
    global _ENCOUNTER_CACHE
    if _ENCOUNTER_CACHE is not None:
        return _ENCOUNTER_CACHE

    import importlib

    registry: dict[str, Callable[..., None]] = {}
    for module_name in _ENCOUNTER_MODULES:
        module = importlib.import_module(module_name)
        for name in dir(module):
            if not name.startswith("setup_"):
                continue
            value = getattr(module, name)
            if callable(value):
                registry[name] = value
    _ENCOUNTER_CACHE = registry
    return registry


def _setup_name_for_encounter_id(encounter_id: str) -> str:
    """Normalise any of the names a bridge may send for an encounter into the
    `setup_X` form `encounter_registry` keys on.

    The mod sends `EncounterModel.Id.Entry` (PascalCase class name like
    "NibbitsWeak"). The Python registry keys on the setup-function name
    ("setup_nibbits_weak"). Round-tripping requires a normaliser rather
    than a one-shot renaming convention, because both representations
    exist on the wire and a fixture might carry either depending on which
    path wrote it.
    """
    name = str(encounter_id)
    if name.startswith("setup_"):
        return name.lower()
    # Strip a `setup_`-prefixed substring at the end (e.g. "TheKin setup_the_kin"
    # never happens, but stay safe). Pass through anything that already looks
    # like a setup name unchanged after lowercasing.
    # PascalCase -> snake_case: insert _ before each uppercase that follows a
    # lowercase, and before each uppercase run that ends in lowercase. Mirrors
    # the regex in RESEARCH.md's `class_name_to_id`.
    import re

    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
    return "setup_" + s.lower()


def resolve_encounter(name: str) -> Callable[..., None]:
    """Look up an encounter setup function by any of its common names.

    Accepts the Python `setup_X` form (the registry key, what
    `from_run_manager` writes), the C# `EncounterModel.Id.Entry` PascalCase
    form (what RlRunInfo sends over the wire), and the UPPER_SNAKE form
    some intermediate fixtures have used. All three resolve to the same
    function.
    """
    registry = encounter_registry()
    if name in registry:
        return registry[name]
    normalised = _setup_name_for_encounter_id(name)
    if normalised in registry:
        return registry[normalised]
    raise KeyError(
        f"No encounter setup named {name!r}. A fixture written against an "
        f"older build can name an encounter this one has renamed or removed; "
        f"regenerate the fixture rather than editing it by hand."
    )


# ---------------------------------------------------------------------------
# The situation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardRef:
    """A card in a deck: which one, and whether it is upgraded."""

    card_id: str
    upgraded: bool = False

    def instantiate(self) -> CardInstance:
        try:
            cid = CardId[self.card_id]
        except KeyError as exc:
            raise KeyError(
                f"No card named {self.card_id!r} in this build. Regenerate the "
                f"fixture against the current game build."
            ) from exc
        return create_card(cid, upgraded=self.upgraded)


@dataclass(frozen=True)
class CombatSituation:
    """One fight, at the moment it begins, reproducible from these fields alone."""

    situation_id: str
    character_id: str
    current_hp: int
    max_hp: int
    deck: tuple[CardRef, ...]
    encounter: str
    encounter_seed: int
    combat_seed: int
    relics: tuple[str, ...] = ()
    potions: tuple[str | None, ...] = ()
    max_potion_slots: int = 3
    gold: int = 0
    room_type: str = "MONSTER"
    act_floor: int = 1
    total_floor: int = 1
    ascension_level: int = 0

    # -- construction ------------------------------------------------------

    def to_combat(self) -> CombatState:
        """Rebuild the fight. Same inputs give the same enemies, every time.

        Mirrors `RunManager._enter_combat` step for step; that ordering is
        load-bearing, because relics and room modifiers fire during setup and a
        different order gives a different opening state.

        The `RunState` is not decoration. `CombatState.shuffle_rng` and
        `monster_ai_rng` resolve through `player_state.run_state.rng`, falling
        back to the combat's own RNG when there is none -- so a player built
        without one draws its shuffles and its enemy moves from a different
        stream than any real run does. The fight would still be reproducible,
        and it would not be representative. Building the streams from
        `combat_seed` gives both.
        """
        reset_instance_counter()

        deck = [ref.instantiate() for ref in self.deck]
        potions = [_instantiate_potion(pid, i) for i, pid in enumerate(self.potions)]

        run_state = RunState(
            seed=self.combat_seed,
            ascension_level=self.ascension_level,
            character_id=self.character_id,
        )
        run_state.act_floor = self.act_floor
        run_state.total_floor = self.total_floor

        player = run_state.player
        player.max_hp = self.max_hp
        player.current_hp = self.current_hp
        player.gold = self.gold
        player.deck = deck
        # In place: RunState aliases `self.relics` to this exact list object, so
        # rebinding the attribute would leave run_state.relics pointing at the
        # old one and the two would disagree about what the player owns.
        player.relics[:] = list(self.relics)
        player.potions = potions
        player.max_potion_slots = self.max_potion_slots

        room = create_room(RoomType[self.room_type])
        combat = CombatState(
            player_hp=self.current_hp,
            player_max_hp=self.max_hp,
            deck=deck,
            rng_seed=self.combat_seed,
            relics=list(self.relics),
            gold=self.gold,
            character_id=self.character_id,
            potions=potions,
            max_potion_slots=self.max_potion_slots,
            player_state=player,
            room=room,
            ascension_level=self.ascension_level,
        )

        resolve_encounter(self.encounter)(combat, Rng(self.encounter_seed))
        combat.start_combat()
        return combat

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["deck"] = [{"card_id": c.card_id, "upgraded": c.upgraded} for c in self.deck]
        data["relics"] = list(self.relics)
        data["potions"] = list(self.potions)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CombatSituation:
        return cls(
            situation_id=data["situation_id"],
            character_id=data["character_id"],
            current_hp=int(data["current_hp"]),
            max_hp=int(data["max_hp"]),
            deck=tuple(
                CardRef(card_id=c["card_id"], upgraded=bool(c.get("upgraded", False)))
                for c in data["deck"]
            ),
            encounter=data["encounter"],
            encounter_seed=int(data["encounter_seed"]),
            combat_seed=int(data["combat_seed"]),
            relics=tuple(data.get("relics", ())),
            potions=tuple(data.get("potions", ())),
            max_potion_slots=int(data.get("max_potion_slots", 3)),
            gold=int(data.get("gold", 0)),
            room_type=data.get("room_type", "MONSTER"),
            act_floor=int(data.get("act_floor", 1)),
            total_floor=int(data.get("total_floor", 1)),
            ascension_level=int(data.get("ascension_level", 0)),
        )

    # -- capture -----------------------------------------------------------

    @classmethod
    def from_run_manager(cls, mgr: Any, situation_id: str) -> CombatSituation:
        """Snapshot the fight a RunManager has just entered.

        Reads `_last_encounter`, which `_enter_combat` records for exactly this
        purpose: the setup function and seed are chosen inside that method and
        are otherwise unrecoverable afterwards -- the enemies exist, but which
        roll produced them does not.
        """
        encounter = getattr(mgr, "_last_encounter", None)
        if encounter is None:
            raise ValueError(
                "RunManager has not entered a combat, or this build does not "
                "record the encounter it rolled."
            )
        name, encounter_seed, combat_seed = encounter

        player = mgr.run_state.player
        return cls(
            situation_id=situation_id,
            character_id=player.character_id,
            current_hp=player.current_hp,
            max_hp=player.max_hp,
            deck=tuple(
                CardRef(card_id=card.card_id.name, upgraded=bool(card.upgraded))
                for card in player.deck
            ),
            encounter=name,
            encounter_seed=encounter_seed,
            combat_seed=combat_seed,
            relics=tuple(mgr.run_state.relics),
            potions=tuple(
                (p.potion_id if p is not None else None)
                for p in (player.potions or [])
            ),
            max_potion_slots=player.max_potion_slots,
            gold=player.gold,
            room_type=_room_type_name(mgr._current_room_type),
            act_floor=mgr.run_state.act_floor,
            total_floor=mgr.run_state.total_floor,
            ascension_level=mgr.run_state.ascension_level,
        )

    @classmethod
    def from_bridge_state(
        cls,
        state: dict[str, Any],
        *,
        situation_id: str | None = None,
    ) -> CombatSituation:
        """Build a CombatSituation from a bridge combat state message.

        The counterpart to `from_run_manager` for the live-game path: where
        the simulator reads from a RunManager mid-run, this reads from the
        JSON payload the C# mod sends at the moment a fight begins. The two
        must agree on every field, because `to_combat` is called on the
        result and a searcher that clones a fight different from the one on
        screen is worse than useless -- it looks correct and is not.

        WHAT THE BRIDGE MUST SEND

        Run-level fields (act, floor, run_hp, run_max_hp, gold, relics,
        potion_slots, deck, room_type, ascension, act_floor) arrive via
        RlRunInfo.Attach on every state. The current mod sends `deck` as a
        list of bare card-id strings -- the upgraded flag is lost, which
        the Phase 1.1 mod patch extends to `{"id", "upgraded"}` dicts.

        Encounter identification (the `encounter` setup-function name,
        `encounter_seed`, and `combat_seed`) is NOT sent by the current
        mod. These three are required: `to_combat` dispatches to
        `resolve_encounter` with the seed, which is the only way to bring
        the same enemies back with the same HP rolls. They are added by the
        Phase 1.1 mod patch; without them this method raises, which is the
        right failure -- a quiet fallback to a random encounter would have
        the search planning against a different fight from the one on
        screen.

        `character_id` is not currently sent by the mod, which is
        hardcoded to Ironclad (RlAutoSlayer.PreferredCharacterId). It
        defaults to "Ironclad" here and is added by the Phase 1.1 patch.
        """
        floor = int(state.get("floor", 0) or 0)
        sid = situation_id or f"bridge-f{floor:02d}"

        # Combat fields are nested inside `combat_state` by some handlers and
        # flat in others -- same fallback as state_adapter.py. Used only for
        # player HP/max_hp when run_hp/run_max_hp are absent, which they
        # always are in the current mod.
        combat = state.get("combat_state") or state
        player = combat.get("player") or {}

        # Deck: accept both the current mod format (list of id strings, the
        # upgraded flag lost) and the target format (list of dicts with id
        # and upgraded). The Phase 1.1 patch moves the current mod to the
        # target; until then an upgraded card is read as a base card, which
        # reproduces the wrong fight rather than a half-right one.
        raw_deck = state.get("deck") or []
        deck = tuple(_parse_deck_entry(d) for d in raw_deck)

        # Encounter identification. Required because `to_combat` calls
        # `resolve_encounter(self.encounter)(combat, Rng(self.encounter_seed))`
        # -- the setup function recreates the enemies, and the seed fixes which
        # HP roll they got. Missing encounter raises here rather than inside
        # `to_combat` because the message is clearer at the point of missing
        # data, and a caller that wants to handle the gap (live_search) needs
        # to know before the fight is entered.
        encounter = state.get("encounter")
        if not encounter:
            raise ValueError(
                "Bridge state has no `encounter` field. The mod must be patched "
                "(Phase 1.1) to send the encounter setup-function name and its "
                "seed; without them the SearchAgent would clone a fight that "
                "does not match the one on screen."
            )
        encounter_seed = int(state.get("encounter_seed", 0) or 0)
        combat_seed = int(state.get("combat_seed", encounter_seed) or 0)

        # Potions: the mod sends `potion_slots` as a positional list with
        # null where empty, same shape as from_run_manager. `potions` is the
        # combat-only list of usable potions, different and not what we want.
        raw_potions = state.get("potion_slots") or state.get("potions") or ()

        # Room type: the mod sends MapPointType as a string (e.g. "Monster",
        # "Elite", "Boss"). Normalised to uppercase for consistency with
        # `_room_type_name`, which is what from_run_manager produces.
        room_type = str(state.get("room_type") or "MONSTER").upper()

        return cls(
            situation_id=sid,
            character_id=(
                state.get("character_id")
                or state.get("character")
                or "Ironclad"
            ),
            current_hp=int(state.get("run_hp", 0) or player.get("hp", 0) or 0),
            max_hp=int(state.get("run_max_hp", 0) or player.get("max_hp", 0) or 0),
            deck=deck,
            encounter=encounter,
            encounter_seed=encounter_seed,
            combat_seed=combat_seed,
            relics=tuple(state.get("relics") or ()),
            potions=tuple(raw_potions),
            max_potion_slots=int(state.get("max_potion_slots") or 3),
            gold=int(state.get("gold") or 0),
            room_type=room_type,
            act_floor=int(state.get("act_floor") or 1),
            total_floor=floor,
            ascension_level=int(state.get("ascension") or 0),
        )


def _instantiate_potion(potion_id: str | None, slot: int):
    """Same contract as cards and encounters: a name this build does not have
    fails here, saying so, rather than further in as a missing effect.

    Returns ``None`` for an empty slot or an unknown potion id. A potion the
    bridge reports that the simulator does not know (a new game-patch potion
    the simulator has not caught up to, or a transient id we cannot coerce)
    is dropped rather than crashing the whole ``to_combat`` build -- a
    searcher that clones a fight missing one potion is still useful; a
    searcher that crashes is not, and the crash used to tank every combat
    step of a live run via the LiveSearch fallback to END_TURN.
    """
    if not potion_id:
        return None
    try:
        return create_potion(potion_id, slot=slot)
    except KeyError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "No potion named %r in this build; dropping it from the "
            "situation clone. Update sts2_env/potions/all.py to support "
            "the new potion; until then the search clone has one fewer "
            "potion slot than the live game, which is wrong but playable.",
            potion_id,
        )
        return None


def _parse_deck_entry(entry: Any) -> CardRef:
    """Build a CardRef from one element of the bridge state's `deck` list.

    The current mod sends bare strings (e.g. "STRIKE_IRONCLAD"); the upgraded
    flag is lost, which is the gap the Phase 1.1 mod patch closes. The target
    format is a dict with `id` and `upgraded`. Both are accepted here so the
    Python side is correct before and after the mod changes.
    """
    if isinstance(entry, dict):
        return CardRef(
            card_id=str(entry["id"]),
            upgraded=bool(entry.get("upgraded", False)),
        )
    return CardRef(card_id=str(entry), upgraded=False)


def _room_type_name(room_type: Any) -> str:
    if isinstance(room_type, RoomType):
        return room_type.name
    return str(room_type or "MONSTER").upper()


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

def save_situations(situations: Iterable[CombatSituation], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "situations": [s.to_dict() for s in situations],
    }
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return path


def load_situations(path: str | Path) -> list[CombatSituation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CombatSituation.from_dict(d) for d in data["situations"]]
