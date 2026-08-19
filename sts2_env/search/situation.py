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
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from sts2_env.cards.base import CardInstance, reset_instance_counter
from sts2_env.cards.factory import create_card
from sts2_env.core.combat import CombatState
from sts2_env.core.creature import Creature
from sts2_env.core.enums import CardId, IntentType, PowerId, RoomType
from sts2_env.core.rng import Rng
from sts2_env.monsters.factory import create_monster_by_id
from sts2_env.monsters.intents import Intent
from sts2_env.search.parity import check_max_hp, report_disparity
from sts2_env.potions.base import create_potion
from sts2_env.powers.base import PowerInstance
from sts2_env.run.rooms import create_room
from sts2_env.run.run_state import RunState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encounter registry
# ---------------------------------------------------------------------------

_ENCOUNTER_MODULES = (
    "sts2_env.encounters.act1",
    "sts2_env.encounters.act2",
    "sts2_env.encounters.act3",
    "sts2_env.encounters.act4",
    # The event fights, which are ordinary Monster rooms reached through an
    # event. Omitted here until it was noticed that the registry is built by
    # import and this module was simply not on the list -- so eight encounters
    # that were fully written could not be resolved by name, and the live game
    # sent two of them 104 times. The same omission in `monsters/factory.py`
    # had been hiding 16 monsters. Adding a module file is not enough; it has
    # to be named here.
    "sts2_env.encounters.events",
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


def _encounter_name_candidates(name: str) -> list[str]:
    """Every setup name `name` could reasonably mean, best guess first.

    A LIST RATHER THAN ONE MORE REGEX, because the wire form and the registry
    form disagree in more than one way at once and each new mismatch was being
    answered by making a single transform cleverer.

    The two that were live: the game sends `DENSE_VEGETATION_EVENT_ENCOUNTER`
    and `BATTLEWORN_DUMMY_EVENT_V1_ENCOUNTER` -- the C# class names
    `DenseVegetationEventEncounter` and `BattlewornDummyEventV1Encounter` -- and
    the registry keys them `setup_dense_vegetation` and
    `setup_battleworn_dummy_v1`. Both encounters were fully modelled the whole
    time; only the name failed to land, which is the FLAME_BARRIER /
    FLAME_BARRIER_CARD failure again, and that one cost 68 unplayable cards.

    Suffixes are stripped, never added: dropping `_encounter` from a name that
    does not end in it, or `_event` from an encounter genuinely called that,
    would fabricate a match rather than find one. Each candidate is only ever
    returned if the registry actually holds it.
    """
    base = _setup_name_for_encounter_id(name)
    candidates = [base]
    trimmed = base
    if trimmed.endswith("_encounter"):
        trimmed = trimmed[: -len("_encounter")]
        candidates.append(trimmed)
    # `_event` sits in the MIDDLE of the versioned names
    # (`battleworn_dummy_event_v1`), so it cannot be handled as a suffix.
    if "_event" in trimmed:
        candidates.append(trimmed.replace("_event", "", 1))
    return candidates


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
    for candidate in _encounter_name_candidates(name):
        if candidate in registry:
            return registry[candidate]
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
        # Resolved, not looked up directly: the bridge sends FLAME_BARRIER and
        # this build spells it FLAME_BARRIER_CARD, one of 68 such members. A raw
        # CardId[...] here dropped every one of them out of the DECK the search
        # plans with. See `resolve_card_id`.
        cid = resolve_card_id(self.card_id)
        if cid is None:
            raise KeyError(
                f"No card named {self.card_id!r} in this build. Regenerate the "
                f"fixture against the current game build."
            )
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
    #: Enemy max HP as it actually was, in slot order -- reported by the bridge
    #: for a live fight, read off the RunManager's combat for a harvested one.
    #: Empty only for a fixture written before this field existed, which falls
    #: back to rolling from `encounter_seed`.
    #:
    #: This exists because monster HP *cannot* be reconstructed from the
    #: encounter seed. `CombatState.cs:499` rolls it from
    #: `RunState.Rng.Niche` -- a run-level stream whose position depends on
    #: everything that happened earlier in the run -- not from the encounter's
    #: own RNG. No amount of generator parity recovers it; the only way to know
    #: a live fight's enemy HP is that the game says so.
    enemy_max_hp: tuple[int, ...] = ()

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

        deck = _instantiate_deck(self.deck)
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

        room = create_room(_room_type_for_combat(self.room_type))
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

        # If the game told us the enemies' HP, use it. The encounter setup just
        # rolled its own from `encounter_seed`, and for a live fight that roll
        # is unrecoverable by construction (see `enemy_max_hp`). Overwriting it
        # here rather than only in `to_combat_mid_fight` means a bridge-built
        # situation cannot quietly produce a fight with the wrong enemy HP --
        # which it did, by 1-2 HP per enemy, for every fight in the first live
        # capture taken.
        for index, reported in enumerate(self.enemy_max_hp):
            if index >= len(combat.enemies):
                break
            enemy = combat.enemies[index]
            enemy.max_hp = int(reported)
            enemy.current_hp = int(reported)

        combat.start_combat()
        return combat

    def to_combat_mid_fight(self, bridge_state: dict[str, Any]) -> CombatState:
        """Build a CombatState that matches the bridge's report of *now*.

        The bridge is ground truth -- whatever HP, block, energy, powers, hand
        and enemy state it reports is what the live game has, and the local
        sim must agree. ``to_combat`` builds a fresh fight from the
        situation's seed/encounter/deck, which matches the opening state but
        diverges within a few turns (different shuffle, different enemy
        intent rolls, relic trigger order). This method takes that fresh
        build and **overwrites** the mutable state with the bridge's report,
        so the SearchAgent plans against the position the player is actually
        in rather than a frozen fiction that drifted two turns ago.

        What is overwritten:

        * Player HP, block, energy, max_energy -- direct assignments.
        * Player powers -- the bridge's list replaces the player's powers
          dict; amounts are set verbatim, no hook re-firing.
        * Hand -- rebuilt from the bridge's hand list as fresh CardInstance
          objects (the simulator's own draw is discarded). The draw and
          discard piles are left as ``to_combat`` built them -- the search
          does not look at piles beyond their counts, and the bridge sends
          only counts, so we cannot do better without more protocol.
        * Each enemy's HP, block, powers -- direct assignments, same as the
          player. The enemy's monster id and the encounter setup are already
          correct from ``to_combat``.
        * Enemy intent -- the bridge sends the live game's next-move intent.
          When the move id it names exists in the simulator's state machine,
          we install the bridge's intent onto that ``MoveState`` and re-point
          the AI at it, so the SearchAgent's _incoming_damage reads the right
          telegraph *and* the move keeps its follow-up chain. When the move
          id is unknown (a parity gap), the override is skipped rather than
          synthesising a follow-up-less state that would crash ``roll_move``.

        What is NOT overwritten (and why):

        * Draw pile order -- the bridge sends only ``draw_pile_count``. We
          leave whatever ``to_combat`` rolled. The search's lookahead uses
          the simulator's draw, which is an approximation; that is the same
          approximation the offline benchmark measured at 20% boss win rate
          per ``MODELS.md:120``.
        * Relics, potions, character -- set at combat_start, do not change
          mid-fight.
        * Encounter / encounter_seed / combat_seed -- already baked in.
        """
        combat = self.to_combat()
        _sync_combat_from_bridge(combat, bridge_state)
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

        # Record the enemies this run actually rolled, for the same reason the
        # bridge path does: monster HP comes from the run-level Niche stream
        # (`CombatState.cs:496`), whose position depends on everything earlier
        # in the run, so `encounter_seed` cannot reproduce it. Without this a
        # fixture drifts away from the fight it was harvested from the moment
        # anything about the RNG changes -- which is exactly what happened when
        # the generator was corrected on 2026-08-06.
        live_combat = getattr(mgr, "_combat", None)
        enemy_max_hp = tuple(
            int(enemy.max_hp) for enemy in getattr(live_combat, "enemies", ()) or ()
        )

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
            enemy_max_hp=enemy_max_hp,
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

        # Enemy max HP, straight from the game. Not reconstructable: the game
        # rolls it from the run-level Niche stream (`CombatState.cs:499`), so
        # the encounter seed cannot produce it no matter how faithful the
        # generator is. Read rather than re-derived -- if the game says 43, it
        # is 43.
        enemy_max_hp = tuple(
            int(enemy.get("max_hp", 0) or 0)
            for enemy in (combat.get("enemies") or state.get("enemies") or [])
            if isinstance(enemy, dict) and enemy.get("max_hp")
        )

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
            enemy_max_hp=enemy_max_hp,
        )


#: The bridge sends the game's MapPointType, not a room type, and the two enums
#: are not the same set:
#:
#:   MapPointType   Unassigned Unknown Shop Treasure RestSite Monster Elite Boss Ancient
#:   RoomType       MONSTER ELITE BOSS SHOP REST_SITE TREASURE EVENT
#:
#: Monster/Elite/Boss/Shop/Treasure line up once upper-cased, and everything else
#: does not. `RoomType[name]` therefore raised KeyError on a `?` node -- and a
#: raise here is not a degraded search, it is NO search: the runner catches it and
#: hands the whole fight to the trained combat model. Measured at 4.2% of live
#: combats (9 of 215), every one of them played by the weaker agent.
_MAP_POINT_TO_ROOM = {
    "MONSTER": RoomType.MONSTER,
    "ELITE": RoomType.ELITE,
    "BOSS": RoomType.BOSS,
    "SHOP": RoomType.SHOP,
    "TREASURE": RoomType.TREASURE,
    "RESTSITE": RoomType.REST_SITE,   # RoomType spells it REST_SITE
    "REST_SITE": RoomType.REST_SITE,
    "EVENT": RoomType.EVENT,
    # A `?` that turned into a fight IS a monster fight -- the node type is what
    # the map showed before it resolved, not what the room became.
    "UNKNOWN": RoomType.MONSTER,
    "UNASSIGNED": RoomType.MONSTER,
    "ANCIENT": RoomType.EVENT,        # Neow
}


def _room_type_for_combat(name: str) -> RoomType:
    """The room to build a combat in, from whatever the bridge called it.

    NEVER RAISES. The room only selects room-scoped modifiers; being wrong about
    it costs a little accuracy, while raising costs the entire searcher for that
    fight. An unmapped name is reported rather than swallowed, so a new one shows
    up as a parity gap instead of as quietly worse play.
    """
    key = str(name or "").upper()
    room = _MAP_POINT_TO_ROOM.get(key)
    if room is not None:
        return room
    report_disparity("room_type", key, "unmapped", "MONSTER")
    return RoomType.MONSTER


def resolve_card_id(name: str) -> CardId | None:
    """A bridge card id as a CardId, or None if this build really lacks it.

    THE SUFFIX. 68 of 600 CardId members are spelled `X_CARD` while the bridge
    sends `X`, so a raw `CardId[name]` lookup misses every one of them --
    Barricade, Corruption, Colossus, Blur, Buffer, Afterimage, Biased Cognition
    among them. The bridge-hand path then "dropped" the card as unknown, which
    means the searcher could not see it and therefore could not play it.

    Observed live: FLAME_BARRIER dropped 45 times in one session, while the card
    sat in hand. The run died holding a card the agent was structurally unable
    to play.

    `reference_static_metadata.card_id_for_reference_class` has known this since
    it was written; it just was not on this path. Same alias set, plus the
    trailing `+` the bridge uses for upgraded cards.
    """
    raw = str(name or "").strip()
    if not raw:
        return None
    base = raw.rstrip("+").upper()
    for alias in (base, f"{base}_CARD", f"{base}_STATUS"):
        if alias in CardId.__members__:
            return CardId[alias]
    return None


def _instantiate_deck(refs) -> list[CardInstance]:
    """Build the deck, dropping cards this build does not have.

    Same contract the potions already use, and for the same reason: a searcher
    that clones a fight missing one card is still useful, a searcher that raises
    is not. `CardRef.instantiate` raises by design -- correct for a FIXTURE,
    where a stale card name should stop the harness -- but on the live path a
    raise inside `to_combat` is caught by the runner and hands the whole fight
    to the trained model.

    Worse than a single fight: the deck persists, so one unknown card would take
    the searcher out of EVERY remaining fight of that run. This was the same
    shape as the `RoomType['UNKNOWN']` KeyError that was costing 4.2% of live
    combats, still armed, and had simply not been hit yet.
    """
    deck: list[CardInstance] = []
    for ref in refs:
        try:
            deck.append(ref.instantiate())
        except KeyError:
            report_disparity("deck_card", str(ref.card_id), "unknown", "dropped")
    return deck


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


# Powers worth coercing between the bridge's UPPER_SNAKE ids and the
# simulator's PowerId enum. Built from the enum rather than hand-listed, so a
# new PowerId lands here automatically; any power the bridge sends that is
# not in the enum is dropped with a warning (the same drop-and-continue
# pattern _instantiate_potion uses, so a single unmodelled power does not
# tank the live fight).
_POWER_ID_BY_NAME: dict[str, PowerId] = {p.name: p for p in PowerId}


def _coerce_power_id(name: str) -> PowerId | None:
    """Resolve a bridge power id to a PowerId enum member, or None.

    The bridge sends ``power.Id.Entry`` which is ``Slugify(ClassName)`` --
    so ``StrengthPower`` -> ``STRENGTH_POWER``. The simulator's ``PowerId``
    enum members are UPPER_SNAKE of the class name; the mod's slug matches
    on most powers (e.g. ``VULNERABLE``, ``WEAK``, ``STRENGTH``) because
    the slugify strips the ``Power`` suffix sometimes and leaves it other
    times. Try (a) the raw name, (b) with a trailing ``_POWER`` removed,
    (c) with a trailing ``_POWER`` added.
    """
    if name in _POWER_ID_BY_NAME:
        return _POWER_ID_BY_NAME[name]
    if name.endswith("_POWER"):
        stripped = name[:-len("_POWER")]
        if stripped in _POWER_ID_BY_NAME:
            return _POWER_ID_BY_NAME[stripped]
    with_suffix = f"{name}_POWER" if not name.endswith("POWER") else name
    if with_suffix in _POWER_ID_BY_NAME:
        return _POWER_ID_BY_NAME[with_suffix]
    return None


def _bridge_power_instance(pid: PowerId, amount: int) -> PowerInstance:
    """Build a bridge-reported power as its registered class, not a bare one.

    A bare ``PowerInstance`` has no hooks, and the search clone carries only
    what this sync installs. Waterfall Giant banks its whole fight into
    ``STEAM_ERUPTION`` and detonates it from ``SteamEruptionPower.after_death``
    -- so with a bare instance the clone let the giant die quietly, every kill
    line scored a clean win, and the live agent nuked itself at 37 HP against
    a 37-damage eruption on 2026-08-16. Constructing the registered class keeps
    the amount verbatim -- nothing re-fires -- while the power's hooks work
    again in every lookahead the search runs from here.
    """
    from sts2_env.core.creature import get_power_class

    cls = get_power_class(pid)
    if cls is not None:
        try:
            return cls(amount)
        except TypeError:
            logger.warning(
                "bridge powers: class for %r does not take a single amount; "
                "falling back to a bare instance.", pid,
            )
    return PowerInstance(power_id=pid, amount=amount)


def _sync_combat_from_bridge(combat: CombatState, state: dict[str, Any]) -> None:
    """Overwrite mutable combat state with the bridge's ground-truth report.

    See ``CombatSituation.to_combat_mid_fight`` for the contract: the bridge
    is the authority on the current position; the local sim matches it, the
    search plans against the match, and any approximation (the draw pile
    order, the enemy's next-next move) is what the search's horizon already
    tolerated.
    """
    crash = state.get("combat_state") or state
    if not isinstance(crash, dict):
        return
    player_json = crash.get("player") or {}
    if not isinstance(player_json, dict):
        return

    # WHICH TURN IT IS. The mod has always sent this and the mid-fight rebuild
    # has always dropped it, so every reconstructed position claimed to be
    # round 1 no matter how long the fight had run. The simulator branches on
    # `round_number` in several places, and `SearchAgent` uses it to know that a
    # planned line belongs to the turn it was planned for -- which it cannot do
    # if the number never moves.
    round_number = state.get("round", crash.get("round"))
    if isinstance(round_number, int) and round_number > 0:
        combat.round_number = round_number
        combat.turn_count = max(combat.turn_count, round_number - 1)

    # -- player HP / block / energy / powers / hand -------------------------
    pcard = combat.primary_player
    if "hp" in player_json:
        pcard.current_hp = int(player_json["hp"])
    if "block" in player_json:
        pcard.block = int(player_json.get("block", 0))
    if "energy" in player_json:
        combat.current_player_state.energy = int(player_json["energy"])
    if "max_energy" in player_json:
        combat.current_player_state.base_max_energy = int(
            player_json.get("max_energy", 3)
        )

    # Player powers: full reset, then apply the bridge's list verbatim.
    # The bridge's report is the live game's authoritative state; the
    # simulator's own power application (which would fire hooks and modify
    # amounts) is bypassed -- we just set the dict, because the goal is to
    # mirror what's on screen, not to re-derive it.
    pcard.powers.clear()
    for p in (player_json.get("powers") or []):
        pid = _coerce_power_id(str(p.get("id", "")).upper())
        if pid is None:
            logger.warning(
                "bridge powers: %r not in this build; dropping. The "
                "search clone has one fewer player power than the live "
                "game, which is wrong but playable.",
                p.get("id"),
            )
            continue
        amount = int(p.get("amount", 0))
        pcard.powers[pid] = _bridge_power_instance(pid, amount)

    # Hand: clear and rebuild from the bridge's list. Cards come back as
    # fresh CardInstance objects via the factory; their internal counters do
    # not matter for the search's plan (which only looks at damage/block/cost/
    # type/target).
    hand_json = crash.get("hand") or []
    if isinstance(hand_json, list):
        new_hand = []
        for card_json in hand_json:
            if not isinstance(card_json, dict):
                continue
            cid_str = card_json.get("id")
            if not cid_str:
                continue
            cid = resolve_card_id(cid_str)
            if cid is None:
                report_disparity("hand_card", str(cid_str), "unknown", "dropped")
                continue
            upgraded = bool(card_json.get("upgraded", False))
            instance = create_card(cid, upgraded=upgraded)
            # Carry the game's own playability verdict. It accounts for every
            # rule the game has -- energy, curses, statuses, RINGING's one-card
            # limit, relics this simulator may not model -- and `can_play_card`
            # treats it as final. Absent means "no opinion", not "unplayable",
            # so a mod that does not send the field changes nothing.
            if "playable" in card_json:
                instance.bridge_playable = bool(card_json.get("playable"))
            new_hand.append(instance)
        combat.current_player_state.hand[:] = new_hand

    # -- enemies: HP, block, powers, intent ---------------------------------
    enemies_json = crash.get("enemies") or []
    if isinstance(enemies_json, list):
        # Match by monster id, not by position. The game drops a dead enemy
        # from its list; `to_combat` always rebuilds the full opening roster.
        # Zipping those by index silently pairs the wrong monsters -- observed
        # live on SLIMES_WEAK, where the sim's 3-slime roster met a 1-slime
        # report and LEAF_SLIME_M's HP was written onto LEAF_SLIME_S, leaving
        # two untouched full-HP phantoms the search could target. The runner
        # then sent target_index 1 at an enemy that was not on screen, the game
        # ignored it, and the same state came back until the stuck-detector
        # stopped the session.
        matched: dict[int, dict] = {}
        bridge_slot_of: dict[int, int] = {}
        unclaimed = list(range(len(combat.enemies)))

        # PASS 1 -- exact id matches only.
        #
        # Separated from the fallback deliberately. These used to run together
        # in one loop, so a bridge enemy the sim had no slot for would take the
        # first free slot BEFORE the monster that actually belonged there was
        # considered, and that monster was then dropped for having no slot left.
        # Fogmog is the case: the bridge reports ["EYE_WITH_TEETH", "FOGMOG"],
        # the Eye claimed Fogmog's slot, and Fogmog vanished.
        leftovers: list[tuple[int, dict]] = []
        for bridge_slot, enemy_json in enumerate(enemies_json):
            if not isinstance(enemy_json, dict):
                continue
            wanted = str(enemy_json.get("id", "")).upper()
            slot = next(
                (
                    i for i in unclaimed
                    if str(combat.enemies[i].monster_id).upper() == wanted
                ),
                None,
            )
            if slot is None:
                leftovers.append((bridge_slot, enemy_json))
                continue
            unclaimed.remove(slot)
            matched[slot] = enemy_json
            # The enemy's ACTUAL position in the bridge's list, captured here
            # rather than re-derived below. See the note on bridge_enemy_index.
            bridge_slot_of[slot] = bridge_slot

        # PASS 2 -- monsters the opening roster never contained.
        #
        # An encounter's roster is not the set of monsters the fight holds:
        # Fogmog summons an Eye, and anything carrying MinionPower is resummoned
        # after it dies. `to_combat` rebuilds the OPENING roster, so a summon has
        # no slot and previously had to steal one. Build it instead -- the
        # simulator already knows how, and a materialised Eye is a target the
        # search can reason about rather than a phantom it mistakes for its
        # summoner.
        for bridge_slot, enemy_json in leftovers:
            wanted = str(enemy_json.get("id", "")).upper()
            built = create_monster_by_id(wanted, combat.niche_rng) if wanted else None
            if built is not None:
                creature, ai = built
                combat.add_enemy(creature, ai)
                slot = len(combat.enemies) - 1
            elif unclaimed:
                # Unknown to this simulator. Keep the old behaviour rather than
                # losing the monster: playing it in the wrong slot beats not
                # seeing it at all, and the id is logged as the parity gap it is.
                logger.warning(
                    "bridge reported monster %r that this simulator cannot "
                    "build; reusing slot %d. This is a parity gap.",
                    wanted, unclaimed[0])
                slot = unclaimed.pop(0)
            else:
                logger.warning(
                    "bridge reported monster %r with no slot to put it in; "
                    "the search will not see it.", wanted)
                continue
            matched[slot] = enemy_json
            bridge_slot_of[slot] = bridge_slot

        # Anything the bridge did not report is dead. Left alive it is a
        # phantom: a target the search can pick and the game will refuse.
        for i in unclaimed:
            combat.enemies[i].current_hp = 0

        # The live game compacts its enemy list as monsters die, so the sim's
        # slot for a survivor is not the index the game expects in a PLAY.
        # Record the translation; `LiveSearch` applies it to the action before
        # it goes on the wire. Without it the fix above merely moves the bug:
        # the search stops targeting phantoms and starts naming a live index
        # that points at the wrong monster, or at nothing.
        #
        # TAKEN FROM THE MATCH, NOT RE-ENUMERATED. This was
        # `enumerate(sorted(matched.items()))`, which hands out bridge slots
        # 0,1,2... in sim-roster order and is therefore only right when the
        # bridge lists enemies in that same order. Fogmog is the case where it
        # is not: `FogmogNormal.Slots` is ["illusion", "fogmog"], so the Eye it
        # summons occupies the EARLIER display slot and the bridge reports
        # ["EYE_WITH_TEETH", "FOGMOG"] while the sim roster is [FOGMOG, EYE].
        # The old mapping sent sim slot 0 (Fogmog) to bridge slot 0 (the Eye),
        # so every attack the search aimed at Fogmog landed on the Eye instead.
        #
        # That is not a small mis-aim. EyeWithTeeth carries IllusionPower: on
        # death it heals to full and is never removed from combat, so it is
        # effectively immortal, and damage spent on it is simply deleted.
        # Measured over 29 live Fogmog fights: 7 of 7 in which Fogmog was never
        # hit ended in death, against 9 of 22 when it was, and the survivors put
        # 56% of their attacks into Fogmog against the dead runs' 24%. Fogmog
        # was the deadliest normal encounter in act 1 by a factor of three
        # (median 30 damage taken, next worst 24, 62% death rate) largely
        # because half the agent's output was going into an illusion.
        combat.bridge_enemy_index = dict(bridge_slot_of)

        for i, enemy_json in matched.items():
            enemy = combat.enemies[i]
            if "hp" in enemy_json:
                enemy.current_hp = int(enemy_json["hp"])
            if "max_hp" in enemy_json:
                # Checked BEFORE the overwrite, which is the only moment the two
                # values coexist. Current HP is deliberately not checked -- it
                # differs on every call because the fight has progressed, which
                # is not a disparity.
                check_max_hp(str(enemy.monster_id or ""), int(enemy_json["max_hp"]))
                enemy.max_hp = int(enemy_json["max_hp"])
            if "block" in enemy_json:
                enemy.block = int(enemy_json.get("block", 0))
            if "is_alive" in enemy_json:
                # The simulator's Creature has no is_alive setter -- alive
                # is current_hp > 0. Set HP to 0 to mark dead if the bridge
                # says so; otherwise trust the HP we just set.
                if not enemy_json["is_alive"] and enemy.current_hp > 0:
                    enemy.current_hp = 0

            # Enemy powers: same reset-and-replace as the player.
            enemy.powers.clear()
            for p in (enemy_json.get("powers") or []):
                pid = _coerce_power_id(str(p.get("id", "")).upper())
                if pid is None:
                    # A power this simulator has no id for is dropped, so the
                    # search plans as though the enemy does not have it. Silent
                    # until now.
                    report_disparity(
                        "unknown_power", str(enemy.monster_id or "?"),
                        "not modelled", str(p.get("id", "")).upper())
                    continue
                amount = int(p.get("amount", 0))
                enemy.powers[pid] = _bridge_power_instance(pid, amount)

            # Intent: the bridge sends the next move's intent/damage/hits.
            # Install it onto the simulator's own MoveState so the
            # SearchAgent's _incoming_damage reads the right telegraphed hit
            # while the move keeps its follow-up chain. See
            # _override_enemy_intent for the unknown-move-id case.
            intent_str = enemy_json.get("intent")
            if intent_str:
                _override_enemy_intent(combat, enemy, enemy_json)

    # -- round number -------------------------------------------------------
    if "round" in crash:
        combat.round_number = int(crash["round"])
        combat.turn_count = max(combat.turn_count, combat.round_number - 1)


#: Powers that change what an attack lands for, so a telegraphed number stops
#: being comparable with a base one. On the attacker: Strength and Weak. On the
#: defender: Vulnerable. Anything here on either side suspends the intent parity
#: check for that decision.
#: `MonsterModel.stunnedMoveId`. Not a per-monster move -- the game synthesises
#: this state on any creature it stuns, so it appears under every monster id and
#: belongs to none of their state machines.
_BRIDGE_STUNNED_MOVE_ID = "STUNNED"

#: IllusionPower's revive, built in `after_death` and therefore absent from any
#: monster the simulator has not yet killed. Eye With Teeth and Parafright.
_BRIDGE_REVIVE_MOVE_ID = "REVIVE_MOVE"


def _install_revive_state(combat: CombatState, enemy: Creature, ai) -> bool:
    """Put the enemy into IllusionPower's revive turn: heal, do nothing else.

    Mirrors what `IllusionPower.after_death` builds -- a HEAL intent, must
    perform once, following up into whatever the monster was doing -- without
    requiring the creature to die on this side first.
    """
    from sts2_env.core.enums import IntentType
    from sts2_env.monsters.state_machine import MoveState

    power = enemy.powers.get(PowerId.ILLUSION)
    follow_up = ai.state_log[-1] if getattr(ai, "state_log", None) else ai.current_move.state_id
    if follow_up == _BRIDGE_REVIVE_MOVE_ID:
        return False

    def _revive(_: CombatState) -> None:
        if power is not None and hasattr(power, "revive"):
            power.revive(enemy)
        else:
            enemy.current_hp = enemy.max_hp

    ai.states[_BRIDGE_REVIVE_MOVE_ID] = MoveState(
        _BRIDGE_REVIVE_MOVE_ID,
        _revive,
        [Intent(IntentType.HEAL)],
        follow_up_id=follow_up,
        must_perform_once=True,
    )
    ai._current_state_id = _BRIDGE_REVIVE_MOVE_ID  # noqa: SLF001
    return True

_ATTACKER_DAMAGE_MODIFIERS = (
    PowerId.STRENGTH,
    PowerId.WEAK,
    # Vigor was the last big false positive: Terror Eel's Thrash applies
    # `Vigor 6` to itself, so its Crash telegraphs 16 + 6 = 22 and the check
    # reported that 43 times in one session against a base of 16 that is
    # correct. Doubling is here for the same reason -- any of these means the
    # telegraphed number and the base number are different quantities.
    PowerId.VIGOR,
    PowerId.DOUBLE_DAMAGE,
    # Shrink carries a "DamageDecrease" and is a Debuff, so a shrunk
    # attacker telegraphs less than its base. Shrinker Beetle reported
    # sim 7 / game 4 and sim 13 / game 9 -- both exactly (base - 1) * 0.75,
    # its own Shrink plus Weak -- against constants that match the
    # decompile precisely. Two false positives, not two bugs.
    PowerId.SHRINK,
)
#: Powers on the DEFENDER that change what a telegraphed hit lands for. Missing
#: these turned a session's report into 23 "disparities" that were all correct
#: modelling -- every halved row had the enemy Vulnerable and the player holding
#: Colossus, which is precisely ColossusPower's condition:
#:
#:     if (!dealer.HasPower<VulnerablePower>()) return 1m;
#:     return DynamicVars["DamageDecrease"].BaseValue;   // 0.5
#:
#: The simulator does that correctly. `CEREMONIAL_BEAST.STOMP sim=15 game=7`
#: was the report not knowing why, on a constant that matches the decompile
#: exactly. Guarded and Tank halve and raise the same way.
_DEFENDER_DAMAGE_MODIFIERS = (
    PowerId.VULNERABLE,
    PowerId.COLOSSUS,
    PowerId.GUARDED,
    PowerId.TANK,
)

#: Attacker-side powers whose effect is monster-specific rather than a general
#: modifier. They showed up in the same report as clean arithmetic --
#: VITAL_SPARK2 was exactly +2, CRAB_RAGE exactly x1.5 -- so a telegraph
#: carrying them is not evidence of a wrong constant.
_ATTACKER_SPECIFIC_MODIFIERS = tuple(
    p for p in (
        getattr(PowerId, name, None) for name in (
            "VITAL_SPARK", "CRAB_RAGE", "BACK_ATTACK_LEFT", "BACK_ATTACK_RIGHT",
            "TERRITORIAL", "PERSONAL_HIVE", "BURROWED", "STEAM_ERUPTION",
        )
    ) if p is not None
)


def _damage_is_modified(combat: CombatState, enemy: Creature) -> bool:
    """Is anything scaling this enemy's attack away from its base value?"""
    for power_id in _ATTACKER_DAMAGE_MODIFIERS + _ATTACKER_SPECIFIC_MODIFIERS:
        if enemy.get_power_amount(power_id):
            return True
    player = getattr(combat, "player", None)
    if player is not None:
        for power_id in _DEFENDER_DAMAGE_MODIFIERS:
            if player.get_power_amount(power_id):
                return True
    return False


def _override_enemy_intent(
    combat: CombatState, enemy: Creature, enemy_json: dict[str, Any],
) -> None:
    """Replace the enemy's current MoveState with one built from the bridge.

    The intent override has one job: make ``ai.current_move.intents`` return
    the bridge's telegraphed intent so the SearchAgent's _incoming_damage
    reads the right incoming hit. The MoveState the bridge reports is the
    live game's actual next move; if the simulator's state machine has that
    state id, install the bridge's intent into it (replacing whatever intents
    the simulator's encounter setup built) and re-point the AI at it -- so
    the move has a follow-up, and the search's cloned enemy turns can
    progress past the bridge-reported move.

    If the bridge sends an intent_move_id the simulator does not have (a
    move name the encounter setup did not register, or the simulator's
    encounter has been refactored), we cannot install the override safely
    -- the synthetic would have no follow_up_id and roll_move would raise.
    In that case skip the intent override entirely, leave the AI as it was,
    and the search falls back to the simulator's telegraphed intent. That is
    wrong but not crashing -- the search will see the simulator's intent
    rather than the bridge's, and in most cases the two agree on attack vs.
    defend vs. buff; the damage/hits may differ, which only matters when
    the enemy is *about* to hit, which is also when the search sees the
    telegraphed intent the simulator built.
    """
    intent_str = str(enemy_json.get("intent", "UNKNOWN")).upper()
    intent_type = _INTENT_NAME_TO_ENUM.get(intent_str)
    if intent_type is None:
        return  # unknown intent; leave the AI's current move as-is.

    ai = combat.enemy_ais.get(enemy.combat_id)
    move_id = enemy_json.get("intent_move_id")
    if ai is None or not move_id:
        return  # Nothing to override; leave the simulator's intent.

    damage = int(enemy_json.get("intent_damage", 0) or 0)
    hits = int(enemy_json.get("intent_hits", 1) or 1)
    # pre_modified: the game telegraphs the FINAL number, Strength and the
    # player's Vulnerable already in it. `_incoming_damage` must not apply
    # them again on top.
    intent = Intent(intent_type=intent_type, damage=damage, hits=hits,
                    pre_modified=True)

    # Prefer the real MoveState in the AI's state dict -- it has a
    # follow_up_id and the simulator's full effect; we override only its
    # intents (the bridge's telegraphed hit is more current than whatever
    # the encounter setup built).
    # STUNNED is not in any monster's state machine, and never will be: the game
    # BUILDS it when a creature is stunned --
    #
    #     MoveState state = new MoveState("STUNNED", stunMove, new StunIntent())
    #     { FollowUpStateId = nextMoveId, MustPerformOnceBeforeTransitioning = true };
    #     Monster.SetMoveImmediate(state);          -- Creature.StunInternal
    #
    # so a lookup against `ai.states` was always going to miss, on every monster,
    # forever. It was the single biggest unknown_move cluster in the live logs --
    # Corpse Slug, Terror Eel, Tunneler, Bowlbug Rock, Ceremonial Beast and
    # Lagavulin Matriarch, the last two being act 1 bosses.
    #
    # The cost of missing it is not a wrong number, it is a wrong TURN. A stunned
    # monster does nothing; the search was instead rolling its whole lookahead on
    # whatever move the simulator thought was next, so it planned around a hit
    # that was not coming and spent the free turn defending.
    #
    # `stun_enemy` already does precisely what StunInternal does, including the
    # follow-up back to the previous move and must_perform_once. It simply had no
    # caller on the live path.
    if str(move_id) == _BRIDGE_STUNNED_MOVE_ID and str(move_id) not in ai.states:
        if combat.stun_enemy(enemy):
            return
        report_disparity(
            "stun_failed", str(getattr(enemy, "monster_id", "?")),
            "could not apply", move_id)
        return

    # REVIVE_MOVE is the same shape of problem as STUNNED. IllusionPower builds
    # it inside `after_death` -- so the simulator only owns the state once the
    # creature has actually died in the simulator, and the live path rebuilds a
    # fresh, undamaged monster on every decision. The Eye With Teeth that has
    # already died and come back in the real game therefore reports a move this
    # side has never constructed.
    #
    # Synthesised rather than skipped, for the same reason: an unmatched move id
    # makes the override bail, and the search then rolls its lookahead on
    # whatever the Eye was doing before it died -- Distract, three Dazed a turn --
    # when what it is actually doing is spending the turn healing and dealing
    # nothing at all. Free turns misread as threatened ones, 14 times in 13 runs.
    if str(move_id) == _BRIDGE_REVIVE_MOVE_ID and str(move_id) not in ai.states:
        if _install_revive_state(combat, enemy, ai):
            return
        report_disparity(
            "revive_failed", str(getattr(enemy, "monster_id", "?")),
            "could not apply", move_id)
        return

    existing = ai.states.get(str(move_id))
    if existing is not None and hasattr(existing, "intents"):
        # The simulator's own telegraph for this exact move, before it is
        # replaced. This is the check that matters most for the search: the
        # bridge corrects the CURRENT turn's intent, but every turn the lookahead
        # rolls past it uses the simulator's number. A move modelled at the wrong
        # damage is planned against wrongly for the whole horizon, and nothing
        # about the live run ever says so.
        #
        # ONLY WHEN NOTHING IS MODIFYING THE DAMAGE. The simulator's intent holds
        # BASE damage; the game telegraphs what will actually land, with Strength
        # and Weak and the player's Vulnerable already folded in. Comparing those
        # two directly is not a parity check, it is a Strength detector -- the
        # first version of this reported 120-odd findings and the two biggest
        # were the simulator being exactly right. Phantasmal Gardener's Bite is 5
        # in the decompile and 5 here; the game said 7, 8 and 9 because
        # ENLARGE_MOVE had granted +2 Strength each time. Damp Cultist's Dark
        # Strike really is base 1, and the 6-to-17 spread was its own ramp.
        #
        # So the check is skipped whenever any damage-modifying power is in play.
        # That is conservative -- it gives up on modified turns rather than
        # guessing at the game's modifier order -- and it still sees the opening
        # turn of nearly every fight, which is where a wrong base shows up.
        if not _damage_is_modified(combat, enemy):
            for prior in (existing.intents or []):
                if getattr(prior, "intent_type", None) is not intent_type:
                    continue
                prior_damage = int(getattr(prior, "damage", 0) or 0)
                prior_hits = int(getattr(prior, "hits", 1) or 1)
                if prior_damage != damage:
                    # POWERS IN THE LABEL. Without them an intent_damage report
                    # cannot be acted on: ROCKET.PRECISION_BEAM sim=18 game=27
                    # is a wrong constant if the enemy is clean and correct
                    # arithmetic if it is holding 9 Strength, and the report as
                    # written could not tell the two apart. `_damage_is_modified`
                    # already suppresses the clear-cut cases; these are the ones
                    # that got past it, which is exactly when the powers matter.
                    powers = "/".join(
                        f"{pid.name}{inst.amount}"
                        for pid, inst in (enemy.powers or {}).items()
                    ) or "clean"
                    report_disparity(
                        "intent_damage", f"{enemy.monster_id}.{move_id}[{powers}]",
                        prior_damage, damage)
                if prior_hits != hits:
                    report_disparity(
                        "intent_hits", f"{enemy.monster_id}.{move_id}",
                        prior_hits, hits)
                break

        existing.intents = [intent]
        ai._current_state_id = str(move_id)
        return

    # If the move_id is not in the simulator's state machine, do NOT install
    # a synthetic with no follow-up -- the search's cloned end_player_turn
    # calls roll_move which calls current.get_next_state which would raise
    # "no follow_up_id" and silently turn every search turn into a crash.
    # Leave the AI as-is; the search sees the simulator's telegraph rather
    # than the bridge's, which loses damage precision but keeps the run
    # playable.
    # Reported rather than logged at debug: a move the simulator has no state
    # for is a bigger parity hole than a wrong number, because the search then
    # rolls the whole lookahead on a move the game is not going to make.
    report_disparity(
        "unknown_move", str(getattr(enemy, "monster_id", "?")),
        "no such state", move_id)


# Intent name -> IntentType. The bridge sends the C# enum's ToString(), which
# for the canonical intents is the same uppercase name we use in our IntentType.
_INTENT_NAME_TO_ENUM = {
    "ATTACK": IntentType.ATTACK,
    "MULTI_ATTACK": IntentType.MULTI_ATTACK,
    "MULTIATTACK": IntentType.MULTI_ATTACK,
    "DEFEND": IntentType.DEFEND,
    "BUFF": IntentType.BUFF,
    "DEBUFF": IntentType.DEBUFF,
    "DEBUFF_STRONG": IntentType.DEBUFF_STRONG,
    "SLEEP": IntentType.SLEEP,
    "SUMMON": IntentType.SUMMON,
    "ESCAPE": IntentType.ESCAPE,
    "UNKNOWN": IntentType.UNKNOWN,
    "STATUS_CARD": IntentType.STATUS_CARD,
    "STUN": IntentType.STUN,
    "HEAL": IntentType.HEAL,
    "DEATH_BLOW": IntentType.DEATH_BLOW,
    "CARD_DEBUFF": IntentType.CARD_DEBUFF,
}


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
