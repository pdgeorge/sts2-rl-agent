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


def resolve_encounter(name: str) -> Callable[..., None]:
    registry = encounter_registry()
    setup = registry.get(name)
    if setup is None:
        raise KeyError(
            f"No encounter setup named {name!r}. A fixture written against an "
            f"older build can name an encounter this one has renamed or removed; "
            f"regenerate the fixture rather than editing it by hand."
        )
    return setup


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


def _instantiate_potion(potion_id: str | None, slot: int):
    """Same contract as cards and encounters: a name this build does not have
    fails here, saying so, rather than further in as a missing effect."""
    if not potion_id:
        return None
    try:
        return create_potion(potion_id, slot=slot)
    except KeyError as exc:
        raise KeyError(
            f"No potion named {potion_id!r} in this build. Regenerate the "
            f"fixture against the current game build."
        ) from exc


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
