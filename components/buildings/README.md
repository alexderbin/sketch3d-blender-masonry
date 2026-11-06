# Buildings — procedural architecture generator for Blender: castles, cathedrals, temples, towns and classical orders

Buildings is a procedural architecture generator for Blender, written in Python: a castle generator, a cathedral and
temple generator and a medieval town and village generator, with ziggurats, houses and Greek and Roman orders. It turns
an intent and a site into architecture in two phases. The *plan* (Python) reads the terrain, lays out courts, walls and
towers, naves and vaults, streets and plots, records a reason for every decision and checks the result; the *builder*
(Blender) turns the plan into masses — walls with real openings, vaults, roofs, towers — coloured by a style palette and
tagged with face roles for the steps that follow. The `building` step reads the `site` layer, refines it (platforms,
terraces, moats, graded streets) and writes the layers neighbours use: occupancy, anchors, an influence raster, masonry
surfaces and face attributes.

**Use it to:**

- Generate a procedural castle on a hill, spur, cliff edge or island, with courts, curtain walls, towers, gates, a keep
  and a moat: `building: {type: castle, purpose: border_fortress, culture: french, era: late, seed: 3}`
- Generate a Gothic or Romanesque cathedral (rib vaults, flying buttresses, rose windows), a Byzantine church or a Khmer
  temple: `building: {type: temple, style: gothic_fr, seed: 1}`
- Generate a ziggurat, a step pyramid or a temple-mountain:
  `building: {type: edifice, kind: ziggurat, style: mesopotamian}`
- Generate a medieval town, a Roman colony, a Greek polis or a village with streets, plots and houses on the terrain:
  `building: {type: settlement, archetype: town, seed: 1}`
- Build Greek and Roman temples in the classical orders (Doric, Ionic, Corinthian, Tuscan), a tholos, a stoa or a
  theatre in a slope: `building: {type: classic, kind: peripteros}`
- Plan a building without Blender and read the reasons behind every decision and its checks:
  `.venv/bin/python -m components.buildings.buildings plan castle <site folder> seed=3 out=out/castle_a`

## Features

- **Castles on any terrain.** The site is read by castellology — summit, ridge, spur, hillside, rock, cliff edge, plain,
  shore, peninsula, island, valley — from what lies beyond the platform edge in 16 directions, suitability after
  Dusterwald (2015). The plan places courts, curtain walls with wall walks, flanking towers, gates facing the approach,
  a keep, court buildings, moat, bridge and road, and grades the terrain. Intent: `purpose` (lord's seat, border fortress,
  watchtower, order castle, palace, citadel, pass fort…), `era` (romanesque, gothic, late, artillery, future), `culture`
  (german, french, english, crusader, moorish, kremlin and fantasy cultures), `scale`, `mood`.
- **Unique elements** chosen by each wall section's neighbourhood, with reasons: hoardings, machicolations, zwinger,
  garderobes, oriels, chapel bays, posterns, water gates, danskers, bartizans; staged towers from one grammar.
- **Temples**: basilicas from style genes — 22 families (romanesque, gothic variants, hall church, byzantine, orthodox,
  mesopotamian, mesoamerican, khmer, egyptian, fantasy), mixed as `a+b` or `a:0.6,b:0.4`; a biography of construction
  campaigns with style changes, an unfinished last campaign, a signature element (lantern, crossing dome, great spire,
  campanile, westwork…), additions, ruin. Walls with true openings (polygon tessellation, no booleans), rib and groin
  vaults per bay, flying buttresses, pinnacles after Roriczer, tracery and rose windows.
- **Edifices**: type × style compositions — ziggurat, step pyramid, temple-mountain, spiral tower, occult ring, dark
  citadel, axial temple, domed centrally-planned building and more.
- **Settlements**: eight archetypes as data (Greek polis, Roman colony, Roman suburra, medieval town, a suburb around a
  castle, a town inside castle walls, village, a tiered megastructure city). Streets by grid (Hippodamian, castrum),
  organic (Voronoi wards, slope-aware arteries from the gates) or growth (Parish–Müller-style)
  generators; plots as strips, rows, whole blocks or subdivisions; a house per plot (Greek courtyard house, domus,
  insula, burgage house, log house, hut…) with hip and gable roofs as the lower envelope of the slope planes; density
  checked by Spacematrix (GSI, FSI).
- **Three influences** shape a settlement: power (centres, `P = max s·exp(−(d/r)²)`), trade (flow along shortest street
  paths, by Hillier's space syntax) and wealth (centre–periphery, uphill, windward); they correct façades, plot depth,
  gardens, shops and colours. Knobs `power`, `trade`, `wealth`, `inequality` (0.5 — as computed), `wind`, `centre`.
- **Classical orders**: Doric, Ionic, Corinthian, Tuscan; peripteral, prostyle and in-antis temples, podium temple,
  tholos, stoa, bouleuterion, palaestra, propylon, altar, a theatre in a slope; module — the lower column diameter.
- **Space scheme before the model**: cells (interior, court, exterior, covered, void, underground), transitions (door,
  gate, arch, stair, ramp, bridge, ladder, stair inside the wall), a cell graph after IndoorGML (depth, entrances, choke
  points); the built model is verified against it by rays (floors, vaults, open sky, open doors, closed walls).
- **Reproducible**: one seed — one building to the millimetre; plan and model fingerprints guard every change.

## How it works

`steps.py:plan_building` (Python: numpy, OpenCV, shapely) writes a plan folder: `<type>.json` (layout, `logic` —
decisions with reasons, `checks`, `spaces`, `colors`), the refined `relief.npz` and `location.json`, `plan.png`; the run
scene receives the refined relief and the layers. `build_building` (Blender) rebuilds the ground through Site's api and
calls the type's builder over the open scene (`castles/bl_castle.py` — it also writes the masonry surfaces,
`temples/bl_temple.py`, `edifices/bl_edifice.py`, `classics/bl_classic.py`, `ensembles/bl_ensemble.py`). Faces carry
`Col` (palette), `sk_role` (0 wall, 1 trim, 2 roof, 3 wood, 4 metal, 5 glass, 6 mortar, 7 rubble, 8 floor, 9 opening,
10 water), `sk_space`/`sk_side` (scheme cell); houses also `sk_house` and influence values. The component is a core
(`base/`, `siting/`, `style/`, `space/`, `influences/`), types and an ensemble; `bench/deps.py` enforces that the core
never calls types, a type never calls another type and the ensemble reaches types only through `buildings.py`.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `building` | site → buildings, site (refined relief) | `type` (castle; temple, edifice, settlement, classic), `seed`, `at` ("x,y"); castle: `site`, `purpose`, `era`, `culture`, `scale`, `mood`, `approach`; temple: `style`, `history`, `wealth`, `ruin`, `irregular` (0–1); edifice: `kind`, `style`; settlement: `archetype`, `size`, `host`, `power`, `trade`, `wealth`, `inequality`, `wind`, `centre`; classic: `kind` (`all` — a show). Unset — from the type's knowledge |

Scene files: `occupancy.json`, `anchors.json` (`castle.centre|gate|keep`, `town.centre|gate.N|end.N|square.N`,
`temple.centre`, `edifice.centre`, `classic.centre`), `influence.npz` (settlements), `surfaces.json` (castles), the
refined relief and its ground maps. Gives `building_plan` and `type`.
All keys with descriptions: [component.yaml](/components/buildings/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, culture: german, scale: medium, seed: 7}
  # - building: {type: temple, style: gothic_fr, seed: 1}
  # - building: {type: edifice, kind: ziggurat, style: mesopotamian, seed: 1}
  - camera: {view: wide, azimuth: 200, elevation: 12, distance: 170}   # aims at the building centre anchor
  - light: {time: day}
```

```sh
.venv/bin/python -m core.manager.pipeline components/buildings/bench/stepwise/settlement.yaml   # also temple.yaml, edifice.yaml
# without the manager; a site folder holds relief.npz + location.json
.venv/bin/python -m components.buildings.buildings plan castle <site folder> culture=french era=late seed=3 out=out/castle_a
.venv/bin/python -m components.buildings.buildings build out/castle_a views=aerial,road res=1280x800 blend=1
.venv/bin/python -m components.buildings.buildings passport out/castle_a      # bounds, entrances, anchors, checks
.venv/bin/python -m components.buildings.bench.all [--blender]                # part rules, fingerprints, part benches
```

## Driving it from an AI agent

The agent writes the intent as keys: `building` after `relief` (it reads the site and edits it), `camera` after the
building (it aims at the centre anchor), `light` last:

```yaml
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, purpose: border_fortress, site: spur, culture: french, era: late, mood: grim, seed: 3}
  - camera: {view: wide, azimuth: 200, elevation: 12, distance: 170}
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry`. The plan alone, in Python:
  `.venv/bin/python -m components.buildings.buildings plan castle <site folder> seed=3 out=<folder>` prints a
  `__RESULT__` JSON line with the plan folder `out`, statistics and `non_passing` (names of failed checks). All keys,
  defaults and values: `components/buildings/component.yaml`.
- A run prints `__RESULT__{"pipeline", "out", "failures", …}` as its last line; the run folder holds `journal.md` and,
  with the `frames` output, a contact sheet `contact.jpg`. The step gives `building_plan` (a folder with `castle.json`,
  `temple.json`…: layout, `logic` with reasons, `checks`) and `type`.
- Request to keys: "a grim late-medieval French border fortress on a spur" → the `building` line above; "a ruined
  Gothic cathedral" → `type: temple, style: gothic_fr, ruin: 0.6`; "a rich trading town" →
  `type: settlement, archetype: town, trade: 0.8, wealth: 0.7`; "a Greek city" → `archetype: polis`.

## Files

| Path | Role |
|---|---|
| `steps.py`, `buildings.py`, `layers.py` | the step; entry and type registry (`types`, `plan`, `build`, `make`, `passport`, `influence`); layers |
| `base/`, `siting/`, `style/`, `space/`, `influences/` | shared geometry and castle elements; site reading and grading; style genes; space scheme; influence fields |
| `castles/`, `temples/`, `edifices/`, `classics/`, `houses/` | plan and builder per type with its knowledge YAML |
| `ensembles/` | settlements: streets, plots, districts, landmarks; `settlements.yaml` |
| `levels/` | experimental: buildings against ground levels over time (slopes, a city over a city, tiered decks), clay models |
| `bench/` | `all.py`, `golden.py` (fingerprints), `deps.py` (part rules); stepwise sample recipes |

## Assets

None — fully procedural. Ground textures of the refined relief come from Site.

## Limitations

- Masonry surfaces are written for castles only (and a settlement's host castle), and only by the step, not by
  `buildings.py build|make`: temples, edifices and classical buildings get no masonry yet.
- Builders hold one plan at a time: several buildings in a scene are built one after another.
- Castles: no doors or stairs inside towers, solid court buildings. Temples: no sculpture, double aisles or double
  transepts. Edifices: interiors only for a courtyard, hall and dome. Settlements: no terraced blocks on slopes, stadium
  or forum; `levels/` is not wired into the step.
- Benches read site catalog places `out/relief/cat_*`; build them with
  `.venv/bin/python -m components.site.terrain_catalog [preset …] [--seeds 1,2,3] [--base real]`.

## FAQ

**How do I generate a procedural castle in Blender with Python?** Run the `building` step after a `relief` step:
`building: {type: castle, culture: german, scale: medium, seed: 7}`. Without the manager,
`.venv/bin/python -m components.buildings.buildings make castle <site folder> culture=french era=late seed=3` plans the
castle in Python and builds the mass in Blender.

**How do I generate a medieval town, a village or a Greek city?** Use `building: {type: settlement, archetype: town}`;
the archetypes are `polis`, `colony`, `subura`, `town`, `posad`, `castle_inside`, `village` and `hive`. The knobs
`power`, `trade`, `wealth`, `inequality`, `wind` and `centre` correct plots, façades, gardens, shops and colours.

**Can it build a Gothic cathedral?** Yes: `building: {type: temple, style: gothic_fr}`. Styles can be mixed
(`style: "gothic_fr+byzantine"` or `"gothic_fr:0.6,byzantine:0.4"`), and `history`, `wealth`, `ruin` and `irregular`
(0–1) set the construction campaigns, the richness, the ruin and the irregularity.

**Can I use it without the rest of sketch3d?** Yes, it also ships as its own repository, `sketch3d-blender-architecture`, with
the core it needs and the components it builds on. The plan phase needs no Blender: its site folder holds `relief.npz`
and `location.json`, which the `relief` step writes.

**Is the result reproducible?** One seed gives one building to the millimetre; plan and model fingerprints
(`components/buildings/bench/golden.json`) guard every change: `.venv/bin/python -m components.buildings.bench.all`.
