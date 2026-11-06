# Constructions — procedural masonry, roofing, paving and timber framing for Blender, detailed by camera distance

Constructions is a procedural stone and brick wall, roof tile, cobblestone paving and half-timbering generator for
Blender, written in Python. Over a building's surfaces it lays real elements at real sizes — stone and brick courses,
tiles, shingles and metal sheets, setts and floor boards, timber frames, log walls and fences — and gives every element
its own piece of a CC0 scan. Detail follows the cameras: blocks near the lens, course prisms farther, a mapped plate in
the distance. The steps read only the scene layers and face attributes written by Buildings (`surfaces.json`,
`occupancy.json`, palette, `sk_role`) and the camera objects of the `.blend`, and write the `constructions` layer.

**Use it to:**

- Lay a stone or brick wall course by course — ashlar, rubble, Flemish or English bond, herringbone, polygonal — on
  castle walls and towers: `masonry: {outer: brick_flemish, inner: brick_flemish, towers: ashlar}`
- Roof buildings with clay tiles, slate, shingles, thatch, copper or gilded standing seam, or the shingles of onion
  domes: `roofing: {covering: beaver, towers: slate, domes: lemekh}`
- Pave courtyards, roads and floors with cobblestones, setts, flagstones, herringbone brick or planks:
  `paving: {court: cobble, road: setts_large, floor: planks}`
- Build half-timbered upper storeys, log houses and board cladding on buildings:
  `timber: {fachwerk: fachwerk_cross}` or `timber: {buildings: log_oblo}`
- Keep the triangle count low while a castle stays sharp — full blocks near the camera, course prisms farther, a mapped
  plate in the distance: `masonry: {detail: cam, Tlo: 6, Thi: 14, Tfar: 3}`
- Give every stone, tile and board its own piece of a CC0 scan with no visible tiling: the `textures` step

## Features

- **Masonry — 23 packs on five layout generators**: coursed (`ashlar`, `ashlar_fine`, `bossage`, `ashlar_broken`,
  `rubble`, `dry_stone`), brick bonds (`brick_stretcher`, `brick_english`, `brick_flemish`, `brick_monk`,
  `brick_garden`), `banded`, `herringbone`, uncoursed cells (`rubble_random`, `polygonal`, `cyclopean`, `ishigaki`),
  `reticulatum`, `plinfa`, `flint`, `cobble_wall`, `whitestone`; voussoirs dressed along arches, quoins, jambs,
  loopholes, putlog holes.
- **Detail ladder** by pixels per course: blocks above `Thi`, a ragged per-stone band down to `Tlo`, course prisms with a
  real bed shadow, below `Tfar` only the mapped plate. Randomness comes from the element id: a stone looks the same in
  every run and detail mode.
- **Roofing — 22 coverings**: clay tiles (`plain`, `beaver`, `pointed`, `hexagon`, `scallop`, `pantile`, `roman`,
  `canal`), slate (`slate`, `slate_random`, `stone_slate`), `shingle`, `lemekh` (aspen shingles of onion domes, three
  cuts), `copper_scale`, `thatch`, standing seam (`copper_seam`, `gilded_seam`, `iron_seam`), `boards`. Overlapping
  bonded rows, hips, open valleys, ridges, finials; roofs are recognised from faces as slopes, cones or solids of
  revolution (onion, dome, helm, bell); far roofs become a shell with a row pattern in the shader.
- **Paving — 11 kinds** (`cobble`, `setts_small`, `setts_large`, `flagstone`, `flagstone_crazy`, `brick_herringbone`,
  `brick_basket`, `wood_pavement`, `planks`, `floor_tiles`, `floor_tiles_diag`) on yards and roads from occupancy, floor
  faces and your own areas, following the site ground.
- **Timber — 8 kinds**: `fachwerk`, `fachwerk_close`, `fachwerk_cross`, log walls with saddle notches (`log_oblo`) and
  dovetails (`log_lapa`), `boards`, `palisade` and `wattle` fences; windows from the dark opening faces.
- **Textures**: a real-size scan piece per element, shifted and rotated by its id (no visible repeat), normalised so the
  layout's tone `Col` is kept; per-stone weathering, worn edges and chips, mortar in the joints, grain along slopes and
  logs.

## How it works

Each step has a plan in Python — the step's section of the run scene's `constructions.yaml` plus its knowledge YAML →
`resolve` → a JSON plan — and a build in Blender over the open scene (`bl_stones.run`, `bl_roofing.run`,
`bl_paving.run`, `bl_timber.run`, `bl_textures.run`; surface engine `core/surface/bl_surface.py`). Masonry follows the
items of `surfaces.json` (a wall side or a tower ring with openings and gaps); roofing reads faces with the roof role or
roof colour; paving and timber read `occupancy.json`. Outputs: `masonry_<tag>`, `plate_<tag>`, `body_<tag>`,
`roof_<n>`, `roof_far_<n>`, `pave_<n>`, `timber_<n>`, `fence_<n>` with face attributes `sk_role`, `sk_stone` (element
id), `Col` (tone) and material slots per role that `textures` replaces. `masonry` writes a default `constructions.yaml`
when there is none; step keys override it; `covers: 0` in its `textures` section keeps coverings in flat tone.

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `masonry` | buildings, cameras → constructions | `outer` (ashlar), `inner` (rubble), `towers` (ashlar; `"0"` — none), `walls` (`"0"`, `"0,1"`, `all`), `detail` (cam), `rows`, `plate` (true), `Tlo`/`Thi` (6/14), `Tfar` (3), `seed` (1), `wall` (rock_surface) |
| `roofing` | buildings (+ cameras) → constructions | `covering` (beaver), per part `towers`, `buildings`, `gates`, per form `tents`, `domes`; `detail`, `Tlo`/`Thi` (5/12), `Tfar` (2.5), `seed` |
| `paving` | buildings, site (+ cameras) → constructions | `court` (cobble), `road` (setts_large), `floor` (planks), `areas` (flagstone), `width` (4 m), `skip`, `detail`, `Tlo`/`Thi` (8/14), `Tfar` (4), `seed` |
| `timber` | buildings, site (+ cameras) → constructions | `fachwerk` (fachwerk), `buildings`, `fences` (palisade), `detail`, `Tfar` (2), `seed` |
| `textures` | constructions (+ buildings) → constructions | the masonry keys; substances from the description |

Thresholds are pixels per course, row or element; `final` sets `detail: all` for the layout steps, `sketch` skips all five.
All keys with descriptions: [component.yaml](/components/constructions/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, culture: german, scale: medium, seed: 7}
  - camera: {view: gate, azimuth: 200, elevation: 10, distance: 170, res: 640x360}   # detail follows the cameras
  - masonry: {detail: cam, outer: bossage}
  - roofing: {covering: beaver, towers: slate}
  - paving: {court: cobble, road: setts_large}
  - timber: {fachwerk: fachwerk_cross}
  - textures
  - light: {time: day}
```

```sh
.venv/bin/python -m core.manager.pipeline components/constructions/textures/bench/castle/castle_textures.yaml   # all five steps
.venv/bin/python -m components.constructions.masonry.bench.bench [--blender]   # layouts vs golden.json; --blender: step contract
.venv/bin/python -m components.constructions.roofing.bench.bench               # also paving.bench.bench, timber.bench.bench
.venv/bin/python -m components.constructions.textures.bench.bench              # substances match the packs, scans present
```

## Driving it from an AI agent

The steps go after the building and the cameras (detail follows the camera objects); `masonry` needs the surfaces that
the `building` step writes for a castle:

```yaml
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, culture: german, scale: medium, seed: 7}
  - camera: {view: gate, azimuth: 200, elevation: 10, distance: 170, res: 640x360}
  - masonry: {detail: cam, outer: brick_flemish, inner: brick_flemish, towers: brick_flemish}
  - roofing: {covering: slate}
  - paving: {court: cobble, road: setts_large}
  - textures
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry`; the layouts are plain Python
  with benches (`.venv/bin/python -m components.constructions.masonry.bench.bench`, also `roofing`, `paving`, `timber`).
  All keys, defaults and values: `components/constructions/component.yaml`; the scene's `constructions.yaml` has a
  section per step, and step keys override it.
- A run prints `__RESULT__{"pipeline", "out", "failures", …}` as its last line; the run folder holds `journal.md` and,
  with the `frames` output, a contact sheet `contact.jpg`.
- Request to keys: "a brick castle with slate roofs and a cobbled yard" → the lines above; "log houses" →
  `timber: {buildings: log_oblo}`; "a half-timbered look" → `timber: {fachwerk: fachwerk_cross}`;
  "rough dry-stone walls" → `masonry: {outer: dry_stone}`.

## Files

| Path | Role |
|---|---|
| `steps.py` | plan and build of the five steps; default description |
| `masonry/` | side unfolding (`Strip`), layouts, maps, `resolve`; builder `bl_stones.py`; `packs.yaml` |
| `roofing/`, `paving/`, `timber/` | layouts and builders; `coverings.yaml`, `paving.yaml`, `timber.yaml` |
| `textures/` | the step, stone weathering, masks, substances of coverings; `states/` — experimental, unused by the steps |
| `*/bench/` | benches with `golden.json` fingerprints; substance stands |

## Assets

- Surface scans for `textures`, Poly Haven, CC0 (17 surfaces, 2K JPG, ~135 MB):
  `.venv/bin/python -m core.surface.surface_tex fetch`
- Moss and mud for `textures/states/`, ambientCG and Poly Haven, CC0: `.venv/bin/python -m components.constructions.textures.states.fetch`

## Limitations

- Buildings write masonry surfaces for castles only: temples, edifices and classical buildings get no masonry yet.
- Masonry openings are round arches and rectangles; gatehouses, keeps, merlons and staged towers carry no masonry;
  `banded` walls are textured as stone; no metal or stair parts.
- Detail follows the camera objects of the `.blend`, not per-frame cameras of an animated shot.
- `detail: all` builds everything: a castle with its suburb reached ~19 million triangles against ~94 thousand by
  camera, for a nearly identical frame.

## FAQ

**How do I generate a procedural brick or stone wall in Blender with Python?** Build a castle with the `building` step,
then `masonry: {outer: brick_flemish}` (23 packs, among them `ashlar`, `brick_english`, `polygonal`, `ishigaki`). The
stone layout is a Python function, `components.constructions.masonry.masonry.layout(strip, pack, gaps=…, seed=…)`;
Blender only builds the geometry.

**How do I keep the triangle count down?** Use `detail: cam`, the default outside the `final` profile: blocks above
`Thi` pixels per course, a ragged band down to `Tlo`, course prisms down to `Tfar`, a mapped plate below. `detail: all`
builds every element: a castle with its suburb reached ~19 million triangles against ~94 thousand by camera.

**How do I add a palisade or a wattle fence?** List the fences under `timber:` in the scene's `constructions.yaml`; a
fence without `kind` takes the `fences` key of the `timber` step (default `palisade`):

```yaml
timber:
  fences: [{path: [[0, 0], [30, 0], [30, 20]], kind: palisade}]
```

**Can I roof or pave a model of my own?** The steps read scene layers and face attributes: `roofing` takes faces with
the roof role (`sk_role` = 2) or the palette's roof colour, `paving` and `timber` read `occupancy.json`, `masonry` reads
`surfaces.json`. The `building` step writes them (masonry surfaces for castles only), so a model needs the same data.

**Can I use it without the rest of sketch3d?** Yes, it also ships as its own repository, `sketch3d-blender-masonry`, with the
core it needs and the `buildings` component that writes those layers. Scans are downloaded once:
`.venv/bin/python -m core.surface.surface_tex fetch`.
