/**
 * src/wasm/terrain.js — loader & zero-copy linear-memory bridge for `terrain.wasm`.
 *
 * Loads once at module import (works in both browser via fetch and Node via fs)
 * so `createTerrain(seed)` stays synchronous everywhere, including `probe/*.mjs`.
 */

let mod = null;
let mem = null;
let views = null;
let activeId = -1;
let nextId = 1;

const wasmUrl = new URL('./terrain.wasm', import.meta.url);

async function loadWasm() {
  try {
    let bytes;
    if (typeof window === 'undefined' && typeof process !== 'undefined' && process.versions?.node) {
      const { readFileSync } = await import('node:fs');
      const { fileURLToPath } = await import('node:url');
      bytes = readFileSync(fileURLToPath(wasmUrl));
    } else {
      const res = await fetch(wasmUrl);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      bytes = await res.arrayBuffer();
    }
    const { instance } = await WebAssembly.instantiate(bytes, {});
    mod = instance.exports;
    mem = mod.memory;
    const buf = mem.buffer;
    views = {
      permA: new Uint8Array(buf, mod.perm_a_ptr(), 512),
      permB: new Uint8Array(buf, mod.perm_b_ptr(), 512),
      permC: new Uint8Array(buf, mod.perm_c_ptr(), 512),
      permW: new Uint8Array(buf, mod.perm_w_ptr(), 512),
      inX: new Float64Array(buf, mod.in_x_ptr(), 4096),
      inZ: new Float64Array(buf, mod.in_z_ptr(), 4096),
      inLat: new Float64Array(buf, mod.in_lat_ptr(), 4096),
      outY: new Float64Array(buf, mod.out_y_ptr(), 4096),
      outFine: new Float64Array(buf, mod.out_fine_ptr(), 4096),
      outBroad: new Float64Array(buf, mod.out_broad_ptr(), 4096),
      outCont: new Float64Array(buf, mod.out_cont_ptr(), 4096),
      outM1: new Float64Array(buf, mod.out_m1_ptr(), 4096),
      outM2: new Float64Array(buf, mod.out_m2_ptr(), 4096),
      outM3: new Float64Array(buf, mod.out_m3_ptr(), 4096),
      tileH: new Float32Array(buf, mod.tile_h_ptr(), 35 * 35),
      tilePos: new Float32Array(buf, mod.tile_pos_ptr(), 1221 * 3),
      tileNor: new Float32Array(buf, mod.tile_nor_ptr(), 1221 * 3),
    };
  } catch (err) {
    console.warn('[terrain_wasm] unavailable, falling back to JS:', err?.message || err);
    mod = null;
  }
}

await loadWasm();

export function hasTerrainWasm() {
  return mod !== null;
}

export function createWasmTerrainHandle(permA, permB, permC, permW) {
  if (!mod) return null;
  const id = nextId++;
  let lr = NaN, lm0 = NaN, lm1 = NaN, lm2 = NaN, lm3 = NaN, lm4 = NaN, lm5 = NaN, ld = -1, lrg = NaN;

  function bind(TERRAIN, WORLD_BIOME) {
    if (activeId !== id) {
      views.permA.set(permA);
      views.permB.set(permB);
      views.permC.set(permC);
      views.permW.set(permW);
      activeId = id;
      lr = NaN;
    }
    const m = TERRAIN.mix;
    const r = TERRAIN.relief;
    const d = WORLD_BIOME.dunes ? 1 : 0;
    const rg = WORLD_BIOME.roughness ?? 1;
    if (r !== lr || m.plains !== lm0 || m.hills !== lm1 || m.valley !== lm2 ||
        m.mountain !== lm3 || m.canyon !== lm4 || m.plateau !== lm5 || d !== ld || rg !== lrg) {
      lr = r; lm0 = m.plains; lm1 = m.hills; lm2 = m.valley;
      lm3 = m.mountain; lm4 = m.canyon; lm5 = m.plateau; ld = d; lrg = rg;
      mod.set_params(r, lm0, lm1, lm2, lm3, lm4, lm5, d, rg);
    }
  }

  return {
    bind,
    views,
    mod,
  };
}
