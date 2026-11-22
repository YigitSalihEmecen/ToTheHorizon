#![allow(static_mut_refs)]

use core::f64::consts::PI;

const MAX_BATCH: usize = 4096;

static mut PERM_A: [u8; 512] = [0; 512];
static mut PERM_B: [u8; 512] = [0; 512];
static mut PERM_C: [u8; 512] = [0; 512];
static mut PERM_W: [u8; 512] = [0; 512];

static mut IN_X: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut IN_Z: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut IN_LAT: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut OUT_Y: [f64; MAX_BATCH] = [0.0; MAX_BATCH];

static mut OUT_FINE: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut OUT_BROAD: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut OUT_CONT: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut OUT_M1: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut OUT_M2: [f64; MAX_BATCH] = [0.0; MAX_BATCH];
static mut OUT_M3: [f64; MAX_BATCH] = [0.0; MAX_BATCH];

// Tile geometry buffers (N=32, V=33, G=35, vCount = 33*33 + 4*33 = 1221)
const TILE_N: usize = 32;
const TILE_V: usize = 33;
const TILE_G: usize = 35;
const TILE_VCOUNT: usize = TILE_V * TILE_V + 4 * TILE_V;

static mut TILE_H: [f32; TILE_G * TILE_G] = [0.0; TILE_G * TILE_G];
static mut TILE_POS: [f32; TILE_VCOUNT * 3] = [0.0; TILE_VCOUNT * 3];
static mut TILE_NOR: [f32; TILE_VCOUNT * 3] = [0.0; TILE_VCOUNT * 3];

static mut RELIEF: f64 = 1.0;
static mut MIX: [f64; 6] = [1.0; 6];
static mut DUNES: bool = false;
static mut ROUGHNESS: f64 = 1.0;

const GRAD_X: [f64; 8] = [1.0, -1.0, 0.0, 0.0, 0.7071, -0.7071, 0.7071, -0.7071];
const GRAD_Y: [f64; 8] = [0.0, 0.0, 1.0, -1.0, 0.7071, 0.7071, -0.7071, -0.7071];

const RIDGE_SOFT: f64 = 0.004;
const SIGMA2: f64 = 2.0 * 0.245 * 0.245;
const CONTINENT_AMP: f64 = 340.0;

// Precomputed centres: (i / 6) * 2*PI + 0.4, radius 0.52
// Matches JS: Math.cos(a) * 0.52, Math.sin(a) * 0.52
#[inline(always)]
fn centres() -> [(f64, f64); 6] {
    let mut c = [(0.0, 0.0); 6];
    let mut i = 0;
    while i < 6 {
        let a = (i as f64 / 6.0) * PI * 2.0 + 0.4;
        c[i] = (a.cos() * 0.52, a.sin() * 0.52);
        i += 1;
    }
    c
}

#[inline(always)]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

#[inline(always)]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[inline(always)]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
fn lod_amp(i: usize, lod_oct: f64) -> f64 {
    if lod_oct >= 99.0 {
        1.0
    } else {
        clamp(lod_oct - i as f64, 0.0, 1.0)
    }
}

#[inline(always)]
fn grad_noise_val(perm: &[u8; 512], x: f64, y: f64) -> f64 {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;

    let u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    let xi = (ix as i32 & 255) as usize;
    let yi = (iy as i32 & 255) as usize;

    unsafe {
        let p0 = *perm.get_unchecked(yi) as usize;
        let p1 = *perm.get_unchecked(yi + 1) as usize;
        let ia = (*perm.get_unchecked(xi + p0) & 7) as usize;
        let ib = (*perm.get_unchecked(xi + 1 + p0) & 7) as usize;
        let ic = (*perm.get_unchecked(xi + p1) & 7) as usize;
        let id = (*perm.get_unchecked(xi + 1 + p1) & 7) as usize;

        let va = *GRAD_X.get_unchecked(ia) * fx + *GRAD_Y.get_unchecked(ia) * fy;
        let vb = *GRAD_X.get_unchecked(ib) * (fx - 1.0) + *GRAD_Y.get_unchecked(ib) * fy;
        let vc = *GRAD_X.get_unchecked(ic) * fx + *GRAD_Y.get_unchecked(ic) * (fy - 1.0);
        let vd = *GRAD_X.get_unchecked(id) * (fx - 1.0) + *GRAD_Y.get_unchecked(id) * (fy - 1.0);

        let k1 = vb - va;
        let k2 = vc - va;
        let k3 = va - vb - vc + vd;
        va + u * k1 + v * k2 + u * v * k3
    }
}

#[inline(always)]
fn grad_noise_deriv(perm: &[u8; 512], x: f64, y: f64) -> (f64, f64, f64) {
    let ix = x.floor();
    let iy = y.floor();
    let fx = x - ix;
    let fy = y - iy;

    let u = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let v = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let du = 30.0 * fx * fx * (fx * (fx - 2.0) + 1.0);
    let dv = 30.0 * fy * fy * (fy * (fy - 2.0) + 1.0);

    let xi = (ix as i32 & 255) as usize;
    let yi = (iy as i32 & 255) as usize;

    unsafe {
        let p0 = *perm.get_unchecked(yi) as usize;
        let p1 = *perm.get_unchecked(yi + 1) as usize;
        let ia = (*perm.get_unchecked(xi + p0) & 7) as usize;
        let ib = (*perm.get_unchecked(xi + 1 + p0) & 7) as usize;
        let ic = (*perm.get_unchecked(xi + p1) & 7) as usize;
        let id = (*perm.get_unchecked(xi + 1 + p1) & 7) as usize;

        let gax = *GRAD_X.get_unchecked(ia);
        let gay = *GRAD_Y.get_unchecked(ia);
        let gbx = *GRAD_X.get_unchecked(ib);
        let gby = *GRAD_Y.get_unchecked(ib);
        let gcx = *GRAD_X.get_unchecked(ic);
        let gcy = *GRAD_Y.get_unchecked(ic);
        let gdx = *GRAD_X.get_unchecked(id);
        let gdy = *GRAD_Y.get_unchecked(id);

        let va = gax * fx + gay * fy;
        let vb = gbx * (fx - 1.0) + gby * fy;
        let vc = gcx * fx + gcy * (fy - 1.0);
        let vd = gdx * (fx - 1.0) + gdy * (fy - 1.0);

        let k1 = vb - va;
        let k2 = vc - va;
        let k3 = va - vb - vc + vd;
        let uv = u * v;

        let val = va + u * k1 + v * k2 + uv * k3;
        let dx = gax + u * (gbx - gax) + v * (gcx - gax) + uv * (gax - gbx - gcx + gdx) + du * (k1 + v * k3);
        let dy = gay + u * (gby - gay) + v * (gcy - gay) + uv * (gay - gby - gcy + gdy) + dv * (k2 + u * k3);
        (val, dx, dy)
    }
}

#[inline(always)]
fn fbm(perm: &[u8; 512], x: f64, y: f64, freq: f64, oct: usize, gain: f64, lod_oct: f64) -> f64 {
    let mut a = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    let mut norm = 0.0;
    let mut i = 0;
    while i < oct {
        let w = amp * lod_amp(i, lod_oct);
        if w <= 0.0 {
            break;
        }
        a += w * grad_noise_val(perm, x * f, y * f);
        norm += w;
        amp *= gain;
        f *= 2.0;
        i += 1;
    }
    if norm > 0.0 { a / norm } else { 0.0 }
}

#[inline(always)]
fn eroded_fbm(perm: &[u8; 512], x: f64, y: f64, freq: f64, oct: usize, gain: f64, damp: f64, lod_oct: f64) -> f64 {
    let mut a = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    let mut norm = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    let mut i = 0;
    while i < oct {
        let w = amp * lod_amp(i, lod_oct);
        if w <= 0.0 {
            break;
        }
        let (v, gx, gy) = grad_noise_deriv(perm, x * f, y * f);
        a += (w * v) / (1.0 + damp * (dx * dx + dy * dy));
        dx += gx;
        dy += gy;
        norm += w;
        amp *= gain;
        f *= 2.0;
        i += 1;
    }
    if norm > 0.0 { a / norm } else { 0.0 }
}

#[inline(always)]
fn ridged_fbm(perm: &[u8; 512], x: f64, y: f64, freq: f64, oct: usize, gain: f64, damp: f64, lod_oct: f64) -> f64 {
    let mut a = 0.0;
    let mut amp = 1.0;
    let mut f = freq;
    let mut norm = 0.0;
    let mut dx = 0.0;
    let mut dy = 0.0;
    let mut i = 0;
    while i < oct {
        let w = amp * lod_amp(i, lod_oct);
        if w <= 0.0 {
            break;
        }
        let (n, gx, gy) = grad_noise_deriv(perm, x * f, y * f);
        let r = 1.0 - (n * n + RIDGE_SOFT).sqrt();
        a += (w * r * r) / (1.0 + damp * (dx * dx + dy * dy));
        dx += gx;
        dy += gy;
        norm += w;
        amp *= gain;
        f *= 2.0;
        i += 1;
    }
    if norm > 0.0 { (a / norm) * 2.0 - 1.0 } else { -1.0 }
}

#[inline(always)]
fn archetypes_inner(x: f64, y: f64, lod_oct: f64) -> [f64; 6] {
    unsafe {
        let u = fbm(&PERM_A, x, y, 0.00014, 2, 0.5, lod_oct) * 2.1;
        let v = fbm(&PERM_B, x + 512.7, y - 311.3, 0.00022, 2, 0.5, lod_oct) * 2.1;
        let c = centres();
        let mut w = [0.0; 6];
        let mut total = 0.0;
        let mut i = 0;
        while i < 6 {
            let dx = u - c[i].0;
            let dy = v - c[i].1;
            let wi = (-(dx * dx + dy * dy) / SIGMA2).exp() * MIX[i];
            w[i] = wi;
            total += wi;
            i += 1;
        }
        if total < 1e-9 {
            w[1] = 1.0;
            total = 1.0;
        }
        let inv = 1.0 / total;
        let mut j = 0;
        while j < 6 {
            w[j] *= inv;
            j += 1;
        }
        w
    }
}

#[inline(always)]
fn plains_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe { eroded_fbm(&PERM_A, x, y, 0.0022, 4, 0.5, 0.30, lod_oct) * 11.0 }
}

#[inline(always)]
fn hills_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe { eroded_fbm(&PERM_A, x, y, 0.0014, 6, 0.52, 0.40, lod_oct) * 105.0 }
}

#[inline(always)]
fn valley_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe {
        let trough = ridged_fbm(&PERM_B, x, y, 0.0011, 5, 0.5, 0.30, lod_oct);
        let floor = -60.0 + eroded_fbm(&PERM_A, x, y, 0.0022, 3, 0.5, 0.30, lod_oct) * 16.0;
        floor + (1.0 - trough) * 128.0
    }
}

#[inline(always)]
fn mountain_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe {
        let ridge = ridged_fbm(&PERM_B, x, y, 0.00082, 7, 0.5, 0.42, lod_oct) * 0.5 + 0.5;
        let bulk = eroded_fbm(&PERM_A, x, y, 0.0006, 4, 0.5, 0.35, lod_oct);
        let peaked = ridge * ridge * (3.0 - 2.0 * ridge);
        bulk * 150.0 + peaked * 620.0
    }
}

#[inline(always)]
fn canyon_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe {
        let r = ridged_fbm(&PERM_C, x, y, 0.0011, 4, 0.5, 0.28, lod_oct) * 0.5 + 0.5;
        let gorge = smoothstep(0.26, 0.68, r);
        let rim = eroded_fbm(&PERM_A, x, y, 0.0019, 3, 0.5, 0.3, lod_oct) * 18.0;
        48.0 + rim - (1.0 - gorge) * 128.0
    }
}

#[inline(always)]
fn plateau_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe {
        let b = eroded_fbm(&PERM_A, x, y, 0.0013, 4, 0.5, 0.38, lod_oct);
        let steps = 3.0;
        let t = b * steps;
        let base = t.floor();
        let frac = t - base;
        let terraced = (base + smoothstep(0.3, 0.7, frac)) / steps;
        terraced * 135.0 + 18.0
    }
}

#[inline(always)]
fn dunes_h(x: f64, y: f64, lod_oct: f64) -> f64 {
    unsafe {
        let sea = smoothstep(-0.12, 0.22, fbm(&PERM_B, x + 1903.1, y - 711.7, 0.00021, 2, 0.5, lod_oct));
        if sea <= 0.0 {
            return 0.0;
        }
        let wind_c = 0.5_f64.cos();
        let wind_s = 0.5_f64.sin();
        let along = x * wind_c + y * wind_s;
        let across = y * wind_c - x * wind_s;
        let ph = (along + fbm(&PERM_C, x - 77.7, y + 301.3, 0.0011, 2, 0.5, lod_oct) * 160.0) / 210.0;
        let p = ph - ph.floor();
        let a = 0.74;
        let t = if p < a { p / a } else { (1.0 - p) / (1.0 - a) };
        let crest = t * t * (3.0 - 2.0 * t);
        let amp = 6.0 + 12.0 * smoothstep(
            -0.4,
            0.5,
            fbm(&PERM_A, across * 0.6 - 431.9, along * 0.15 + 88.3, 0.0021, 2, 0.5, lod_oct),
        );
        (crest - 0.35) * amp * sea
    }
}

#[inline(always)]
fn continent_inner(x: f64, y: f64) -> f64 {
    unsafe { eroded_fbm(&PERM_A, x + 4211.7, y - 1877.3, 0.00009, 2, 0.55, 0.22, 99.0) * CONTINENT_AMP }
}

#[inline(always)]
fn base_inner(x: f64, z: f64, lod_oct: f64) -> f64 {
    unsafe {
        let wx = x + fbm(&PERM_W, x + 137.2, z - 41.7, 0.0006, 3, 0.5, lod_oct) * 130.0;
        let wy = z + fbm(&PERM_W, x - 88.4, z + 219.6, 0.0006, 3, 0.5, lod_oct) * 130.0;
        let w = archetypes_inner(x, z, lod_oct);
        let e = 1e-4;
        let mut h = 0.0;
        if w[0] > e { h += w[0] * plains_h(wx, wy, lod_oct); }
        if w[1] > e { h += w[1] * hills_h(wx, wy, lod_oct); }
        if w[2] > e { h += w[2] * valley_h(wx, wy, lod_oct); }
        if w[3] > e { h += w[3] * mountain_h(wx, wy, lod_oct); }
        if w[4] > e { h += w[4] * canyon_h(wx, wy, lod_oct); }
        if w[5] > e { h += w[5] * plateau_h(wx, wy, lod_oct); }
        h = h * RELIEF + continent_inner(x, z);
        if DUNES {
            h += dunes_h(wx, wy, lod_oct);
        }
        h
    }
}

#[inline(always)]
fn height_inner(x: f64, z: f64, lateral: f64) -> f64 {
    unsafe {
        let octaves = lerp(8.0, 3.2, smoothstep(80.0, 420.0, lateral));
        let mut h = base_inner(x, z, octaves);
        let detail = 1.0 - smoothstep(60.0, 200.0, lateral);
        let graded = lerp(0.2, 1.0, smoothstep(16.0, 90.0, lateral));
        let mid = 1.0 - smoothstep(150.0, 380.0, lateral);
        let rough = ROUGHNESS;
        if mid > 0.0 {
            h += eroded_fbm(&PERM_C, x, z, 0.0075, 3, 0.5, 0.5, 99.0) * 6.2 * lerp(0.35, 1.0, graded) * mid * rough;
        }
        if detail > 0.0 {
            h += fbm(&PERM_C, x + 91.3, z - 55.1, 0.028, 2, 0.5, 99.0) * 0.85 * detail * graded * rough;
        }
        h
    }
}

// ------------------------------------------------------------ WASM exports --

#[no_mangle] pub extern "C" fn perm_a_ptr() -> *mut u8 { unsafe { PERM_A.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn perm_b_ptr() -> *mut u8 { unsafe { PERM_B.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn perm_c_ptr() -> *mut u8 { unsafe { PERM_C.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn perm_w_ptr() -> *mut u8 { unsafe { PERM_W.as_mut_ptr() } }

#[no_mangle] pub extern "C" fn in_x_ptr() -> *mut f64 { unsafe { IN_X.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn in_z_ptr() -> *mut f64 { unsafe { IN_Z.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn in_lat_ptr() -> *mut f64 { unsafe { IN_LAT.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn out_y_ptr() -> *mut f64 { unsafe { OUT_Y.as_mut_ptr() } }

#[no_mangle] pub extern "C" fn out_fine_ptr() -> *mut f64 { unsafe { OUT_FINE.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn out_broad_ptr() -> *mut f64 { unsafe { OUT_BROAD.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn out_cont_ptr() -> *mut f64 { unsafe { OUT_CONT.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn out_m1_ptr() -> *mut f64 { unsafe { OUT_M1.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn out_m2_ptr() -> *mut f64 { unsafe { OUT_M2.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn out_m3_ptr() -> *mut f64 { unsafe { OUT_M3.as_mut_ptr() } }

#[no_mangle] pub extern "C" fn tile_h_ptr() -> *mut f32 { unsafe { TILE_H.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn tile_pos_ptr() -> *mut f32 { unsafe { TILE_POS.as_mut_ptr() } }
#[no_mangle] pub extern "C" fn tile_nor_ptr() -> *mut f32 { unsafe { TILE_NOR.as_mut_ptr() } }

#[no_mangle]
pub extern "C" fn set_params(
    relief: f64,
    m0: f64, m1: f64, m2: f64, m3: f64, m4: f64, m5: f64,
    dunes: i32,
    roughness: f64,
) {
    unsafe {
        RELIEF = relief;
        MIX = [m0, m1, m2, m3, m4, m5];
        DUNES = dunes != 0;
        ROUGHNESS = roughness;
    }
}

#[no_mangle] pub extern "C" fn height_one(x: f64, z: f64, lateral: f64) -> f64 { height_inner(x, z, lateral) }
#[no_mangle] pub extern "C" fn base_one(x: f64, z: f64, octaves: f64) -> f64 { base_inner(x, z, octaves) }
#[no_mangle] pub extern "C" fn continent_one(x: f64, z: f64) -> f64 { continent_inner(x, z) }

#[no_mangle]
pub extern "C" fn mask_one(x: f64, z: f64, freq: f64, ox: f64, oz: f64) -> f64 {
    unsafe { fbm(&PERM_C, x + ox, z + oz, freq, 2, 0.5, 99.0) * 0.5 + 0.5 }
}

#[no_mangle]
pub extern "C" fn forest_density_one(x: f64, z: f64) -> f64 {
    unsafe {
        let stand = smoothstep(-0.28, 0.42, fbm(&PERM_B, x - 812.3, z + 244.9, 0.0016, 2, 0.5, 99.0));
        let broken = smoothstep(-0.55, 0.5, fbm(&PERM_C, x + 55.5, z - 120.2, 0.0068, 2, 0.5, 99.0));
        stand * lerp(0.3, 1.0, broken)
    }
}

#[no_mangle]
pub extern "C" fn region_one(x: f64, z: f64) -> f64 {
    let w = archetypes_inner(x, z, 99.0);
    clamp(w[3] + w[4] * 0.7 + w[5] * 0.35, 0.0, 1.0)
}

#[no_mangle] pub extern "C" fn na_one(x: f64, z: f64) -> f64 { unsafe { grad_noise_val(&PERM_A, x, z) } }
#[no_mangle] pub extern "C" fn nb_one(x: f64, z: f64) -> f64 { unsafe { grad_noise_val(&PERM_B, x, z) } }
#[no_mangle] pub extern "C" fn nc_one(x: f64, z: f64) -> f64 { unsafe { grad_noise_val(&PERM_C, x, z) } }

/// Topographic Position Index at (x, z): 5 fused `base_inner(..., 3.0)` evaluations.
#[no_mangle]
pub extern "C" fn tpi_one(x: f64, z: f64) -> f64 {
    const R: f64 = 120.0;
    base_inner(x, z, 3.0)
        - 0.25 * (
            base_inner(x + R, z, 3.0)
            + base_inner(x - R, z, 3.0)
            + base_inner(x, z + R, 3.0)
            + base_inner(x, z - R, 3.0)
        )
}

/// Computes main grid + skirt positions and normals from `TILE_H[35*35]` in one pass.
#[no_mangle]
pub extern "C" fn build_tile_geom(step: f32, drop: f32) {
    let inv2step = 1.0 / (2.0 * step);
    unsafe {
        for vj in 0..TILE_V {
            let z = vj as f32 * step;
            let row_g = (vj + 1) * TILE_G;
            let row_v = vj * TILE_V;
            for vi in 0..TILE_V {
                let g = row_g + vi + 1;
                let y = *TILE_H.get_unchecked(g);
                let dx = (*TILE_H.get_unchecked(g + 1) - *TILE_H.get_unchecked(g - 1)) * inv2step;
                let dz = (*TILE_H.get_unchecked(g + TILE_G) - *TILE_H.get_unchecked(g - TILE_G)) * inv2step;
                let inv = 1.0 / (1.0 + dx * dx + dz * dz).sqrt();
                let k = (row_v + vi) * 3;
                *TILE_POS.get_unchecked_mut(k) = vi as f32 * step;
                *TILE_POS.get_unchecked_mut(k + 1) = y;
                *TILE_POS.get_unchecked_mut(k + 2) = z;
                *TILE_NOR.get_unchecked_mut(k) = -dx * inv;
                *TILE_NOR.get_unchecked_mut(k + 1) = inv;
                *TILE_NOR.get_unchecked_mut(k + 2) = -dz * inv;
            }
        }
        // Skirts: 4 borders of TILE_V vertices
        let mut b = 0usize;
        let copy_border = |src_vi: usize, dst_idx: usize| {
            let s = src_vi * 3;
            let d = (TILE_V * TILE_V + dst_idx) * 3;
            *TILE_POS.get_unchecked_mut(d) = *TILE_POS.get_unchecked(s);
            *TILE_POS.get_unchecked_mut(d + 1) = *TILE_POS.get_unchecked(s + 1) - drop;
            *TILE_POS.get_unchecked_mut(d + 2) = *TILE_POS.get_unchecked(s + 2);
            *TILE_NOR.get_unchecked_mut(d) = *TILE_NOR.get_unchecked(s);
            *TILE_NOR.get_unchecked_mut(d + 1) = *TILE_NOR.get_unchecked(s + 1);
            *TILE_NOR.get_unchecked_mut(d + 2) = *TILE_NOR.get_unchecked(s + 2);
        };
        for t in 0..TILE_V { copy_border(t, b); b += 1; }
        for t in 0..TILE_V { copy_border(TILE_N * TILE_V + t, b); b += 1; }
        for t in 0..TILE_V { copy_border(t * TILE_V, b); b += 1; }
        for t in 0..TILE_V { copy_border(t * TILE_V + TILE_N, b); b += 1; }
    }
}

#[no_mangle]
pub extern "C" fn height_batch(count: usize) {
    let n = if count > MAX_BATCH { MAX_BATCH } else { count };
    unsafe {
        let mut i = 0;
        while i < n {
            let x = *IN_X.get_unchecked(i);
            let z = *IN_Z.get_unchecked(i);
            let lat = *IN_LAT.get_unchecked(i);
            *OUT_Y.get_unchecked_mut(i) = height_inner(x, z, lat);
            i += 1;
        }
    }
}

/// Evaluates all 6 noise channels used by `chunks.js:_groundColor` for a batch of (x, z) points.
#[no_mangle]
pub extern "C" fn ground_noise_batch(count: usize) {
    let n = if count > MAX_BATCH { MAX_BATCH } else { count };
    unsafe {
        let mut i = 0;
        while i < n {
            let x = *IN_X.get_unchecked(i);
            let z = *IN_Z.get_unchecked(i);
            *OUT_FINE.get_unchecked_mut(i) = grad_noise_val(&PERM_C, x * 0.014, z * 0.014) * 0.5 + 0.5;
            *OUT_BROAD.get_unchecked_mut(i) = grad_noise_val(&PERM_B, x * 0.0029, z * 0.0029) * 0.5 + 0.5;
            *OUT_CONT.get_unchecked_mut(i) = continent_inner(x, z);
            *OUT_M1.get_unchecked_mut(i) = fbm(&PERM_C, x + 311.0, z - 97.0, 0.012, 2, 0.5, 99.0) * 0.5 + 0.5;
            *OUT_M2.get_unchecked_mut(i) = fbm(&PERM_C, x - 523.0, z + 410.0, 0.009, 2, 0.5, 99.0) * 0.5 + 0.5;
            *OUT_M3.get_unchecked_mut(i) = fbm(&PERM_C, x + 77.0, z + 1203.0, 0.016, 2, 0.5, 99.0) * 0.5 + 0.5;
            i += 1;
        }
    }
}
