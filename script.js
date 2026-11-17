/* Drum Scribe signal processing: band-split spectral flux, peak picking, tempo and downbeat. */
const DSP = (function () {
  const N = 1024, HOP = 256;

  function makeFFT(n) {
    const levels = Math.round(Math.log2(n));
    const cos = new Float32Array(n / 2), sin = new Float32Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / n); sin[i] = Math.sin(2 * Math.PI * i / n); }
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < levels; b++) r |= ((i >> b) & 1) << (levels - 1 - b);
      rev[i] = r;
    }
    return function (re, im) {
      for (let i = 0; i < n; i++) {
        const j = rev[i];
        if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
      }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const l = j + half;
            const tre = re[l] * cos[k] + im[l] * sin[k];
            const tim = im[l] * cos[k] - re[l] * sin[k];
            re[l] = re[j] - tre; im[l] = im[j] - tim;
            re[j] += tre; im[j] += tim;
          }
        }
      }
    };
  }

  function percentile(a, p) {
    const c = Float32Array.from(a).sort();
    return c[Math.min(c.length - 1, Math.floor(p * (c.length - 1)))];
  }

  function normalize(a) {
    const p = percentile(a, 0.97);
    const sc = p > 1e-6 ? 1 / p : 1;
    for (let i = 0; i < a.length; i++) a[i] *= sc;
  }

  function bandFlux(cur, prevMax, lo, hi) {
    let s = 0;
    for (let b = lo; b <= hi; b++) { const d = cur[b] - prevMax[b]; if (d > 0) s += d; }
    return s / (hi - lo + 1);
  }

  function localMedian(arr, radius) {
    const n = arr.length, out = new Float32Array(n), win = new Float32Array(2 * radius + 1);
    const stride = 4;
    for (let i = 0; i < n; i += stride) {
      const lo = Math.max(0, i - radius), hi = Math.min(n - 1, i + radius), m = hi - lo + 1;
      const w = win.subarray(0, m);
      for (let j = 0; j < m; j++) w[j] = arr[lo + j];
      w.sort();
      const v = w[m >> 1];
      for (let k = i; k < Math.min(n, i + stride); k++) out[k] = v;
    }
    return out;
  }

  // Three passes of an 8-sample average, then keep every 8th sample: a cheap low-pass and downsample.
  function lowBand(x0, len) {
    let x = x0;
    for (let p = 0; p < 3; p++) {
      const y = new Float32Array(len); let acc = 0;
      for (let i = 0; i < len; i++) { acc += x[i]; if (i >= 8) acc -= x[i - 8]; y[i] = acc / 8; }
      x = y;
    }
    const ld = Math.floor(len / 8), out = new Float32Array(ld);
    for (let i = 0; i < ld; i++) out[i] = x[i * 8 + 4];
    return out;
  }

  function maxFilterR(a, r) {
    const n = a.length, o = new Float32Array(n);
    for (let i = 0; i < n; i++) { let m = 0; for (let j = Math.max(0, i - r); j <= Math.min(n - 1, i + r); j++) if (a[j] > m) m = a[j]; o[i] = m; }
    return o;
  }

  function maxFilter3(a) {
    const n = a.length, o = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let m = a[i];
      if (i > 0 && a[i - 1] > m) m = a[i - 1];
      if (i < n - 1 && a[i + 1] > m) m = a[i + 1];
      o[i] = m;
    }
    return o;
  }

  async function analyzeSamples(chans, sr0, onProgress) {
    const factor = Math.max(1, Math.round(sr0 / 22050));
    const sr = sr0 / factor;
    const len = Math.floor(chans[0].length / factor);
    if (len / sr < 6) throw new Error('That file is too short to find a beat in. Try a full song.');
    const mono = new Float32Array(len), nc = chans.length;
    for (let i = 0; i < len; i++) {
      let acc = 0; const base = i * factor;
      for (let c = 0; c < nc; c++) { const d = chans[c]; let s = 0; for (let k = 0; k < factor; k++) s += d[base + k]; acc += s; }
      mono[i] = acc / (factor * nc);
    }

    const frames = Math.floor((len - N) / HOP) + 1;
    const fft = makeFFT(N), half = N / 2, binHz = sr / N;
    const win = new Float32Array(N);
    for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
    const bin = hz => Math.max(1, Math.min(half - 2, Math.round(hz / binHz)));
    const kLo = bin(30), kHi = bin(120), bLo = bin(160), bHi = bin(320), nLo = bin(1800), nHi = bin(5500);
    const hHi = bin(10500), hLo = Math.min(bin(6500), hHi - 4);

    const re = new Float32Array(N), im = new Float32Array(N);
    let cur = new Float32Array(half);
    const prevMax = new Float32Array(half);
    const fK = new Float32Array(frames), fB = new Float32Array(frames), fN = new Float32Array(frames), fH = new Float32Array(frames);
    const norm = 0.03;
    const mLo = bin(2000), mHi = bin(6000);
    const levHi = new Float32Array(frames), levMid = new Float32Array(frames);

    for (let f = 0; f < frames; f++) {
      const off = f * HOP;
      for (let i = 0; i < N; i++) { re[i] = mono[off + i] * win[i]; im[i] = 0; }
      fft(re, im);
      for (let b = 0; b < half; b++) cur[b] = Math.log(1 + Math.sqrt(re[b] * re[b] + im[b] * im[b]) * norm);
      { let sh = 0, sm = 0; for (let b = hLo; b <= hHi; b++) sh += cur[b]; for (let b = mLo; b <= mHi; b++) sm += cur[b]; levHi[f] = sh / (hHi - hLo + 1); levMid[f] = sm / (mHi - mLo + 1); }
      if (f > 0) {
        fK[f] = bandFlux(cur, prevMax, kLo, kHi);
        fB[f] = bandFlux(cur, prevMax, bLo, bHi);
        fN[f] = bandFlux(cur, prevMax, nLo, nHi);
        fH[f] = bandFlux(cur, prevMax, hLo, hHi);
      }
      for (let b = 0; b < half; b++) {
        const a = cur[b > 0 ? b - 1 : b], c = cur[b + 1 < half ? b + 1 : b];
        prevMax[b] = Math.max(cur[b], a, c);
      }
      if ((f & 127) === 0) {
        if (onProgress) onProgress(f / frames);
        await new Promise(r => setTimeout(r, 0));
      }
    }
    if (onProgress) onProgress(1);

    // Second, low-rate analysis (2.7 kHz, finer frequency steps) to tell the three toms apart from the kick.
    const ld = Math.floor(len / 8), lowSig = lowBand(mono, len);
    const N2 = 256, HOP2 = 32, half2 = N2 / 2, binHz2 = (sr / 8) / N2;
    const bin2 = hz => Math.max(1, Math.min(half2 - 2, Math.round(hz / binHz2)));
    const fft2 = makeFFT(N2), win2 = new Float32Array(N2);
    for (let i = 0; i < N2; i++) win2[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N2 - 1));
    const re2 = new Float32Array(N2), im2 = new Float32Array(N2), cur2 = new Float32Array(half2), pm2 = new Float32Array(half2);
    const kS = [bin2(30), bin2(80)], tLb = [bin2(85), bin2(140)], tMb = [bin2(145), bin2(215)], tHb = [bin2(220), bin2(340)];
    const rKs = new Float32Array(frames), rTl = new Float32Array(frames), rTm = new Float32Array(frames), rTh = new Float32Array(frames);
    for (let f = 0; f < frames; f++) {
      const off = f * HOP2;
      if (off + N2 > ld) break;
      for (let i = 0; i < N2; i++) { re2[i] = lowSig[off + i] * win2[i]; im2[i] = 0; }
      fft2(re2, im2);
      for (let b = 0; b < half2; b++) cur2[b] = Math.log(1 + Math.sqrt(re2[b] * re2[b] + im2[b] * im2[b]) * 0.12);
      if (f > 0) {
        rKs[f] = bandFlux(cur2, pm2, kS[0], kS[1]);
        rTl[f] = bandFlux(cur2, pm2, tLb[0], tLb[1]);
        rTm[f] = bandFlux(cur2, pm2, tMb[0], tMb[1]);
        rTh[f] = bandFlux(cur2, pm2, tHb[0], tHb[1]);
      }
      for (let b = 0; b < half2; b++) {
        const a = cur2[b > 0 ? b - 1 : b], c = cur2[b + 1 < half2 ? b + 1 : b];
        pm2[b] = Math.max(cur2[b], a, c);
      }
    }
    // Same scale for all the low lanes, taken from the kick, so a song without toms does not get its noise blown up.
    const lowScale = 1 / Math.max(percentile(fK, 0.97), 1e-6);
    for (const a of [rKs, rTl, rTm, rTh]) for (let i = 0; i < frames; i++) a[i] *= lowScale;

    normalize(fK); normalize(fB); normalize(fN); normalize(fH);
    const fS = new Float32Array(frames);
    for (let i = 0; i < frames; i++) fS[i] = Math.sqrt(fB[i] * fN[i]);
    normalize(fS);

    // Take the kick (deep sub-bass, a little late because of its pitch drop) and the snare (noise) out of each tom lane.
    const ksM = maxFilterR(rKs, 3), nzM = maxFilterR(fN, 1);
    const aL = new Float32Array(frames), aM = new Float32Array(frames), aH = new Float32Array(frames);
    for (let i = 0; i < frames; i++) {
      aL[i] = Math.max(0, rTl[i] - 1.0 * ksM[i] - 0.8 * nzM[i]);
      aM[i] = Math.max(0, rTm[i] - 1.0 * ksM[i] - 1.0 * nzM[i]);
      aH[i] = Math.max(0, rTh[i] - 1.0 * ksM[i] - 1.0 * nzM[i]);
    }

    const an = {
      frameDur: HOP / sr,
      bias: 0.4 * N / sr,
      biasLow: 0.4 * 8 * N2 / sr,
      duration: len / sr,
      levHi, levMid,
      env: { kick: fK, snare: fS, hat: fH, tomLo: aL, tomMid: aM, tomHi: aH, kSub: ksM, rTl: rTl },
      med: { kick: localMedian(fK, 43), snare: localMedian(fS, 43), hat: localMedian(fH, 43), tomLo: localMedian(aL, 43), tomMid: localMedian(aM, 43), tomHi: localMedian(aH, 43) }
    };
    const t = estimateTempo(an);
    an.bpm = t.bpm; an.t0 = t.t0; an.bpmRaw = t.bpm0;
    return an;
  }

  // rival: the envelope of a lane that tends to bleed into this one (kick <-> snare).
  // A peak is dropped when the rival is clearly stronger at the same moment.
  function pick(env, med, sens, frameDur, bias, rival, rivalRatio, floorOverride, rl, rh) {
    const n = env.length, d = 0.5 - 0.44 * (sens / 100), floor = floorOverride != null ? floorOverride : 0.9 - 0.75 * (sens / 100), out = [];
    for (let i = 1; i < n - 1; i++) {
      const v = env[i];
      if (v < med[i] + d || v < floor) continue;
      let ok = true;
      for (let j = Math.max(0, i - 3); j <= Math.min(n - 1, i + 3); j++) {
        if (j !== i && (env[j] > v || (env[j] === v && j < i))) { ok = false; break; }
      }
      if (!ok) continue;
      if (rival) {
        let r = 0;
        for (let j = Math.max(0, i + (rl == null ? -2 : rl)); j <= Math.min(n - 1, i + (rh == null ? 2 : rh)); j++) if (rival[j] > r) r = rival[j];
        if (r > (rivalRatio || 1.6) * v) continue;
      }
      const t = bias + i * frameDur, last = out[out.length - 1];
      if (last && t - last.t < 0.055) { if (v > last.v) { last.t = t; last.v = v; } continue; }
      out.push({ t, v });
    }
    return out;
  }

  function estimateTempo(an) {
    const { frameDur, bias } = an, { kick, snare, hat } = an.env, n = kick.length;
    const env = new Float32Array(n);
    for (let i = 0; i < n; i++) env[i] = kick[i] + snare[i] + 0.4 * hat[i];

    const ps = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) ps[i + 1] = ps[i] + env[i];
    const w = Math.round(0.4 / frameDur), x = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const lo = Math.max(0, i - w), hi = Math.min(n, i + w + 1);
      x[i] = Math.max(0, env[i] - (ps[hi] - ps[lo]) / (hi - lo));
    }

    const minLag = Math.max(3, Math.floor(60 / 200 / frameDur));
    const maxLag = Math.min((n >> 1) - 2, Math.ceil(60 / 55 / frameDur));
    const acf = new Float32Array(maxLag + 3);
    for (let lag = minLag - 1; lag <= maxLag + 1; lag++) {
      let s = 0;
      for (let i = 0; i + lag < n; i++) s += x[i] * x[i + lag];
      acf[lag] = s / (n - lag);
    }
    let bestLag = minLag, bestScore = -1;
    for (let lag = minLag; lag <= maxLag; lag++) {
      if (acf[lag] < acf[lag - 1] || acf[lag] < acf[lag + 1]) continue;
      const bpm = 60 / (lag * frameDur);
      const sc = acf[lag] * Math.exp(-0.5 * Math.pow(Math.log2(bpm / 110) / 0.9, 2));
      if (sc > bestScore) { bestScore = sc; bestLag = lag; }
    }
    let lagF = bestLag;
    {
      const y0 = acf[bestLag - 1], y1 = acf[bestLag], y2 = acf[bestLag + 1], den = y0 - 2 * y1 + y2;
      if (den !== 0) lagF += Math.max(-0.5, Math.min(0.5, 0.5 * (y0 - y2) / den));
    }
    const bpm0 = 60 / (lagF * frameDur);

    const envS = maxFilter3(env);
    let bestB = bpm0, bestPh = 0, bestSc = -1;
    for (let bpm = bpm0 * 0.96; bpm <= bpm0 * 1.04; bpm += bpm0 * 0.0005) {
      const P = 60 / bpm / frameDur, nb = Math.floor((n - 2) / P);
      if (nb < 4) continue;
      for (let ph = 0; ph < P; ph += 1) {
        let sc = 0;
        for (let k = 0; k < nb; k++) {
          const i0 = Math.round(ph + k * P), i1 = Math.round(ph + (k + 0.5) * P);
          if (i0 < n) sc += envS[i0];
          if (i1 < n) sc += 0.45 * envS[i1];
        }
        sc /= nb;
        if (sc > bestSc) { bestSc = sc; bestB = bpm; bestPh = ph; }
      }
    }

    const P = 60 / bestB / frameDur, nb = Math.floor((n - 2) / P);
    const kS = maxFilter3(kick), sS = maxFilter3(snare);
    const K = [0, 0, 0, 0], S = [0, 0, 0, 0];
    for (let k = 0; k < nb; k++) {
      const i = Math.round(bestPh + k * P);
      if (i >= n) break;
      K[k % 4] += kS[i]; S[k % 4] += sS[i];
    }
    let bestD = 0, bestDS = -Infinity;
    for (let d = 0; d < 4; d++) {
      const sc = K[d] + 0.5 * K[(d + 2) % 4] + S[(d + 1) % 4] + S[(d + 3) % 4] - 0.5 * S[d] - 0.5 * S[(d + 2) % 4];
      if (sc > bestDS) { bestDS = sc; bestD = d; }
    }
    const beatDur = 60 / bestB;
    return { bpm: bestB, bpm0, t0: bias + bestPh * frameDur + bestD * beatDur };
  }

  function refineT0(hits, step, t0) {
    for (let it = 0; it < 3; it++) {
      let sw = 0, sr = 0;
      for (const h of hits) {
        const x = (h.t - t0) / step, r = x - Math.round(x);
        if (Math.abs(r) < 0.35) { sw += h.w; sr += h.w * r; }
      }
      if (sw > 0) t0 += (sr / sw) * step;
    }
    return t0;
  }

  // A hit in the hi-hat band is a closed hi-hat if the level falls back within about a quarter second,
  // and a ride or crash if it keeps ringing. The louder ringing hits are crashes.
  function classifyHat(an, peaks, sRide, sCrash) {
    const L = an.levHi, fd = an.frameDur, n = L.length, out = { hat: [], ride: [], crash: [] };
    const tRing = 0.25 - 0.2 * (sRide / 100), tSus = 0.2 - 0.2 * (sRide / 100), tCrash = 3.6 - 1.5 * (sCrash / 100);
    // The quiet level of the whole song's high end. A closed hi-hat falls back to it between hits; a ride or crash does not.
    const sortedL = L.slice().sort(), floor = sortedL[Math.floor(n * 0.05)], span = Math.max(sortedL[Math.floor(n * 0.97)] - floor, 1e-9);
    for (const p of peaks) {
      const i = Math.round((p.t - an.bias) / fd);
      let peak = -Infinity, mn = Infinity;
      for (let j = Math.max(0, i); j <= Math.min(n - 1, i + 3); j++) if (L[j] > peak) peak = L[j];
      for (let j = i + 8; j <= Math.min(n - 1, i + 18); j++) if (L[j] < mn) mn = L[j];
      const rise = peak - floor, rm = (mn === Infinity || rise < 0.05 * span) ? 0 : (mn - floor) / rise;
      p.ring = rm; p.peakN = rise / span; p.susN = mn === Infinity ? 0 : (mn - floor) / span;
      if (rm >= tRing && p.susN >= tSus) (p.v >= tCrash ? out.crash : out.ride).push(p); else out.hat.push(p);
    }
    return out;
  }

  // Toms: pitched hits in three low bands. Each lane has already had the kick and snare taken out of it.
  function pickToms(an, sens) {
    const e = an.env, m = an.med, all = [];
    [['tomLo', 'tomLo'], ['tomMid', 'tomMid'], ['tomHi', 'tomHi']].forEach(([k, cls]) => {
      // the tom lanes see a hit about 1.5 frames earlier than the snare lane, so look a little ahead for the rival
      for (const h of pick(e[k], m[k], sens, an.frameDur, an.biasLow, e.snare, 1.0, 1.7 - 1.4 * (sens / 100), -1, 4)) all.push({ t: h.t, v: h.v, cls });
    });
    all.sort((a, b) => a.t - b.t);
    const out = { tomLo: [], tomMid: [], tomHi: [] };
    let grp = [];
    // A real kick has deep sub-bass; a tom does not. That is the one rule that keeps kicks out of the tom lanes.
    const deep = t => { const i = Math.round((t - an.biasLow) / an.frameDur); let m = 0; for (let j = Math.max(0, i - 1); j <= Math.min(e.kSub.length - 1, i + 1); j++) if (e.kSub[j] > m) m = e.kSub[j]; return m; };
    const flush = () => { if (!grp.length) return; let best = grp[0]; for (const g of grp) if (g.v > best.v) best = g; if (deep(best.t) < 0.25) out[best.cls].push({ t: best.t, v: best.v }); grp = []; };
    for (const h of all) { if (grp.length && h.t - grp[grp.length - 1].t > 0.06) flush(); grp.push(h); }
    flush();
    return out;
  }

  // sens: { kick, snare, hat, tom, ride, crash } each 0-100
  function detectAll(an, sens) {
    const e = an.env, m = an.med, out = {};
    const toms = pickToms(an, sens.tom), tomT = [];
    for (const k in toms) for (const h of toms[k]) tomT.push({ t: h.t - an.biasLow + an.bias, v: h.v });
    // time of a tom on the main analysis clock; hits of other drums right on top of a tom are its leakage
    const nearTom = (t, v, ratio) => tomT.some(h => Math.abs(h.t - t) < 0.045 && v < ratio * h.v);
    out.kick = pick(e.kick, m.kick, sens.kick, an.frameDur, an.bias, e.snare).filter(h => !tomT.some(x => Math.abs(x.t - h.t) < 0.045));
    out.snare = pick(e.snare, m.snare, sens.snare, an.frameDur, an.bias, e.kick).filter(h => !nearTom(h.t, h.v, 1.5));
    const hats = pick(e.hat, m.hat, sens.hat, an.frameDur, an.bias).filter(h => !nearTom(h.t, h.v, 1.5));
    const cym = classifyHat(an, hats, sens.ride, sens.crash);
    out.hat = cym.hat; out.ride = cym.ride; out.crash = cym.crash;
    Object.assign(out, toms);
    return out;
  }

  // Song without drums: harmonic/percussive separation. Steady sounds (voice, guitar, keys, bass) show up as long
  // horizontal lines on a spectrogram and drum hits as short vertical ones. Median filters along each direction tell
  // them apart, and a soft mask keeps the steady part. strength is 0-100; a higher value is a harsher cut.
  async function removeDrums(chans, strength, onProgress, sr, hits) {
    const N = 2048, HOP = 1024, K = N / 2 + 1, L = chans[0].length, nc = chans.length;
    const F = Math.floor((L - 1 + HOP) / HOP) + 1;
    const beta = 1 + 4 * (strength / 100);
    const fft = makeFFT(N), win = new Float32Array(N);
    for (let i = 0; i < N; i++) win[i] = Math.sin(Math.PI * (i + 0.5) / N);
    const re = new Float32Array(N), im = new Float32Array(N);
    const pause = () => new Promise(r => setTimeout(r, 0));
    const prog = (a, b, x) => { if (onProgress) onProgress(a + (b - a) * x); };

    // 1. spectrogram of the middle of the stereo picture
    const mag = new Float32Array(F * K);
    for (let f = 0; f < F; f++) {
      const base = f * HOP - HOP;
      for (let i = 0; i < N; i++) {
        const s = base + i; let v = 0;
        if (s >= 0 && s < L) { for (let c = 0; c < nc; c++) v += chans[c][s]; v /= nc; }
        re[i] = v * win[i]; im[i] = 0;
      }
      fft(re, im);
      for (let k = 0; k < K; k++) mag[f * K + k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      if ((f & 63) === 0) { prog(0, 0.12, f / F); await pause(); }
    }
    // 2. steady part: median over 9 frames in time
    const H = new Float32Array(F * K), v = new Float32Array(9);
    for (let f = 0; f < F; f++) {
      for (let k = 0; k < K; k++) {
        for (let j = 0; j < 9; j++) { const ff = Math.min(F - 1, Math.max(0, f - 4 + j)); v[j] = mag[ff * K + k]; }
        for (let a = 1; a < 9; a++) { const x = v[a]; let b = a - 1; while (b >= 0 && v[b] > x) { v[b + 1] = v[b]; b--; } v[b + 1] = x; }
        H[f * K + k] = v[4];
      }
      if ((f & 31) === 0) { prog(0.12, 0.4, f / F); await pause(); }
    }
    // 3. hit part: median over 17 bins in frequency, then the soft mask for the steady part
    const mask = new Uint8Array(F * K), pad = new Float32Array(K + 16), w = new Float32Array(17);
    for (let f = 0; f < F; f++) {
      const o = f * K;
      for (let k = 0; k < K; k++) pad[8 + k] = mag[o + k];
      for (let j = 1; j <= 8; j++) { pad[8 - j] = mag[o + Math.min(K - 1, j)]; pad[8 + K - 1 + j] = mag[o + Math.max(0, K - 1 - j)]; }
      for (let j = 0; j < 17; j++) w[j] = pad[j];
      w.sort();
      for (let k = 0; k < K; k++) {
        const h = H[o + k], p = beta * w[8], hh = h * h;
        mask[o + k] = Math.round(255 * hh / (hh + p * p + 1e-12));
        if (k + 17 < K + 16) {
          const out = pad[k], nv = pad[k + 17];
          let idx = w.indexOf(out); w.copyWithin(idx, idx + 1);
          let j = 16; while (j > 0 && w[j - 1] > nv) { w[j] = w[j - 1]; j--; } w[j] = nv;
        }
      }
      if ((f & 31) === 0) { prog(0.4, 0.7, f / F); await pause(); }
    }
    // 3b. low drums (kick, toms, snare body) are tonal and look "steady", so also duck the low
    // frequencies for a moment after every hit we detected
    if (hits && hits.length && sr) {
      const depth = 0.7 + 0.29 * (strength / 100), dur = 0.3, fr = sr / HOP;
      for (const h of hits) {
        const k0 = Math.max(0, Math.floor(h.f0 * N / sr)), k1 = Math.min(K - 1, Math.ceil(h.f1 * N / sr));
        const fa = Math.max(0, Math.floor((h.t - 0.03) * fr)), fb = Math.min(F - 1, Math.ceil((h.t + dur) * fr));
        for (let f = fa; f <= fb; f++) {
          const dt = f / fr - h.t, g = dt < 0 ? 1 + dt / 0.03 : Math.exp(-dt / h.tau);
          const mulv = 1 - depth * g;
          for (let k = k0; k <= k1; k++) { const i = f * K + k; mask[i] = Math.round(mask[i] * mulv); }
        }
      }
    }
    // 4. apply the mask to each channel and rebuild the audio
    const outs = [];
    for (let c = 0; c < nc; c++) {
      const out = new Float32Array(L), src = chans[c];
      for (let f = 0; f < F; f++) {
        const base = f * HOP - HOP, o = f * K;
        for (let i = 0; i < N; i++) { const s = base + i; re[i] = (s >= 0 && s < L ? src[s] : 0) * win[i]; im[i] = 0; }
        fft(re, im);
        for (let k = 0; k < K; k++) {
          const m = mask[o + k] / 255;
          re[k] *= m; im[k] *= m;
          if (k > 0 && k < K - 1) { re[N - k] *= m; im[N - k] *= m; }
        }
        for (let i = 0; i < N; i++) im[i] = -im[i];
        fft(re, im);
        for (let i = 0; i < N; i++) { const s = base + i; if (s >= 0 && s < L) out[s] += re[i] * win[i] / N; }
        if ((f & 31) === 0) { prog(0.7 + 0.3 * c / nc, 0.7 + 0.3 * (c + 1) / nc, f / F); await pause(); }
      }
      outs.push(out);
    }
    if (onProgress) onProgress(1);
    return outs;
  }

  return { analyzeSamples, pick, refineT0, detectAll, classifyHat, removeDrums };
})();
if (typeof module !== 'undefined') module.exports = DSP;


(function () {
  'use strict';
  const $ = s => document.querySelector(s);
  const IDS = ['kick', 'snare', 'hat', 'tomHi', 'tomMid', 'tomLo', 'ride', 'crash'];
  const BIT = { kick: 1, snare: 2, hat: 4, tomHi: 8, tomMid: 16, tomLo: 32, ride: 64, crash: 128 };
  const FEET = BIT.kick, HANDS = 255 ^ BIT.kick;
  // Top to bottom, the way they sit on the staff.
  const LANES = [
    { id: 'crash', tag: 'CC', name: 'Crash' },
    { id: 'ride', tag: 'RD', name: 'Ride' },
    { id: 'hat', tag: 'HH', name: 'Hi-hat' },
    { id: 'tomHi', tag: 'T1', name: 'High tom' },
    { id: 'tomMid', tag: 'T2', name: 'Mid tom' },
    { id: 'snare', tag: 'SD', name: 'Snare' },
    { id: 'tomLo', tag: 'FT', name: 'Floor tom' },
    { id: 'kick', tag: 'BD', name: 'Kick' }
  ];
  const LN = LANES.length;
  const freshEdits = () => { const o = {}; IDS.forEach(id => { o[id] = { add: [], del: [] }; }); return o; };
  const emptyDet = () => { const o = {}; IDS.forEach(id => { o[id] = []; }); return o; };
  const freshSens = () => ({ kick: 55, snare: 55, hat: 55, tom: 55, ride: 55, crash: 55 });

  const state = {
    name: 'Example groove', demo: true, buffer: null, duration: 0,
    bpm: 96, t0: 0, shift: 0, bpmDetected: 96,
    view: 'sheet', count: true, tapDrum: 'snare', an: null, sens: freshSens(),
    detected: emptyDet(),
    edits: freshEdits(), grid: null, alt: null, altFile: null, hearAlt: false
  };

  /* ---------- helpers ---------- */
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const fmt = t => { t = Math.max(0, t || 0); return Math.floor(t / 60) + ':' + String(Math.floor(t % 60)).padStart(2, '0'); };
  const stepDur = () => 60 / state.bpm / 4;
  const stepTime = s => state.t0 + (s + state.grid.stepOffset) * stepDur();
  const totalDur = () => state.buffer ? state.duration : (state.grid ? Math.max(1, stepTime(state.grid.bars * 16)) : 1);

  /* ---------- grid ---------- */
  function buildGrid() {
    const sd = stepDur(), t0 = state.t0, sets = {};
    for (const id of IDS) sets[id] = new Set();
    for (const id of IDS) {
      for (const h of state.detected[id]) sets[id].add(Math.round((h.t - t0) / sd));
      for (const t of state.edits[id].del) sets[id].delete(Math.round((t - t0) / sd));
      for (const t of state.edits[id].add) sets[id].add(Math.round((t - t0) / sd));
    }
    let min = Infinity, max = -Infinity;
    for (const id of IDS) for (const b of sets[id]) { if (b < min) min = b; if (b > max) max = b; }
    if (min === Infinity) { min = 0; max = 63; }
    const raw = b => b + 64 + state.shift;
    const originBar = Math.floor(raw(min) / 16), lastBar = Math.floor(raw(max) / 16);
    const bars = lastBar - originBar + 1, stepOffset = originBar * 16 - 64 - state.shift;
    const bits = new Uint8Array(bars * 16);
    for (const id of IDS) for (const b of sets[id]) bits[b - stepOffset] |= BIT[id];
    state.grid = { bars, bits, stepOffset };
  }

  let cells = [], barEls = [], rowEls = [], sbs = [];

  /* ----- sheet music -----
     Local y: 0 is the top staff line, SP between lines. Crash on the first ledger line above the staff, hi-hat in the
     space above the staff, ride on the top line. High tom in the top space, mid tom on the next line down, snare in the
     third space, floor tom in the second space, kick in the bottom space. Everything played by hand has stems up, the kick stems down. */
  const SP = 8, TOP = 5 * SP, HDR = 46, PADL = 10, PADR = 8;
  const rowH = () => (state.count ? 15 : 13) * SP;
  const Y = { crash: -1.0 * SP, hat: -0.5 * SP, ride: 0, tomHi: 0.5 * SP, tomMid: 1.0 * SP, snare: 1.5 * SP, tomLo: 2.5 * SP, kick: 3.5 * SP, up: -3.2 * SP, down: 7.2 * SP };
  const HEAD = { crash: 'x', hat: 'x', ride: 'd', tomHi: 'o', tomMid: 'o', tomLo: 'o', snare: 'o', kick: 'o' };
  const HAND_IDS = ['crash', 'hat', 'ride', 'tomHi', 'tomMid', 'snare', 'tomLo'];
  const sheet = { W: 0, bpr: 4 };
  const calcBpr = W => clamp(Math.floor((W - HDR - 6) / 190), 1, 4);

  // Split one beat (4 sixteenths) into notes and rests. A note lasts until the next hit or the end of the beat.
  function beatEvents(p) {
    const on = []; for (let i = 0; i < 4; i++) if (p[i]) on.push(i);
    if (!on.length) return [{ start: 0, dur: 4, rest: true }];
    const ev = [];
    if (on[0] === 3) ev.push({ start: 0, dur: 2, rest: true }, { start: 2, dur: 1, rest: true });
    else if (on[0] > 0) ev.push({ start: 0, dur: on[0], rest: true });
    for (let k = 0; k < on.length; k++) ev.push({ start: on[k], dur: (k + 1 < on.length ? on[k + 1] : 4) - on[k], rest: false });
    return ev;
  }
  const R = (x, y, w, h) => '<rect class="beam" x="' + x.toFixed(2) + '" y="' + y.toFixed(2) + '" width="' + w.toFixed(2) + '" height="' + h + '"/>';

  function drawRest(o, e, x, up, clash) {
    const y = up ? 1.0 * SP : (clash ? 5.1 * SP : 3.4 * SP);
    if (e.dur === 4) {
      o.push('<path class="rest" d="M' + (x - 2) + ' ' + (y - 10) + ' L' + (x + 3) + ' ' + (y - 4) + ' L' + (x - 2) + ' ' + (y + 1) + ' L' + (x + 3) + ' ' + (y + 7) + ' Q' + (x - 3) + ' ' + (y + 6) + ' ' + (x - 2) + ' ' + (y + 12) + '"/>');
    } else if (e.dur === 2) {
      o.push('<circle class="dot" cx="' + (x - 2.5) + '" cy="' + (y - 3) + '" r="2.2"/><path class="rest" d="M' + (x - 2.5) + ' ' + (y - 3) + ' Q' + (x + 1) + ' ' + (y - 3.5) + ' ' + (x + 3.2) + ' ' + (y - 6) + ' L' + (x - 2) + ' ' + (y + 9) + '"/>');
    } else {
      o.push('<circle class="dot" cx="' + (x - 2.5) + '" cy="' + (y - 6) + '" r="2.1"/><circle class="dot" cx="' + (x - 4) + '" cy="' + (y - 0.5) + '" r="2.1"/><path class="rest" d="M' + (x - 2.5) + ' ' + (y - 6) + ' Q' + (x + 1) + ' ' + (y - 6.5) + ' ' + (x + 3.5) + ' ' + (y - 9) + ' L' + (x - 3) + ' ' + (y + 9) + '"/>');
    }
  }
  function drawFlags(o, n, up) {
    const count = n.dur === 1 ? 2 : 1, yT = up ? Y.up : Y.down, d = up ? 1 : -1;
    for (let k = 0; k < count; k++) {
      const y = yT + k * 5 * d;
      o.push('<path class="flag" d="M' + n.sx + ' ' + y + ' C' + n.sx + ' ' + (y + 4 * d) + ' ' + (n.sx + 7) + ' ' + (y + 6 * d) + ' ' + (n.sx + 6) + ' ' + (y + 12 * d) + '"/>');
    }
  }
  function drawBeams(o, grp, up) {
    const th = 3.2, yT = up ? Y.up : Y.down, yP = up ? yT : yT - th, yS = up ? yT + 5 : yT - th - 5;
    o.push(R(grp[0].sx - 0.65, yP, grp[grp.length - 1].sx - grp[0].sx + 1.3, th));
    for (let i = 0; i < grp.length; i++) {
      const n = grp[i]; if (n.dur !== 1) continue;
      const nx = grp[i + 1];
      if (nx && nx.dur === 1) o.push(R(n.sx - 0.65, yS, nx.sx - n.sx + 1.3, th));
      else if (!(i > 0 && grp[i - 1].dur === 1)) o.push(nx ? R(n.sx - 0.65, yS, 6, th) : R(n.sx - 5.35, yS, 6, th));
    }
  }
  function drawHead(o, cx, y, kind) {
    if (kind === 'd') o.push('<path class="head" d="M' + cx + ' ' + (y - 4.6) + ' L' + (cx + 4.2) + ' ' + y + ' L' + cx + ' ' + (y + 4.6) + ' L' + (cx - 4.2) + ' ' + y + ' Z"/>');
    else if (kind === 'x') o.push('<path class="xh" d="M' + (cx - 3.4) + ' ' + (y - 3.4) + ' L' + (cx + 3.4) + ' ' + (y + 3.4) + ' M' + (cx - 3.4) + ' ' + (y + 3.4) + ' L' + (cx + 3.4) + ' ' + (y - 3.4) + '"/>');
    else o.push('<ellipse class="head" cx="' + cx + '" cy="' + y + '" rx="4.5" ry="3.2" transform="rotate(-18 ' + cx + ' ' + y + ')"/>');
  }

  function drawBeat(o, bits, b, beat, x0, stepW, up) {
    const base = b * 16 + beat * 4, mask = up ? HANDS : FEET;
    const ev = beatEvents([0, 1, 2, 3].map(i => (bits[base + i] & mask) !== 0));
    let grp = [];
    const flush = () => { if (grp.length === 1) drawFlags(o, grp[0], up); else if (grp.length > 1) drawBeams(o, grp, up); grp = []; };
    for (const e of ev) {
      const s = base + e.start, local = beat * 4 + e.start;
      if (e.rest) { drawRest(o, e, x0 + (local + e.dur / 2) * stepW, up, !up && [0, 1, 2, 3].some(i => (bits[base + i] & BIT.tomLo) !== 0)); flush(); continue; }
      const cx = x0 + (local + 0.5) * stepW, b0 = bits[s], heads = [];
      if (up) { for (const id of HAND_IDS) if (b0 & BIT[id]) heads.push([Y[id], HEAD[id], id]); } else heads.push([Y.kick, 'o', 'kick']);
      const sx = up ? cx + 4.3 : cx - 4.3, yLow = Math.max.apply(null, heads.map(h => h[0]));
      for (const h of heads) {
        if (h[2] === 'crash') o.push('<line class="ln" x1="' + (cx - 8) + '" x2="' + (cx + 8) + '" y1="' + Y.crash + '" y2="' + Y.crash + '"/>');
        drawHead(o, cx, h[0], h[1]);
        if (e.dur === 3) o.push('<circle class="dot" cx="' + (cx + 8) + '" cy="' + h[0] + '" r="1.5"/>');
      }
      o.push('<line class="stem" x1="' + sx + '" x2="' + sx + '" y1="' + yLow + '" y2="' + (up ? Y.up : Y.down) + '"/>');
      const n = { sx, dur: e.dur };
      if (e.dur === 4) flush(); else grp.push(n);
    }
    flush();
  }

  function rowSVG(r) {
    const g = state.grid, W = sheet.W, bpr = sheet.bpr, b0 = r * bpr, b1 = Math.min(g.bars, b0 + bpr);
    const barW = (W - HDR - 6) / bpr, stepW = (barW - PADL - PADR) / 16, xEnd = HDR + (b1 - b0) * barW;
    const o = ['<svg class="row" viewBox="0 0 ' + W + ' ' + rowH() + '" role="img" aria-label="Drum chart, bars ' + (b0 + 1) + ' to ' + b1 + '"><g class="sg" transform="translate(0,' + TOP + ')">'];
    for (let i = 0; i < 5; i++) o.push('<line class="ln" x1="2" x2="' + xEnd + '" y1="' + i * SP + '" y2="' + i * SP + '"/>');
    o.push('<line class="ln" x1="2" x2="2" y1="0" y2="' + 4 * SP + '"/>');
    o.push('<rect class="clef" x="9" y="' + SP + '" width="3.4" height="' + 2 * SP + '"/><rect class="clef" x="15.5" y="' + SP + '" width="3.4" height="' + 2 * SP + '"/>');
    if (r === 0) o.push('<text class="ts" x="33" y="' + (2 * SP - 1) + '">4</text><text class="ts" x="33" y="' + (4 * SP - 1) + '">4</text>');
    for (let b = b0; b < b1; b++) {
      const bx = HDR + (b - b0) * barW, x0 = bx + PADL;
      for (let j = 0; j < 16; j++) o.push('<rect class="sb" data-s="' + (b * 16 + j) + '" x="' + (x0 + j * stepW).toFixed(2) + '" y="' + (-4 * SP) + '" width="' + stepW.toFixed(2) + '" height="' + (12.4 * SP) + '"/>');
      o.push('<rect data-seek="' + b + '" x="' + (bx + 1) + '" y="' + (-5 * SP) + '" width="26" height="14" fill="transparent" style="cursor:pointer"><title>Play from bar ' + (b + 1) + '</title></rect><text class="bn" x="' + (bx + 4) + '" y="' + (-3.9 * SP) + '">' + (b + 1) + '</text>');
      for (let beat = 0; beat < 4; beat++) { drawBeat(o, g.bits, b, beat, x0, stepW, true); drawBeat(o, g.bits, b, beat, x0, stepW, false); }
      if (state.count) {
        const lab = ['', 'e', '&amp;', 'a'];
        for (let j = 0; j < 16; j++) {
          const isBeat = j % 4 === 0, hit = g.bits[b * 16 + j] !== 0;
          o.push('<text class="cnt' + (isBeat ? ' beat' : '') + (hit ? ' hit' : '') + '" x="' + (x0 + (j + 0.5) * stepW).toFixed(2) + '" y="' + (Y.down + 17) + '">' + (isBeat ? j / 4 + 1 : lab[j % 4]) + '</text>');
        }
      }
      const xr = bx + barW;
      o.push('<line class="ln" x1="' + xr + '" x2="' + xr + '" y1="0" y2="' + 4 * SP + '"/>');
      if (b === g.bars - 1) o.push('<line class="ln" x1="' + (xr - 5) + '" x2="' + (xr - 5) + '" y1="0" y2="' + 4 * SP + '"/><rect class="clef" x="' + (xr - 3) + '" y="0" width="3.2" height="' + 4 * SP + '"/>');
    }
    o.push('</g></svg>');
    return o.join('');
  }

  function renderSheet() {
    const el = $('#chart');
    sheet.W = Math.max(300, Math.round(el.clientWidth)); sheet.bpr = calcBpr(sheet.W);
    const rows = Math.ceil(state.grid.bars / sheet.bpr), out = [];
    for (let r = 0; r < rows; r++) out.push(rowSVG(r));
    el.className = 'chart sheet'; el.innerHTML = out.join('');
    rowEls = el.querySelectorAll('svg.row'); sbs = el.querySelectorAll('.sb');
    lastStep = -1; lastBar = -1;
  }
  function redrawRow(r) {
    const old = rowEls[r]; if (!old) return;
    old.outerHTML = rowSVG(r);
    const el = $('#chart'); rowEls = el.querySelectorAll('svg.row'); sbs = el.querySelectorAll('.sb');
    lastStep = -1;
  }

  function renderChart() { if (state.view === 'sheet') renderSheet(); else renderGrid(); }
  function renderGrid() {
    const g = state.grid, out = [];
    for (let b = 0; b < g.bars; b++) {
      out.push('<section class="bar"><button class="barnum" type="button" data-seek="' + b + '" title="Play from bar ' + (b + 1) + '">' + (b + 1) + '</button>');
      for (const L of LANES) {
        out.push('<div class="lane ' + L.id + '"><span class="lname">' + L.tag + '</span><div class="cells">');
        for (let beat = 0; beat < 4; beat++) {
          out.push('<div class="beat">');
          for (let q = 0; q < 4; q++) {
            const s = b * 16 + beat * 4 + q;
            out.push('<button type="button" tabindex="-1" class="cell' + ((g.bits[s] & BIT[L.id]) ? ' on' : '') + '" data-s="' + s + '" data-l="' + L.id + '"></button>');
          }
          out.push('</div>');
        }
        out.push('</div></div>');
      }
      out.push('</section>');
    }
    const el = $('#chart');
    el.className = 'chart grid'; el.innerHTML = out.join('');
    cells = el.querySelectorAll('.cell');
    barEls = el.children;
    lastStep = -1; lastBar = -1;
  }

  function updateMeta() {
    const g = state.grid, c = {};
    IDS.forEach(id => { c[id] = 0; });
    for (let i = 0; i < g.bits.length; i++) for (const id of IDS) if (g.bits[i] & BIT[id]) c[id]++;
    $('#meta').textContent = state.bpm.toFixed(1) + ' BPM · 4/4 · ' + g.bars + (g.bars === 1 ? ' bar' : ' bars');
    $('#counts').textContent = countText(c);
    return c;
  }
  function countText(c) {
    const toms = c.tomHi + c.tomMid + c.tomLo, parts = [c.kick + ' kicks', c.snare + ' snares', c.hat + ' hi-hats'];
    if (toms) parts.push(toms + ' toms');
    if (c.ride) parts.push(c.ride + ' ride');
    if (c.crash) parts.push(c.crash + (c.crash === 1 ? ' crash' : ' crashes'));
    return parts.join(' · ');
  }

  function updateTab() {
    const g = state.grid, used = {}, out = [state.name + '  |  ' + state.bpm.toFixed(1) + ' BPM  |  4/4', ''];
    for (let i = 0; i < g.bits.length; i++) for (const id of IDS) if (g.bits[i] & BIT[id]) used[id] = true;
    out.push('CC crash, RD ride, HH hi-hat, T1 high tom, T2 mid tom, SD snare, FT floor tom, BD kick', '');
    for (let b0 = 0; b0 < g.bars; b0 += 4) {
      const end = Math.min(g.bars, b0 + 4);
      out.push('Bars ' + (b0 + 1) + '-' + end);
      for (const L of LANES) {
        // hi-hat, snare and kick always get a line; the other drums only when they are played
        if (!['hat', 'snare', 'kick'].includes(L.id) && !used[L.id]) continue;
        const sym = (L.id === 'hat' || L.id === 'ride' || L.id === 'crash') ? 'x' : 'o';
        let row = L.tag + '|';
        for (let b = b0; b < end; b++) {
          for (let j = 0; j < 16; j++) row += (g.bits[b * 16 + j] & BIT[L.id]) ? sym : '-';
          row += '|';
        }
        out.push(row);
      }
      let cnt = '  |';
      for (let b = b0; b < end; b++) cnt += '1e&a2e&a3e&a4e&a|';
      out.push(cnt);
      out.push('');
    }
    $('#tabText').value = out.join('\n');
  }

  function rebuild() {
    buildGrid(); renderChart(); updateMeta(); updateTab();
    if (play.on) play.nextStep = firstStepAtOrAfter(curPos() + 0.25);
    updateTimeUI(play.on ? curPos() : play.pos);
  }

  /* ---------- editing ---------- */
  function toggleCell(id, s) {
    const sd = stepDur(), g = state.grid, base = s + g.stepOffset, ed = state.edits[id];
    const same = t => Math.round((t - state.t0) / sd) === base;
    const present = (g.bits[s] & BIT[id]) !== 0;
    if (present) {
      ed.add = ed.add.filter(t => !same(t));
      if (state.detected[id].some(h => same(h.t))) ed.del.push(state.t0 + base * sd);
      g.bits[s] &= ~BIT[id];
    } else {
      ed.del = ed.del.filter(t => !same(t));
      ed.add.push(state.t0 + base * sd);
      g.bits[s] |= BIT[id];
      if (ctx && ctx.state === 'running') voices[id](ctx.currentTime + 0.01);
    }
    if (state.view === 'sheet') redrawRow(Math.floor(Math.floor(s / 16) / sheet.bpr));
    else {
      const idx = Math.floor(s / 16) * LN * 16 + LANES.findIndex(L => L.id === id) * 16 + (s % 16);
      cells[idx].classList.toggle('on', !present);
    }
    updateMeta(); updateTab();
  }

  /* ---------- audio ---------- */
  let ctx = null, songGain, drumGain, noiseBuf;
  const play = { on: false, pos: 0, startCtx: 0, startPos: 0, src: null, nextStep: 0, timer: 0, raf: 0 };
  let lastStep = -1, lastBar = -1, seeking = false;

  function ensureCtx() {
    if (!ctx) {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      songGain = ctx.createGain(); drumGain = ctx.createGain();
      songGain.connect(ctx.destination); drumGain.connect(ctx.destination);
      const len = ctx.sampleRate; noiseBuf = ctx.createBuffer(1, len, ctx.sampleRate);
      const d = noiseBuf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
      applyVolumes();
    }
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }
  function applyVolumes() {
    if (!ctx) return;
    songGain.gain.value = Math.pow($('#volSong').value / 100, 1.5);
    drumGain.gain.value = Math.pow($('#volDrum').value / 100, 1.5);
  }
  function noiseSrc(t, dur) {
    const s = ctx.createBufferSource(); s.buffer = noiseBuf; s.loop = true;
    s.start(t, Math.random() * 0.5); s.stop(t + dur); return s;
  }
  function tom(freq) {
    return t => {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'sine';
      o.frequency.setValueAtTime(freq * 1.45, t); o.frequency.exponentialRampToValueAtTime(freq, t + 0.07);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.95, t + 0.003); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.42);
      o.connect(g); g.connect(drumGain); o.start(t); o.stop(t + 0.44);
      const n = noiseSrc(t, 0.03), bp = ctx.createBiquadFilter(), ng = ctx.createGain();
      bp.type = 'bandpass'; bp.frequency.value = 1800;
      ng.gain.setValueAtTime(0.3, t); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.02);
      n.connect(bp); bp.connect(ng); ng.connect(drumGain);
    };
  }
  const voices = {
    kick(t) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = 'sine';
      o.frequency.setValueAtTime(165, t); o.frequency.exponentialRampToValueAtTime(46, t + 0.09);
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(1, t + 0.004); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.34);
      o.connect(g); g.connect(drumGain); o.start(t); o.stop(t + 0.36);
      const n = noiseSrc(t, 0.02), hp = ctx.createBiquadFilter(), ng = ctx.createGain();
      hp.type = 'highpass'; hp.frequency.value = 2500;
      ng.gain.setValueAtTime(0.25, t); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.015);
      n.connect(hp); hp.connect(ng); ng.connect(drumGain);
    },
    snare(t) {
      const n = noiseSrc(t, 0.22), hp = ctx.createBiquadFilter(), ng = ctx.createGain();
      hp.type = 'highpass'; hp.frequency.value = 1400;
      ng.gain.setValueAtTime(0.0001, t); ng.gain.exponentialRampToValueAtTime(0.75, t + 0.002); ng.gain.exponentialRampToValueAtTime(0.0001, t + 0.2);
      n.connect(hp); hp.connect(ng); ng.connect(drumGain);
      const o = ctx.createOscillator(), og = ctx.createGain();
      o.type = 'triangle'; o.frequency.setValueAtTime(200, t); o.frequency.exponentialRampToValueAtTime(130, t + 0.08);
      og.gain.setValueAtTime(0.0001, t); og.gain.exponentialRampToValueAtTime(0.55, t + 0.002); og.gain.exponentialRampToValueAtTime(0.0001, t + 0.12);
      o.connect(og); og.connect(drumGain); o.start(t); o.stop(t + 0.14);
    },
    hat(t) {
      const n = noiseSrc(t, 0.06), hp = ctx.createBiquadFilter(), g = ctx.createGain();
      hp.type = 'highpass'; hp.frequency.value = 7500;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.4, t + 0.001); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.05);
      n.connect(hp); hp.connect(g); g.connect(drumGain);
    },
    tomHi: null, tomMid: null, tomLo: null,
    ride(t) {
      const n = noiseSrc(t, 1.1), hp = ctx.createBiquadFilter(), g = ctx.createGain();
      hp.type = 'highpass'; hp.frequency.value = 6000;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.28, t + 0.002); g.gain.exponentialRampToValueAtTime(0.05, t + 0.18); g.gain.exponentialRampToValueAtTime(0.0001, t + 1.0);
      n.connect(hp); hp.connect(g); g.connect(drumGain);
      [3150, 4720, 6310].forEach((f, i) => {
        const o = ctx.createOscillator(), og = ctx.createGain();
        o.type = 'square'; o.frequency.value = f;
        og.gain.setValueAtTime(0.0001, t); og.gain.exponentialRampToValueAtTime(0.035 - i * 0.008, t + 0.002); og.gain.exponentialRampToValueAtTime(0.0001, t + 0.35);
        const bp = ctx.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = f; bp.Q.value = 4;
        o.connect(bp); bp.connect(og); og.connect(drumGain); o.start(t); o.stop(t + 0.4);
      });
    },
    crash(t) {
      const n = noiseSrc(t, 1.8), hp = ctx.createBiquadFilter(), g = ctx.createGain();
      hp.type = 'highpass'; hp.frequency.value = 3200;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.55, t + 0.003); g.gain.exponentialRampToValueAtTime(0.12, t + 0.35); g.gain.exponentialRampToValueAtTime(0.0001, t + 1.7);
      n.connect(hp); hp.connect(g); g.connect(drumGain);
    }
  };
  voices.tomHi = tom(210); voices.tomMid = tom(150); voices.tomLo = tom(105);

  function curPos() {
    if (!play.on) return play.pos;
    return clamp(play.startPos + ctx.currentTime - play.startCtx, play.startPos, totalDur());
  }
  function firstStepAtOrAfter(pos) {
    return Math.max(0, Math.ceil((pos - state.t0) / stepDur()) - state.grid.stepOffset);
  }
  function schedule() {
    if (!play.on) return;
    const g = state.grid, now = play.startPos + (ctx.currentTime - play.startCtx), horizon = now + 0.2;
    const n = g.bars * 16;
    while (play.nextStep < n) {
      const t = stepTime(play.nextStep);
      if (t > horizon) break;
      const bits = g.bits[play.nextStep];
      if (bits) {
        const when = play.startCtx + (t - play.startPos);
        if (when >= ctx.currentTime - 0.01) {
          for (const id of IDS) if (bits & BIT[id]) voices[id](when);
        }
      }
      play.nextStep++;
    }
    if (now >= totalDur()) finish();
  }
  function stopSources() {
    const s = play.src; play.src = null;
    if (s) { try { s.stop(); } catch (e) { /* already stopped */ } try { s.disconnect(); } catch (e) { /* ignore */ } }
  }
  function startPlay(pos) {
    ensureCtx(); stopSources(); clearInterval(play.timer); cancelAnimationFrame(play.raf);
    pos = clamp(pos, 0, totalDur() - 0.05);
    play.startCtx = ctx.currentTime + 0.06; play.startPos = pos; play.pos = pos;
    if (state.buffer) {
      const src = ctx.createBufferSource(); src.buffer = (state.hearAlt && state.alt) ? state.alt : state.buffer; src.connect(songGain);
      src.start(play.startCtx, pos); src.onended = () => { if (play.src === src && play.on) finish(); };
      play.src = src;
    }
    play.nextStep = firstStepAtOrAfter(pos);
    play.on = true;
    play.timer = setInterval(schedule, 25);
    play.raf = requestAnimationFrame(frame);
    $('#play').textContent = 'Pause';
  }
  function stopPlay() {
    if (play.on) play.pos = curPos();
    play.on = false; stopSources(); clearInterval(play.timer); cancelAnimationFrame(play.raf);
    $('#play').textContent = 'Play';
    updateTimeUI(play.pos);
  }
  function finish() {
    stopPlay(); play.pos = 0; updateTimeUI(0); setNow(-1);
  }
  function frame() {
    if (!play.on) return;
    const pos = curPos();
    updateTimeUI(pos);
    setNow(Math.floor((pos - state.t0) / stepDur()) - state.grid.stepOffset);
    play.raf = requestAnimationFrame(frame);
  }
  function updateTimeUI(pos) {
    const dur = totalDur();
    $('#time').textContent = fmt(pos) + ' / ' + fmt(dur);
    if (!seeking) $('#seek').value = Math.round(clamp(pos / dur, 0, 1) * 1000);
  }
  function nowEls(s) {
    if (state.view === 'sheet') return sbs[s] ? [sbs[s]] : [];
    const b = Math.floor(s / 16), j = s % 16;
    return LANES.map((L, l) => cells[b * LN * 16 + l * 16 + j]).filter(Boolean);
  }
  function setNow(s) {
    if (s === lastStep) return;
    const n = state.grid.bars * 16;
    if (lastStep >= 0 && lastStep < n) nowEls(lastStep).forEach(c => c.classList.remove('now'));
    lastStep = s;
    if (s < 0 || s >= n) return;
    nowEls(s).forEach(c => c.classList.add('now'));
    const b = Math.floor(s / 16);
    if (b !== lastBar) {
      let target;
      if (state.view === 'sheet') target = rowEls[Math.floor(b / sheet.bpr)];
      else {
        if (lastBar >= 0 && barEls[lastBar]) barEls[lastBar].classList.remove('cur');
        target = barEls[b]; target.classList.add('cur');
      }
      if (target) target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      lastBar = b;
    }
  }

  /* ---------- demo ---------- */
  function loadDemo() {
    stopPlay(); play.pos = 0;
    Object.assign(state, { name: 'Example groove', demo: true, buffer: null, duration: 0, an: null, bpm: 96, t0: 0, shift: 0 });
    state.detected = emptyDet(); state.edits = freshEdits();
    const sd = 60 / 96 / 4, add = (id, s) => state.edits[id].add.push(s * sd);
    for (let bar = 0; bar < 4; bar++) {
      const o = bar * 16;
      const hatEnd = bar === 3 ? 12 : 16;
      for (let i = 0; i < hatEnd; i += 2) add(bar === 2 ? 'ride' : 'hat', o + i);
      add('kick', o); add('kick', o + (bar === 3 ? 8 : 10));
      if (bar === 1) add('kick', o + 7);
      add('snare', o + 4);
      if (bar === 3) { add('tomHi', o + 12); add('tomMid', o + 13); add('tomLo', o + 14); add('tomLo', o + 15); } else add('snare', o + 12);
    }
    add('crash', 0);
    $('#tune').hidden = true;
    $('#chartTitle').textContent = 'Example groove';
    $('#volSong').disabled = true;
    updateNote();
    rebuild();
  }

  /* ---------- analysis ---------- */
  function setStatus(msg, err) { const s = $('#status'); s.textContent = msg; s.classList.toggle('err', !!err); }
  function setProgress(p) { $('#prog').hidden = p == null; if (p != null) $('#pfill').style.width = Math.round(p * 100) + '%'; }

  function hitsForRefine() {
    const w = { kick: 1, snare: 1, hat: 0.4, ride: 0.4, crash: 0.4, tomHi: 0.8, tomMid: 0.8, tomLo: 0.8 }, out = [];
    for (const id of IDS) for (const h of state.detected[id]) out.push({ t: h.t, w: h.v * w[id] });
    return out;
  }
  function redetect() { state.detected = Object.assign(emptyDet(), DSP.detectAll(state.an, state.sens)); }

  async function handleFile(file) {
    if (!file) return;
    stopPlay();
    $('#pick').disabled = true;
    setStatus('Reading ' + file.name + '…'); setProgress(0);
    try {
      ensureCtx();
      const ab = await file.arrayBuffer();
      setStatus('Decoding audio…');
      const buf = await new Promise((res, rej) => { const p = ctx.decodeAudioData(ab, res, rej); if (p && p.catch) p.catch(rej); });
      if (buf.duration > 900) throw new Error('That file is longer than 15 minutes. Try a single song.');
      setStatus('Listening for drums…');
      const chans = []; for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
      const an = await DSP.analyzeSamples(chans, buf.sampleRate, p => setProgress(p));

      Object.assign(state, {
        name: file.name.replace(/\.[^.]+$/, ''), demo: false, buffer: buf, duration: buf.duration,
        an, bpm: an.bpm, bpmDetected: an.bpm, t0: an.t0, shift: 0
      });
      state.sens = freshSens();
      state.edits = freshEdits();
      resetNoDrums(); $('#nodrums').hidden = false;
      redetect();
      state.t0 = DSP.refineT0(hitsForRefine(), stepDur(), state.t0);
      play.pos = 0;

      ['#sensKick', '#sensSnare', '#sensHat', '#sensTom', '#sensRide', '#sensCrash'].forEach(s => { $(s).value = 55; });
      $('#bpm').value = state.bpm.toFixed(1);
      $('#bpmHint').textContent = 'Detected ' + state.bpmDetected.toFixed(1) + ' BPM. If the chart looks twice as busy or half as busy as the song, use ½× or 2×.';
      $('#tune').hidden = false;
      $('#volSong').disabled = false;
      $('#chartTitle').textContent = state.name;
      updateNote();
      rebuild();
      const c = updateMeta();
      setStatus('Done. Found ' + countText(c) + ' at ' + state.bpm.toFixed(1) + ' BPM.');
      $('#transport').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      const msg = (e && e.message && !/decode|EncodingError|unable/i.test(e.message)) ? e.message : 'This browser could not read that file. Try an MP3, WAV or M4A.';
      setStatus(msg, true);
    } finally {
      setProgress(null);
      $('#pick').disabled = false;
    }
  }

  /* ---------- wiring ---------- */
  $('#pick').addEventListener('click', () => $('#file').click());
  $('#file').addEventListener('change', e => { const f = e.target.files[0]; e.target.value = ''; handleFile(f); });
  const drop = $('#drop');
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', e => { const f = e.dataTransfer && e.dataTransfer.files[0]; if (f) handleFile(f); });
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => e.preventDefault());

  $('#play').addEventListener('click', () => { if (play.on) stopPlay(); else startPlay(play.pos >= totalDur() - 0.1 ? 0 : play.pos); });
  window.addEventListener('keydown', e => {
    if (e.code === 'Space' && !/INPUT|TEXTAREA|BUTTON|SUMMARY/.test((e.target.tagName || ''))) { e.preventDefault(); $('#play').click(); }
  });
  $('#seek').addEventListener('input', () => { seeking = true; $('#time').textContent = fmt($('#seek').value / 1000 * totalDur()) + ' / ' + fmt(totalDur()); });
  $('#seek').addEventListener('change', () => {
    const pos = $('#seek').value / 1000 * totalDur(); seeking = false;
    if (play.on) startPlay(pos); else { play.pos = pos; updateTimeUI(pos); }
  });
  $('#volSong').addEventListener('input', applyVolumes);
  $('#volDrum').addEventListener('input', applyVolumes);

  $('#chart').addEventListener('click', e => {
    const sk = e.target.closest('[data-seek]');
    if (sk) { startPlay(Math.max(0, stepTime(+sk.dataset.seek * 16))); return; }
    const sb = e.target.closest('.sb');
    if (sb) { toggleCell(state.tapDrum, +sb.dataset.s); return; }
    const cell = e.target.closest('.cell');
    if (cell) toggleCell(cell.dataset.l, +cell.dataset.s);
  });

  function updateNote() {
    const how = state.view === 'sheet'
      ? 'Pick a drum below, then tap the staff at the beat where it should be added or removed. '
      : 'Tap any cell to add or remove a note. ';
    $('#chartNote').textContent = (state.demo
      ? 'This is an example groove so you can see the chart. Press play, then drop in your own song above. '
      : 'Press play to hear the drum chart over your song. ') + how + 'Tap a bar number to play from there.';
  }
  document.querySelectorAll('.segb').forEach(btn => btn.addEventListener('click', () => {
    if (state.view === btn.dataset.view) return;
    state.view = btn.dataset.view;
    document.querySelectorAll('.segb').forEach(x => x.setAttribute('aria-pressed', x === btn ? 'true' : 'false'));
    $('#legendSheet').hidden = state.view !== 'sheet'; $('#legendGrid').hidden = state.view === 'sheet';
    $('#tapbar').hidden = state.view !== 'sheet';
    updateNote(); renderChart();
  }));
  document.querySelectorAll('.chip').forEach(ch => ch.addEventListener('click', () => {
    state.tapDrum = ch.dataset.drum;
    document.querySelectorAll('.chip').forEach(x => x.setAttribute('aria-pressed', x === ch ? 'true' : 'false'));
  }));
  $('#showCount').addEventListener('change', e => { state.count = e.target.checked; renderChart(); });
  /* ---------- downloads ---------- */
  const EXP_W = 700, EXP_BPR = 2;
  const EXP_CSS = '.ln{stroke:#111;stroke-width:1}.stem{stroke:#111;stroke-width:1.3}.head,.dot,.beam,.clef{fill:#111}' +
    '.xh{stroke:#111;stroke-width:1.7;stroke-linecap:round}.flag{fill:none;stroke:#111;stroke-width:1.8;stroke-linecap:round}' +
    '.rest{fill:none;stroke:#111;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}' +
    '.bn{font:10px "Courier New",monospace;fill:#666}.ts{font:bold 21px Arial,sans-serif;fill:#111;text-anchor:middle}' +
    '.sb{display:none}.cnt{font:9.5px "Courier New",monospace;fill:#777;text-anchor:middle}' +
    '.cnt.beat{font-weight:bold;fill:#111}.cnt.hit{fill:#111}';
  const esc = s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const exportRows = () => Math.ceil(state.grid.bars / EXP_BPR);

  function exportSVG(r0, r1, withHeader) {
    const keep = { W: sheet.W, bpr: sheet.bpr };
    sheet.W = EXP_W; sheet.bpr = EXP_BPR;
    const rh = rowH(), gap = 14, head = withHeader ? 90 : 0, h = head + (r1 - r0) * (rh + gap) + 8;
    const o = ['<svg xmlns="http://www.w3.org/2000/svg" width="' + EXP_W + '" height="' + h + '" viewBox="0 0 ' + EXP_W + ' ' + h + '"><style>' + EXP_CSS + '</style><rect width="' + EXP_W + '" height="' + h + '" fill="#fff"/>'];
    if (withHeader) {
      o.push('<text x="8" y="34" style="font:bold 28px Arial,sans-serif;fill:#111">' + esc(state.name) + '</text>');
      o.push('<text x="8" y="56" style="font:13px Arial,sans-serif;fill:#555">' + state.bpm.toFixed(1) + ' BPM, 4/4. Crash: x on the line above the staff. Hi-hat: x above the staff. Ride: diamond on the top line.</text>');
      o.push('<text x="8" y="71" style="font:13px Arial,sans-serif;fill:#555">Toms: high in the top space, mid on the next line, floor in the second space. Snare: middle space. Kick: bottom space.</text>');
    }
    for (let r = r0; r < r1; r++) {
      const y = head + (r - r0) * (rh + gap);
      o.push(rowSVG(r).replace('<svg class="row" viewBox', '<svg x="0" y="' + y + '" width="' + EXP_W + '" height="' + rh + '" viewBox'));
    }
    sheet.W = keep.W; sheet.bpr = keep.bpr;
    o.push('</svg>');
    return { svg: o.join(''), w: EXP_W, h };
  }
  function rasterize(s, scale) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas'); c.width = Math.round(s.w * scale); c.height = Math.round(s.h * scale);
        const x = c.getContext('2d'); x.fillStyle = '#fff'; x.fillRect(0, 0, c.width, c.height);
        x.drawImage(img, 0, 0, c.width, c.height); res(c);
      };
      img.onerror = () => rej(new Error('Could not draw the chart.'));
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(s.svg);
    });
  }
  async function makePDF() {
    const doc = new window.jspdf.jsPDF({ unit: 'pt', format: 'letter', compress: true });
    const unit = rowH() + 14, k = 540 / EXP_W, rows = exportRows();
    const per1 = Math.max(1, Math.floor((700 / k - 98) / unit)), perN = Math.max(1, Math.floor((700 / k - 8) / unit));
    let r = 0, page = 0;
    while (r < rows) {
      const end = Math.min(rows, r + (page === 0 ? per1 : perN));
      const s = exportSVG(r, end, page === 0), c = await rasterize(s, 2);
      if (page > 0) doc.addPage();
      doc.addImage(c.toDataURL('image/png'), 'PNG', 36, 36, 540, s.h * k, undefined, 'FAST');
      r = end; page++;
    }
    const n = doc.getNumberOfPages();
    for (let i = 1; i <= n; i++) { doc.setPage(i); doc.setFontSize(8); doc.setTextColor(120); doc.text('Made with Drum Scribe, page ' + i + ' of ' + n, 306, 772, { align: 'center' }); }
    return doc.output('blob');
  }
  // ---- song without drums ----
  function resetNoDrums() {
    state.alt = null; state.altFile = null; state.hearAlt = false;
    $('#ndHear').checked = false; $('#ndDone').hidden = true; $('#ndMsg').textContent = ''; $('#ndProg').hidden = true;
  }
  async function resample48(chs, sr) {
    if (sr === 48000) return chs;
    const len = chs[0].length, n2 = Math.ceil(len * 48000 / sr);
    const oc = new OfflineAudioContext(chs.length, n2, 48000);
    const b = oc.createBuffer(chs.length, len, sr); chs.forEach((c, i) => b.copyToChannel(c, i));
    const s = oc.createBufferSource(); s.buffer = b; s.connect(oc.destination); s.start();
    const r = await oc.startRendering();
    return chs.map((_, i) => new Float32Array(r.getChannelData(i)));
  }
  async function encodeWebm(chs, sr, onProg) {
    if (!(window.AudioEncoder && window.AudioData && window.WebMMuxer)) throw new Error('nocodec');
    const cfg = { codec: 'opus', sampleRate: 48000, numberOfChannels: Math.min(2, chs.length), bitrate: 160000 };
    if (AudioEncoder.isConfigSupported) { const ok = await AudioEncoder.isConfigSupported(cfg); if (!ok || !ok.supported) throw new Error('nocodec'); }
    chs = chs.slice(0, 2); const nc = chs.length;
    const d = await resample48(chs, sr), total = d[0].length;
    const target = new WebMMuxer.ArrayBufferTarget();
    const muxer = new WebMMuxer.Muxer({ target, audio: { codec: 'A_OPUS', numberOfChannels: nc, sampleRate: 48000 } });
    let err = null;
    const enc = new AudioEncoder({ output: (c, m) => muxer.addAudioChunk(c, m), error: e => { err = e; } });
    enc.configure(cfg);
    const CH = 48000;
    for (let o = 0; o < total; o += CH) {
      if (err) throw err;
      const n = Math.min(CH, total - o), pl = new Float32Array(n * nc);
      for (let c = 0; c < nc; c++) pl.set(d[c].subarray(o, o + n), c * n);
      const ad = new AudioData({ format: 'f32-planar', sampleRate: 48000, numberOfFrames: n, numberOfChannels: nc, timestamp: Math.round(o / 48000 * 1e6), data: pl });
      enc.encode(ad); ad.close();
      while (enc.encodeQueueSize > 6) await new Promise(r => setTimeout(r, 5));
      onProg(o / total);
      await new Promise(r => setTimeout(r, 0));
    }
    await enc.flush(); if (err) throw err;
    muxer.finalize();
    return { blob: new Blob([target.buffer], { type: 'audio/webm' }), ext: 'webm' };
  }
  async function recordBuffer(buf, onProg) {
    const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
    const mt = window.MediaRecorder && types.find(t => MediaRecorder.isTypeSupported(t));
    if (!mt) throw new Error('norec');
    ensureCtx();
    const dest = ctx.createMediaStreamDestination(), rec = new MediaRecorder(dest.stream, { mimeType: mt }), parts = [];
    rec.ondataavailable = e => { if (e.data && e.data.size) parts.push(e.data); };
    const stopped = new Promise(r => { rec.onstop = r; });
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(dest);
    const ended = new Promise(r => { src.onended = r; });
    rec.start(1000); src.start(); const t0 = ctx.currentTime;
    const iv = setInterval(() => onProg(Math.min(1, (ctx.currentTime - t0) / buf.duration)), 250);
    await ended; clearInterval(iv); rec.stop(); await stopped;
    return { blob: new Blob(parts, { type: mt }), ext: mt.indexOf('mp4') >= 0 ? 'mp4' : 'webm' };
  }
  async function makeNoDrums() {
    if (!state.buffer || state.demo) return;
    const btn = $('#ndMake'), msg = $('#ndMsg'), fill = $('#ndFill');
    const setP = p => { $('#ndProg').hidden = p == null; if (p != null) fill.style.width = Math.round(p * 100) + '%'; };
    btn.disabled = true; $('#ndDone').hidden = true; state.altFile = null;
    stopPlay(); state.hearAlt = false; $('#ndHear').checked = false;
    const buf = state.buffer, sr = buf.sampleRate;
    try {
      msg.textContent = 'Taking the drums out…'; setP(0);
      const chans = []; for (let c = 0; c < buf.numberOfChannels; c++) chans.push(buf.getChannelData(c));
      const hits = [], D = state.detected;
      for (const h of D.kick) hits.push({ t: h.t, f0: 20, f1: 260, tau: 0.14 });
      for (const id of ['tomHi', 'tomMid', 'tomLo']) for (const h of D[id]) hits.push({ t: h.t, f0: 60, f1: 340, tau: 0.16 });
      for (const h of D.snare) hits.push({ t: h.t, f0: 120, f1: 300, tau: 0.05 });
      const strength = +$('#ndStrength').value;
      const outs = await DSP.removeDrums(chans, strength, p => setP(p * 0.7), sr, hits);
      // keep the result from clipping
      let pk = 0; for (const o of outs) for (let i = 0; i < o.length; i += 7) { const a = Math.abs(o[i]); if (a > pk) pk = a; }
      if (pk > 0.98) { const g = 0.98 / pk; for (const o of outs) for (let i = 0; i < o.length; i++) o[i] *= g; }
      ensureCtx();
      const alt = ctx.createBuffer(outs.length, outs[0].length, sr); outs.forEach((o, i) => alt.copyToChannel(o, i));
      state.alt = alt;
      let file = null;
      if (dlNs) {
        msg.textContent = 'Making the file…';
        try { file = await encodeWebm(outs, sr, p => setP(0.7 + 0.3 * p)); }
        catch (e) {
          if (!/nocodec/.test(String(e && e.message))) throw e;
          msg.textContent = 'Making the file. This browser does it in real time, so keep this tab open for ' + Math.ceil(buf.duration / 60 * 10) / 10 + ' min…';
          file = await recordBuffer(alt, p => setP(0.7 + 0.3 * p));
        }
      }
      state.altFile = file;
      $('#ndDone').hidden = false; $('#ndSave').hidden = !file;
      msg.textContent = file ? 'Ready. Listen first, then download.' : 'Ready to listen to. Saving a file is not available here.';
    } catch (e) {
      state.alt = null;
      msg.textContent = 'Could not make that. Try again, or try a shorter song.';
    } finally { setP(null); btn.disabled = false; }
  }
  $('#ndMake').addEventListener('click', makeNoDrums);
  $('#ndHear').addEventListener('change', () => {
    state.hearAlt = $('#ndHear').checked && !!state.alt;
    if (play.on) startPlay(curPos());
  });
  $('#ndSave').addEventListener('click', async () => {
    const msg = $('#ndMsg'); if (!state.altFile || !dlNs) return;
    try {
      const base = (state.name.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '').trim() || 'Song') + ' (no drums)';
      await dlNs.save({ filename: base + '.' + state.altFile.ext, data: state.altFile.blob });
      msg.textContent = 'Saved. It is a .' + state.altFile.ext + ' audio file (plays in VLC, Chrome and Firefox).';
    } catch (e) {
      msg.textContent = e && e.code === 'declined' ? '' : 'Could not save that file.';
    }
  });

  let dlNs = null;
  async function saveAs(kind) {
    const msg = $('#dlMsg'); msg.textContent = 'Preparing…';
    try {
      let data, ext;
      if (kind === 'pdf') { data = await makePDF(); ext = 'pdf'; }
      else if (kind === 'png') {
        const s = exportSVG(0, exportRows(), true), c = await rasterize(s, s.h * 3 > 14000 ? 2 : 3);
        data = await new Promise(r => c.toBlob(r, 'image/png')); ext = 'png';
      } else { data = $('#tabText').value; ext = 'txt'; }
      const base = (state.name.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, '').trim() || 'Drum notes') + ' - drum notes';
      await dlNs.save({ filename: base + '.' + ext, data });
      msg.textContent = 'Saved.';
    } catch (e) {
      msg.textContent = e && e.code === 'declined' ? '' : 'Could not save that file. Try another format.';
    }
  }
  $('#dlPdf').addEventListener('click', () => saveAs('pdf'));
  $('#dlPng').addEventListener('click', () => saveAs('png'));
  $('#dlTxt').addEventListener('click', () => saveAs('txt'));
  if (window.claude && window.claude.use) {
    window.claude.use('downloads').then(ns => {
      if (!ns) return;
      dlNs = ns; $('#dl').hidden = false;
      if (!(window.jspdf && window.jspdf.jsPDF)) $('#dlPdf').hidden = true;
    }).catch(() => {});
  }

  let rzTimer = 0;
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      clearTimeout(rzTimer);
      rzTimer = setTimeout(() => {
        if (state.view !== 'sheet') return;
        const W = Math.max(300, Math.round($('#chart').clientWidth));
        if (calcBpr(W) !== sheet.bpr || Math.abs(W - sheet.W) > 80) renderChart();
      }, 150);
    }).observe($('#chart'));
  }

  function setBpm(v) {
    v = clamp(v, 40, 260);
    state.bpm = v; $('#bpm').value = v.toFixed(1);
    state.t0 = DSP.refineT0(hitsForRefine(), stepDur(), state.t0);
    rebuild();
  }
  $('#bpm').addEventListener('change', e => { const v = parseFloat(e.target.value); if (v > 0) setBpm(v); else e.target.value = state.bpm.toFixed(1); });
  $('#halve').addEventListener('click', () => setBpm(state.bpm / 2));
  $('#double').addEventListener('click', () => setBpm(state.bpm * 2));
  $('#barEarly').addEventListener('click', () => { state.shift = clamp(state.shift + 4, -12, 12); rebuild(); });
  $('#barLate').addEventListener('click', () => { state.shift = clamp(state.shift - 4, -12, 12); rebuild(); });
  $('#nudgeL').addEventListener('click', () => { state.t0 -= 0.01; rebuild(); });
  $('#nudgeR').addEventListener('click', () => { state.t0 += 0.01; rebuild(); });
  $('#resetEdits').addEventListener('click', () => { state.edits = freshEdits(); rebuild(); });

  let sensTimer = 0;
  [['#sensKick', 'kick'], ['#sensSnare', 'snare'], ['#sensHat', 'hat'], ['#sensTom', 'tom'], ['#sensRide', 'ride'], ['#sensCrash', 'crash']].forEach(([sel, id]) => {
    $(sel).addEventListener('input', e => {
      state.sens[id] = +e.target.value;
      clearTimeout(sensTimer);
      sensTimer = setTimeout(() => { redetect(); rebuild(); }, 150);
    });
  });

  $('#copyTab').addEventListener('click', async () => {
    const ta = $('#tabText'), btn = $('#copyTab');
    let msg = 'Copied';
    try { await navigator.clipboard.writeText(ta.value); } catch (e) { ta.focus(); ta.select(); msg = 'Selected, press Ctrl+C'; }
    btn.textContent = msg; setTimeout(() => { btn.textContent = 'Copy text'; }, 1800);
  });

  loadDemo();
})();
